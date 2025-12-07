import os
import re
import json
import hashlib
import subprocess
import requests
from datetime import datetime
from collections.abc import Sequence

from pydantic import Field

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Action,
    Observation,
    TextContent,
    ImageContent,
    ToolDefinition,
)
from openhands.sdk.tool import Tool, ToolExecutor, register_tool

# =============================================================================
# Config & globals
# =============================================================================

# Only this file is writable by the agent
ALLOWED_WRITE_PATHS = {"our_train_gpt.py"}

# Command used to run the modded-nanogpt benchmark
BENCHMARK_CMD = [
    "torchrun",
    "--standalone",
    "--nproc_per_node=2",
    "our_train_gpt.py",
]

# Timeout for a single benchmark run (seconds)
BENCHMARK_TIMEOUT = int(
    float(os.getenv("BENCHMARK_TIMEOUT_SECONDS", str(60 * 60)))  # default 1 hour
)

# Where to store benchmark history (JSONL, one JSON record per line)
RUN_HISTORY_FILE_NAME = "agent_logs/benchmark_runs.jsonl"

# Tavily usage limits (to avoid burning all credits)
TAVILY_MAX_CALLS = int(os.getenv("TAVILY_MAX_CALLS", "20"))
_tavily_calls_used = 0


# =============================================================================
# Helper functions
# =============================================================================

def _is_path_allowed(workspace_root: str, path: str) -> bool:
    """
    Return True iff this path is allowed to be written by the agent.

    Currently we allow exactly 'our_train_gpt.py' in the repo root.
    """
    abs_path = os.path.abspath(os.path.join(workspace_root, path))
    rel = os.path.relpath(abs_path, workspace_root)
    normalized = rel.replace(os.sep, "/")
    return normalized in ALLOWED_WRITE_PATHS


def _parse_benchmark_score(stdout: str) -> tuple[float, float] | None:
    """
    Parse the benchmark metrics from stdout.

    We expect a line like:
        BENCH_RESULT time_seconds=123.456 best_val_loss=3.2790

    Returns:
        (time_seconds, best_val_loss) if found, else None.

    Interpretation:
        - best_val_loss MUST be <= 3.28 to be considered a valid run.
        - Among runs that meet the loss requirement, smaller time_seconds is better.
    """
    pattern = r"BENCH_RESULT\s+time_seconds=([0-9.]+)\s+best_val_loss=([0-9.]+)"
    m = re.search(pattern, stdout)
    if not m:
        return None
    time_s = float(m.group(1))
    best_val = float(m.group(2))
    return (time_s, best_val)


def _parse_hp_config(stdout: str) -> dict:
    """
    Parse hyperparameter config from our_train_gpt.py output.

    Preferred format (new):
        HP_CONFIG_JSON { ...json... }

    where the JSON object looks like, e.g.:

        {
            "hyperparameters": { ...from asdict(args)... },
            "optimizer1": { ... },
            "optimizer2": { ... },
            ...
        }

    Returns:
        A dict with that JSON object, or {} if nothing is found.

    Fallback:
        Supports a legacy "HP_CONFIG key=value ..." format if ever used.
    """
    # First, look for HP_CONFIG_JSON line(s)
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("HP_CONFIG_JSON"):
            continue
        # Everything after the token should be JSON
        json_part = line[len("HP_CONFIG_JSON"):].strip()
        if not json_part:
            continue
        try:
            return json.loads(json_part)
        except json.JSONDecodeError:
            continue

    # Fallback: legacy key=value style
    config = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("HP_CONFIG"):
            continue
        rest = line[len("HP_CONFIG"):].strip()
        parts = rest.split()
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            config[k.strip()] = v.strip()
        break

    return config


def _sha256_of_file(path: str) -> str | None:
    """
    Compute SHA-256 digest of a file, or return None if it doesn't exist.
    """
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_benchmark_record(
    workspace_root: str,
    time_seconds: float | None,
    best_val_loss: float | None,
    note: str | None,
    stdout: str,
) -> None:
    """
    Append a benchmark run record to JSONL history under agent_logs/.

    Each record includes:
      - timestamp
      - time_seconds
      - best_val_loss
      - valid (bool or None)
      - note (from tool call)
      - hp_config (parsed from HP_CONFIG_JSON in stdout)
      - our_train_gpt hash
      - git revision (if available)
    """
    history_path = os.path.join(workspace_root, RUN_HISTORY_FILE_NAME)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    hp_config = _parse_hp_config(stdout)

    # Hash of our_train_gpt.py to identify the exact code variant
    train_path = os.path.join(workspace_root, "our_train_gpt.py")
    our_train_sha = _sha256_of_file(train_path)

    # Git revision (optional, if repo is a git repo)
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            text=True,
        ).strip()
    except Exception:
        git_rev = None

    valid = None
    if best_val_loss is not None:
        valid = best_val_loss <= 3.28

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "time_seconds": time_seconds,
        "best_val_loss": best_val_loss,
        "valid": valid,
        "note": note,
        "hp_config": hp_config,
        "our_train_gpt_sha256": our_train_sha,
        "git_rev": git_rev,
    }

    try:
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        # Don't crash the tool if logging fails; just print to stderr
        print(f"[WARN] Failed to append benchmark record: {e}", flush=True)


# =============================================================================
# Tool: safe_write_file (only our_train_gpt.py)
# =============================================================================

class SafeWriteAction(Action):
    path: str = Field(
        description="Path to file to create/overwrite, relative to repo root. Must be 'our_train_gpt.py'."
    )
    content: str = Field(
        description="Full content for the file after this operation (no diffs).",
    )


class SafeWriteObservation(Observation):
    message: str = "ok"

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.message)]


class SafeWriteExecutor(ToolExecutor[SafeWriteAction, SafeWriteObservation]):
    """
    Executor for safe_write_file: only allows overwriting our_train_gpt.py.
    """

    async def run(
        self,
        action: SafeWriteAction,
        workspace,
        **kwargs,
    ) -> SafeWriteObservation:
        root = workspace.root_path
        if not _is_path_allowed(root, action.path):
            raise ValueError(
                f"Editing path '{action.path}' is not allowed. Only our_train_gpt.py may be edited."
            )

        abs_path = os.path.join(root, action.path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(action.content)

        return SafeWriteObservation(message=f"Wrote {action.path}")


safe_write_tool_def = ToolDefinition(
    name="safe_write_file",
    description=(
        "Overwrite our_train_gpt.py with new contents. "
        "Only this file is writable; all other paths will be rejected."
    ),
    action_type=SafeWriteAction,
    observation_type=SafeWriteObservation,
    executor=SafeWriteExecutor(),
)

SafeFileTool = register_tool(safe_write_tool_def)


# =============================================================================
# Tool: read_file (read-only access to repo files)
# =============================================================================

class ReadFileAction(Action):
    path: str = Field(description="Path to a file to read, relative to the repo root")


class ReadFileObservation(Observation):
    content: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.content)]


class ReadFileExecutor(ToolExecutor[ReadFileAction, ReadFileObservation]):
    """
    Executor for read_file: read and return the contents of a project file.
    """

    async def run(
        self,
        action: ReadFileAction,
        workspace,
        **kwargs,
    ) -> ReadFileObservation:
        root = workspace.root_path
        abs_path = os.path.abspath(os.path.join(root, action.path))
        if not os.path.exists(abs_path):
            return ReadFileObservation(content=f"[ERROR] File not found: {action.path}")
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            return ReadFileObservation(content=f"[ERROR] Failed to decode file as text: {action.path}")

        # Truncate to keep context manageable
        if len(text) > 8000:
            text = text[:8000] + "\n...[truncated]..."
        return ReadFileObservation(content=text)


read_file_tool_def = ToolDefinition(
    name="read_file",
    description="Read text from a project file (README.md, our_train_gpt.py, etc.).",
    action_type=ReadFileAction,
    observation_type=ReadFileObservation,
    executor=ReadFileExecutor(),
)

ReadFileTool = register_tool(read_file_tool_def)


# =============================================================================
# Tool: run_modded_nanogpt_benchmark
# =============================================================================

class RunBenchmarkAction(Action):
    note: str | None = Field(
        default=None,
        description="Optional explanation of what changed before this benchmark run.",
    )


class RunBenchmarkObservation(Observation):
    time_seconds: float | None = None
    best_val_loss: float | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        msg_lines = []
        if self.time_seconds is not None and self.best_val_loss is not None:
            msg_lines.append(
                f"Benchmark result: time_seconds={self.time_seconds:.3f}, "
                f"best_val_loss={self.best_val_loss:.4f}"
            )
            msg_lines.append(
                "Interpretation: best_val_loss must be <= 3.28 to be valid; "
                "among valid runs, smaller time_seconds is better."
            )
        msg_lines.append("Last lines of stdout:\n" + self.stdout_tail)
        if self.stderr_tail:
            msg_lines.append("\nLast lines of stderr:\n" + self.stderr_tail)
        return [TextContent(text="\n".join(msg_lines))]


class RunBenchmarkExecutor(ToolExecutor[RunBenchmarkAction, RunBenchmarkObservation]):
    """
    Executor for run_modded_nanogpt_benchmark: runs 'torchrun our_train_gpt.py'
    with a fixed environment and parses BENCH_RESULT + HP_CONFIG_JSON.
    """

    async def run(
        self,
        action: RunBenchmarkAction,
        workspace,
        **kwargs,
    ) -> RunBenchmarkObservation:
        root = workspace.root_path

        env = dict(os.environ)
        # Adjust GPU layout as desired; this just defaults to the first 2 GPUs
        env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

        proc = subprocess.run(
            BENCHMARK_CMD,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=BENCHMARK_TIMEOUT,
        )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        parsed = _parse_benchmark_score(stdout)
        if parsed is None:
            # Log even failed parses so we can debug weird runs
            _append_benchmark_record(
                workspace_root=root,
                time_seconds=None,
                best_val_loss=None,
                note=action.note,
                stdout=stdout,
            )
            return RunBenchmarkObservation(
                time_seconds=None,
                best_val_loss=None,
                stdout_tail="\n".join(stdout.splitlines()[-80:]),
                stderr_tail="\n".join(stderr.splitlines()[-40:]),
            )

        time_s, best_val = parsed

        # Log this run in the history file
        _append_benchmark_record(
            workspace_root=root,
            time_seconds=time_s,
            best_val_loss=best_val,
            note=action.note,
            stdout=stdout,
        )

        return RunBenchmarkObservation(
            time_seconds=time_s,
            best_val_loss=best_val,
            stdout_tail="\n".join(stdout.splitlines()[-80:]),
            stderr_tail="\n".join(stderr.splitlines()[-40:]),
        )


run_bench_tool_def = ToolDefinition(
    name="run_modded_nanogpt_benchmark",
    description=(
        "Run the full modded-nanogpt benchmark (torchrun our_train_gpt.py) and return "
        "time_seconds and best_val_loss parsed from stdout. Also logs a record to "
        "agent_logs/benchmark_runs.jsonl."
    ),
    action_type=RunBenchmarkAction,
    observation_type=RunBenchmarkObservation,
    executor=RunBenchmarkExecutor(),
)

RunBenchmarkTool = register_tool(run_bench_tool_def)


# =============================================================================
# Tool: web_search (Tavily)
# =============================================================================

class WebSearchAction(Action):
    query: str = Field(
        description="Web search query for research related to ML, optimization, or modded-nanogpt."
    )
    top_k: int = Field(default=5, description="Number of search results to return (max 10)")


class WebSearchObservation(Observation):
    results: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.results)]


class WebSearchExecutor(ToolExecutor[WebSearchAction, WebSearchObservation]):
    """
    Executor for web_search: uses Tavily's /search API to retrieve a small
    set of results (title/url/snippet). Uses a global call budget to avoid
    burning all credits.
    """

    async def run(
        self,
        action: WebSearchAction,
        workspace,
        **kwargs,
    ) -> WebSearchObservation:
        global _tavily_calls_used

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return WebSearchObservation(results="Tavily API key not provided (set TAVILY_API_KEY).")

        if _tavily_calls_used >= TAVILY_MAX_CALLS:
            return WebSearchObservation(
                results=(
                    "Tavily web search budget exhausted for this run. "
                    "Rely on existing knowledge, code, and logs instead."
                )
            )

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": action.query,
            "max_results": min(action.top_k, 10),
            "search_depth": "basic",  # cheaper than 'advanced'
            "include_answer": False,
        }

        try:
            resp = requests.post(url, json=payload, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            return WebSearchObservation(results=f"[Tavily search error] {e}")

        _tavily_calls_used += 1
        data = resp.json()
        results = data.get("results", [])

        if not results:
            return WebSearchObservation(results="No results found.")

        lines = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "No title")
            rurl = r.get("url", "No URL")
            snippet = r.get("content", "")
            lines.append(f"{i}. {title}\n{rurl}\n{snippet}\n")

        return WebSearchObservation(results="\n".join(lines))


web_search_def = ToolDefinition(
    name="web_search",
    description=(
        "Search the web using Tavily for research papers, ML optimizations, code ideas, etc. "
        "Use sparingly; there is a limited web research budget (TAVILY_MAX_CALLS)."
    ),
    action_type=WebSearchAction,
    observation_type=WebSearchObservation,
    executor=WebSearchExecutor(),
)

WebSearchTool = register_tool(web_search_def)


# =============================================================================
# Tool: fetch_url (Tavily content extraction)
# =============================================================================

class FetchUrlAction(Action):
    url: str = Field(description="URL to fetch and extract clean text from.")


class FetchUrlObservation(Observation):
    content: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.content)]


class FetchUrlExecutor(ToolExecutor[FetchUrlAction, FetchUrlObservation]):
    """
    Executor for fetch_url: uses Tavily's /extract API to fetch and extract
    clean text content from a URL. Shares the same call budget as web_search.
    """

    async def run(
        self,
        action: FetchUrlAction,
        workspace,
        **kwargs,
    ) -> FetchUrlObservation:
        global _tavily_calls_used

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return FetchUrlObservation(content="Tavily API key not provided (set TAVILY_API_KEY).")

        if _tavily_calls_used >= TAVILY_MAX_CALLS:
            return FetchUrlObservation(
                content=(
                    "Tavily web fetch budget exhausted for this run. "
                    "Rely on existing knowledge, code, and logs instead."
                )
            )

        url = "https://api.tavily.com/extract"
        payload = {
            "api_key": api_key,
            "url": action.url,
        }

        try:
            resp = requests.post(url, json=payload, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            return FetchUrlObservation(content=f"[Tavily fetch error] {e}")

        _tavily_calls_used += 1
        data = resp.json()
        text = data.get("content", "")

        if len(text) > 12000:
            text = text[:12000] + "\n...[truncated]..."

        return FetchUrlObservation(content=text)


fetch_url_def = ToolDefinition(
    name="fetch_url",
    description="Fetch and extract clean text content from a URL using Tavily.",
    action_type=FetchUrlAction,
    observation_type=FetchUrlObservation,
    executor=FetchUrlExecutor(),
)

FetchUrlTool = register_tool(fetch_url_def)


# =============================================================================
# Tool: read_benchmark_history
# =============================================================================

class ReadHistoryAction(Action):
    max_rows: int = Field(
        default=20,
        description="Maximum number of most recent benchmark runs to return.",
    )


class ReadHistoryObservation(Observation):
    summary: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.summary)]


class ReadHistoryExecutor(ToolExecutor[ReadHistoryAction, ReadHistoryObservation]):
    """
    Executor for read_benchmark_history: loads the JSONL history file and
    returns a compact human-readable summary of the most recent runs.
    """

    async def run(
        self,
        action: ReadHistoryAction,
        workspace,
        **kwargs,
    ) -> ReadHistoryObservation:
        root = workspace.root_path
        history_path = os.path.join(root, RUN_HISTORY_FILE_NAME)
        if not os.path.exists(history_path):
            return ReadHistoryObservation(summary="No benchmark history found yet.")

        rows = []
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            return ReadHistoryObservation(summary=f"Failed to read history: {e}")

        if not rows:
            return ReadHistoryObservation(summary="History file is empty or unreadable.")

        # Take the most recent N rows (file is append-only)
        n = max(1, action.max_rows)
        rows = rows[-n:]

        lines = []
        lines.append(
            "Recent benchmark runs (most recent last):\n"
            "idx | timestamp              | valid | time_s   | val_loss | note / key hparams"
        )
        for idx, r in enumerate(rows, start=1):
            ts = r.get("timestamp", "?")
            time_s = r.get("time_seconds")
            loss = r.get("best_val_loss")
            valid = r.get("valid")
            note = (r.get("note") or "").strip()

            # Try to pull a few key hyperparameters from hp_config.hyperparameters
            hp = r.get("hp_config") or {}
            hargs = hp.get("hyperparameters") or {}
            lr = None
            batch_size = None
            seq_len = None
            try:
                # might not exist depending on dataclass fields
                lr = hargs.get("lr", None)
            except Exception:
                pass
            try:
                batch_size = hargs.get("train_batch_size", None)
            except Exception:
                pass
            try:
                seq_len = hargs.get("train_max_seq_len", None)
            except Exception:
                pass

            hp_bits = []
            if lr is not None:
                hp_bits.append(f"lr={lr}")
            if batch_size is not None:
                hp_bits.append(f"train_batch_size={batch_size}")
            if seq_len is not None:
                hp_bits.append(f"train_max_seq_len={seq_len}")

            hp_str = ", ".join(hp_bits)
            if note and hp_str:
                summary_str = f"{note} | {hp_str}"
            elif note:
                summary_str = note
            else:
                summary_str = hp_str

            lines.append(
                f"{idx:3d} | {ts[:19]:19} | {str(valid):5} | "
                f"{str(time_s):7} | {str(loss):7} | {summary_str}"
            )

        return ReadHistoryObservation(summary="\n".join(lines))


read_history_def = ToolDefinition(
    name="read_benchmark_history",
    description=(
        "Read a summary of recent benchmark runs, including time, val loss, validity, "
        "and a few key hyperparameters from HP_CONFIG_JSON."
    ),
    action_type=ReadHistoryAction,
    observation_type=ReadHistoryObservation,
    executor=ReadHistoryExecutor(),
)

ReadHistoryTool = register_tool(read_history_def)


# =============================================================================
# LLM & Agent builders
# =============================================================================

def build_llm() -> LLM:
    """
    Build the LLM client.

    Assumes an OpenAI-compatible server (e.g. vLLM) exposed via LLM_BASE_URL.
    Default model name matches your Qwen3 coder model.
    """
    return LLM(
        model=os.getenv("LLM_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        api_key=os.getenv("LLM_API_KEY", "dummy"),  # local servers often ignore this
        base_url=os.getenv("LLM_BASE_URL"),         # e.g. http://localhost:8000/v1
    )


def build_agent(llm: LLM) -> Agent:
    """
    Construct an Agent with our custom tools.
    """
    tools = [
        Tool(name=SafeFileTool.name),
        Tool(name=ReadFileTool.name),
        Tool(name=RunBenchmarkTool.name),
        Tool(name=WebSearchTool.name),
        Tool(name=FetchUrlTool.name),
        Tool(name=ReadHistoryTool.name),
    ]
    return Agent(
        llm=llm,
        tools=tools,
    )


# =============================================================================
# Main entrypoint
# =============================================================================

def main() -> None:
    """
    Entry point: sets up the workspace, builds the agent, and runs a single
    long-lived Conversation.

    Environment variables:
      - NANOGPT_REPO_ROOT: path to the modded-nanogpt repo (default /mnt/disk/modded-nanogpt)
      - LLM_MODEL, LLM_BASE_URL, LLM_API_KEY: for your local OpenAI-compatible server
      - TAVILY_API_KEY, TAVILY_MAX_CALLS: for Tavily search budget
    """
    repo_root = os.path.abspath(os.getenv("NANOGPT_REPO_ROOT", "/mnt/disk/modded-nanogpt"))
    llm = build_llm()
    agent = build_agent(llm)

    conversation = Conversation(agent=agent, workspace=repo_root)

    # System prompt to shape the agent's behavior
    system_prompt = """
You are an autonomous research engineer improving the modded-nanogpt benchmark.

At the start of your work, call read_file on README.md (especially the 'Rules' section)
and on our_train_gpt.py. Use this to guide your changes.

Hard constraints:
- You may ONLY modify a single file: our_train_gpt.py in the repository root.
- You MUST NOT modify any data scripts, validation scripts, benchmark scripts, or the README.
- You MUST obey the rules in README.md under the 'Rules' section, including:
  - Do not change the underlying train/validation token streams.
  - Do not add extra torch._inductor.config or torch.compile flags.
  - Ensure your changes are consistent with training to <= 3.28 validation loss on FineWeb.
- The only way to run the benchmark is via the `run_modded_nanogpt_benchmark` tool.
- The only way to edit code is via the `safe_write_file` tool, which takes the FULL new file contents.

Objective and metrics:
- The benchmark tool returns two metrics: time_seconds and best_val_loss.
- Validation loss is a HARD constraint: any result with best_val_loss > 3.28 is unacceptable,
  regardless of speed.
- Among runs that achieve best_val_loss <= 3.28, your goal is to minimize time_seconds
  (i.e., reach the target loss as fast as possible).
- When comparing variants, always discard any configuration that fails the loss requirement,
  even if it's faster.

Experiment tracking:
- Every time you run the benchmark via run_modded_nanogpt_benchmark, the system automatically
  logs a record (time_seconds, best_val_loss, note, git revision, our_train_gpt.py hash, and
  any HP_CONFIG_JSON printed by our_train_gpt.py) into agent_logs/benchmark_runs.jsonl.
- Before proposing new changes or runs, call read_benchmark_history to review recent
  experiments and avoid repeating obviously bad configurations.
- Whenever you call run_modded_nanogpt_benchmark, include a clear note describing what you
  changed in our_train_gpt.py and your hypothesis (e.g. 'increase batch, slightly reduce lr').

Hyperparameters and logging:
- our_train_gpt.py contains a Hyperparameters dataclass that defines many key settings.
  You should generally modify that dataclass to change high-level hyperparameters,
  rather than sprinkling magic constants throughout the code.
- our_train_gpt.py prints a single HP_CONFIG_JSON line that contains a JSON snapshot of:
  - the Hyperparameters dataclass (via asdict(args))
  - optimizer param group settings (e.g. lr, betas, momentum, weight_decay)
- That HP_CONFIG_JSON is parsed into hp_config in the benchmark history. Use it when
  analyzing the effect of your changes.

Online research tools:
- `web_search` uses Tavily to search the web for papers, blog posts, GitHub issues, and
  optimization techniques related to GPT training, attention patterns, optimizers
  (e.g. SOAP, Muon), fused kernels, and modded-nanogpt.
- `fetch_url` fetches and extracts full text from a URL (also via Tavily).
- There is a LIMITED web research budget per run (TAVILY_MAX_CALLS), shared across web_search
  and fetch_url, so use these tools sparingly and only when local reasoning is insufficient.

Research workflow:
- First, carefully read README.md (especially the 'Rules' section) and our_train_gpt.py to
  understand the current setup, including the Hyperparameters dataclass and optimizer logic.
- Use read_benchmark_history to understand what configurations have already been tried,
  and how they affected time_seconds and best_val_loss.
- Develop hypotheses grounded in either:
  - your own reasoning and prior results, or
  - clearly-cited external sources discovered via web_search + fetch_url.
- Propose small, incremental modifications that respect all rules, and explain your rationale.
- Run the benchmark sparingly to evaluate meaningful changes and compare valid runs
  based on time_seconds, subject to the best_val_loss <= 3.28 constraint.
"""

    conversation.send_message(system_prompt.strip())
    conversation.send_message(
        "First, inspect README.md and our_train_gpt.py, summarize the rules and current training "
        "setup (including the Hyperparameters dataclass), and then propose concrete hypotheses "
        "for improving the benchmark metrics under the given constraints."
    )

    conversation.run()
    print("Agent finished.")


if __name__ == "__main__":
    main()
