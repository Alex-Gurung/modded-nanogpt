#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON
from rich.text import Text
from rich import box

console = Console()


def find_conversations(root: Path) -> List[Path]:
    conv_root = root / ".conversations"
    if not conv_root.is_dir():
        console.print(f"[red]No .conversations directory found at {conv_root}[/red]")
        return []
    # Only directories that look like UUIDs, but be flexible.
    return sorted(
        [p for p in conv_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_events(conv_dir: Path) -> List[Dict[str, Any]]:
    events_dir = conv_dir / "events"
    if not events_dir.is_dir():
        console.print(f"[yellow]No events/ directory in {conv_dir}[/yellow]")
        return []

    event_files = sorted(events_dir.glob("event-*.json"))
    events: List[Dict[str, Any]] = []
    for f in event_files:
        try:
            with f.open("r") as fh:
                data = json.load(fh)
            data["_filename"] = f.name  # for reference
            events.append(data)
        except Exception as e:
            console.print(f"[red]Failed to load {f}: {e}[/red]")
    return events


def guess_event_type(ev: Dict[str, Any]) -> str:
    # Try common fields
    for key in ("event_type", "type", "kind"):
        if key in ev:
            return str(ev[key])
    # Sometimes everything is inside "data"
    data = ev.get("data", {})
    for key in ("event_type", "type", "kind"):
        if key in data:
            return str(data[key])
    return "unknown"


def extract_message(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Try to interpret an event as a chat message.
    """
    data = ev.get("data", ev)

    role = data.get("role")
    content = data.get("content")

    if role is None and isinstance(data.get("message"), dict):
        # Some formats embed "message": {"role":..., "content":...}
        msg = data["message"]
        role = msg.get("role")
        content = msg.get("content")

    if role is None or content is None:
        return None

    # content might be list (OpenAI-style) or plain string
    if isinstance(content, list):
        # keep only text parts
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(str(part["text"]))
            elif isinstance(part, str):
                texts.append(part)
        content = "\n".join(texts)

    if not isinstance(content, str):
        try:
            content = json.dumps(content, indent=2, ensure_ascii=False)
        except Exception:
            content = str(content)

    return {"role": role, "content": content}


def is_tool_event(ev: Dict[str, Any]) -> bool:
    etype = guess_event_type(ev).lower()
    if "tool" in etype:
        return True
    data = ev.get("data", {})
    if isinstance(data, dict):
        name = str(data.get("tool_name") or data.get("name") or "").lower()
        if name:
            return True
    return False


def pretty_print_message(ev: Dict[str, Any], idx: int) -> None:
    msg = extract_message(ev)
    if not msg:
        return pretty_print_generic(ev, idx)

    role = msg["role"]
    content = msg["content"]

    header = Text(f"[{idx}] {role}", style="bold cyan" if role == "assistant" else "bold magenta")
    panel = Panel(
        Text(content),
        title=header,
        border_style="cyan" if role == "assistant" else "magenta",
        padding=(1, 2),
    )
    console.print(panel)


def pretty_print_tool(ev: Dict[str, Any], idx: int) -> None:
    data = ev.get("data", ev)
    etype = guess_event_type(ev)

    table = Table(
        title=f"[{idx}] Tool Event: {etype}",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_lines=False,
        expand=True,
    )
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")

    # Common interesting fields
    interesting_keys = [
        "tool_name",
        "name",
        "call_id",
        "status",
        "exit_code",
        "command",
        "path",
        "stdout",
        "stderr",
        "error",
        "duration",
    ]

    seen = set()
    for key in interesting_keys:
        if key in data and data[key] is not None:
            val = data[key]
            if isinstance(val, (dict, list)):
                val = json.dumps(val, indent=2, ensure_ascii=False)
            table.add_row(key, str(val))
            seen.add(key)

    # Also show any args / input
    for key in ["arguments", "args", "input"]:
        if key in data and key not in seen:
            val = data[key]
            if isinstance(val, (dict, list)):
                val = json.dumps(val, indent=2, ensure_ascii=False)
            table.add_row(key, str(val))
            seen.add(key)

    console.print(table)


def pretty_print_generic(ev: Dict[str, Any], idx: int) -> None:
    etype = guess_event_type(ev)
    title = f"[{idx}] Event: {etype}"

    # Prefer the "data" field if present
    payload = ev.get("data", ev)
    panel = Panel(
        JSON.from_data(payload),
        title=title,
        border_style="dim",
        padding=(1, 2),
    )
    console.print(panel)


def print_conversation(conv_dir: Path, max_events: int) -> None:
    console.rule(f"[bold green]Conversation: {conv_dir.name}")
    events = load_events(conv_dir)
    if not events:
        console.print("[yellow]No events found.[/yellow]")
        return

    total = len(events)
    if max_events > 0 and total > max_events:
        events = events[-max_events:]
        console.print(f"[dim]Showing last {max_events} of {total} events[/dim]\n")
    else:
        console.print(f"[dim]Showing all {total} events[/dim]\n")

    for idx, ev in enumerate(events, start=1):
        msg = extract_message(ev)
        if msg is not None:
            pretty_print_message(ev, idx)
        elif is_tool_event(ev):
            pretty_print_tool(ev, idx)
        else:
            pretty_print_generic(ev, idx)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretty-inspect OpenHands conversation logs using Rich."
    )
    parser.add_argument(
        "-c",
        "--conversation-id",
        type=str,
        help="Conversation ID (directory name under .conversations). If omitted, use most recent.",
    )
    parser.add_argument(
        "-n",
        "--num-events",
        type=int,
        default=50,
        help="Max number of most recent events to show (0 = all).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory of the repo (where .conversations/ lives).",
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()

    conversations = find_conversations(root)
    if not conversations:
        return

    if args.conversation_id:
        conv_dir = root / ".conversations" / args.conversation_id
        if not conv_dir.is_dir():
            console.print(f"[red]Conversation {args.conversation_id} not found.[/red]")
            console.print("Available conversations:")
            for c in conversations[:10]:
                console.print(f"  - {c.name}")
            return
    else:
        conv_dir = conversations[0]
        console.print(f"[dim]No conversation-id given; using most recent: {conv_dir.name}[/dim]")

    print_conversation(conv_dir, args.num_events)


if __name__ == "__main__":
    main()
