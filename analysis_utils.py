# ===== analysis_utils.py (inline or separate) =====
import os, math, json
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

@dataclass
class AnalysisConfig:
    enabled: bool = bool(int(os.environ.get("ANALYSIS_MODE", "1")))  # default ON for this script
    log_every: int = int(os.environ.get("LOG_EVERY_STEPS", "100"))
    zsl_every: int = int(os.environ.get("ZSL_EVERY_STEPS", "1000"))  # 0 = disable
    zsl_batch_size: int = int(os.environ.get("ZSL_BATCH_SIZE", "16"))
    neuron_dump_every: int = int(os.environ.get("NEURON_DUMP_EVERY", "500"))

ANALYSIS = AnalysisConfig()

class JSONLLogger:
    def __init__(self, path: str, enabled: bool = True):
        self.enabled = enabled
        self.path = path
        self.f = open(path, "a") if enabled else None

    def log(self, record: Dict[str, Any]):
        if not self.enabled or self.f is None:
            return
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()

def compute_token_stats_from_logits(
    logits_for_loss: torch.Tensor,
    target_seq: torch.Tensor,
    sample_size: int = 32,
) -> Dict[str, Any]:
    # logits_for_loss: (T, V) or (B, T, V)
    with torch.no_grad():
        vocab_size = logits_for_loss.size(-1)
        logits_flat = logits_for_loss.view(-1, vocab_size).float()
        targets_flat = target_seq.view(-1)
        idx = torch.arange(logits_flat.size(0), device=logits_flat.device)

        log_probs = F.log_softmax(logits_flat, dim=-1)
        probs = log_probs.exp()

        target_logp = log_probs[idx, targets_flat]
        target_p = target_logp.exp()
        token_loss = -target_logp

        entropy = -(probs * log_probs).sum(dim=-1)

        top_values, top_indices = probs.topk(2, dim=-1)
        top1_idx = top_indices[:, 0]
        top1_p = top_values[:, 0]
        top2_p = top_values[:, 1]
        top2_gap = top1_p - top2_p

        correct = (top1_idx == targets_flat).float()
        solved_95 = (target_p > 0.95).float()
        solved_99 = (target_p > 0.99).float()

        def q(x, qs):
            qt = torch.quantile(x, torch.tensor(qs, device=x.device))
            return [float(v) for v in qt.cpu().tolist()]

        loss_q50, loss_q90, loss_q99 = q(token_loss, [0.5, 0.9, 0.99])
        p_q10, p_q50, p_q90 = q(target_p, [0.1, 0.5, 0.9])

        overconf_wrong = (((correct == 0) & (top1_p >= 0.9)).float().mean()).item()
        underconf_right = (((correct == 1) & (target_p <= 0.5)).float().mean()).item()

        N = token_loss.numel()
        k = min(sample_size, N)
        perm = torch.randperm(N, device=token_loss.device)[:k]

        return {
            "num_tokens": int(N),
            "loss_mean": float(token_loss.mean().item()),
            "loss_std": float(token_loss.std(unbiased=False).item()),
            "loss_quantiles": {
                "p50": float(loss_q50),
                "p90": float(loss_q90),
                "p99": float(loss_q99),
            },
            "acc_top1": float(correct.mean().item()),
            "p_target_mean": float(target_p.mean().item()),
            "p_target_std": float(target_p.std(unbiased=False).item()),
            "p_target_quantiles": {
                "p10": float(p_q10),
                "p50": float(p_q50),
                "p90": float(p_q90),
            },
            "frac_solved_95": float(solved_95.mean().item()),
            "frac_solved_99": float(solved_99.mean().item()),
            "entropy_mean": float(entropy.mean().item()),
            "entropy_std": float(entropy.std(unbiased=False).item()),
            "top2_gap_mean": float(top2_gap.mean().item()),
            "top2_gap_std": float(top2_gap.std(unbiased=False).item()),
            "overconf_wrong_frac": float(overconf_wrong),
            "underconf_right_frac": float(underconf_right),
            "sample": {
                "loss": [float(v) for v in token_loss[perm].cpu().tolist()],
                "p_target": [float(v) for v in target_p[perm].cpu().tolist()],
                "entropy": [float(v) for v in entropy[perm].cpu().tolist()],
                "correct": [int(v) for v in correct[perm].cpu().tolist()],
            },
        }

@torch.no_grad()
def compute_global_grad_and_param_norms(model: torch.nn.Module) -> Dict[str, float]:
    total_grad_sq = 0.0
    total_param_sq = 0.0
    max_grad = 0.0

    with torch.no_grad():
        for p in model.parameters():
            pf = p.float()
            total_param_sq += float((pf * pf).sum().item())
            if p.grad is not None:
                g = p.grad.float()
                total_grad_sq += float((g * g).sum().item())
                g_max = float(g.abs().max().item())
                if g_max > max_grad:
                    max_grad = g_max

    return {
        "global_grad_norm": math.sqrt(total_grad_sq) if total_grad_sq > 0 else 0.0,
        "global_grad_max": max_grad,
        "global_param_norm": math.sqrt(total_param_sq) if total_param_sq > 0 else 0.0,
    }

def compute_optimizer_grad_stats(optimizers) -> Dict[str, float]:
    stats = {}
    for opt_idx, opt in enumerate(optimizers):
        total_sq = 0.0
        max_abs = 0.0
        with torch.no_grad():
            for group in opt.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.float()
                    total_sq += float((g * g).sum().item())
                    g_max = float(g.abs().max().item())
                    if g_max > max_abs:
                        max_abs = g_max
        stats[f"opt{opt_idx}_grad_norm"] = math.sqrt(total_sq) if total_sq > 0 else 0.0
        stats[f"opt0{opt_idx}_grad_max"] = max_abs
    return stats

def compute_layer_stats(model) -> Dict[str, Any]:
    """Collect per-block weight/grad norms and gate/activation EMAs."""
    out: Dict[str, Any] = {}

    with torch.no_grad():
        for i, block in enumerate(model.blocks):
            key = f"layer_{i}"
            layer_info: Dict[str, Any] = {}

            # weights / grads
            if block.mlp is not None:
                mlp = block.mlp
                for name, w in [("c_fc", mlp.c_fc), ("c_proj", mlp.c_proj)]:
                    w_norm = float(w.float().norm().item())
                    g_norm = float(w.grad.float().norm().item()) if w.grad is not None else 0.0
                    layer_info[f"{name}_w_norm"] = w_norm
                    layer_info[f"{name}_g_norm"] = g_norm

                if hasattr(mlp, "act_mean_ema"):
                    mean = mlp.act_mean_ema
                    var = mlp.act_sq_mean_ema - mean * mean
                    # simple summaries
                    layer_info["mlp_act_mean_mean"] = float(mean.mean().item())
                    layer_info["mlp_act_mean_std"] = float(mean.std(unbiased=False).item())
                    layer_info["mlp_act_var_mean"] = float(var.mean().item())
                    layer_info["mlp_act_var_std"] = float(var.std(unbiased=False).item())

            if block.attn is not None:
                attn = block.attn
                w = attn.qkvo_w
                layer_info["attn_qkvo_w_norm"] = float(w.float().norm().item())
                layer_info["attn_qkvo_g_norm"] = float(w.grad.float().norm().item()) if w.grad is not None else 0.0

                if hasattr(attn, "gate_mean_ema"):
                    gm = attn.gate_mean_ema
                    gv = attn.gate_sq_mean_ema - gm * gm
                    layer_info["attn_gate_mean_mean"] = float(gm.mean().item())
                    layer_info["attn_gate_mean_std"] = float(gm.std(unbiased=False).item())
                    layer_info["attn_gate_var_mean"] = float(gv.mean().item())
                    layer_info["attn_gate_var_std"] = float(gv.std(unbiased=False).item())

            out[key] = layer_info

    return out

@torch.no_grad()
def zsl_gradient_probe(
    model,
    batch_inputs,
    batch_targets,
    seqlens,
    ws_short: int,
    ws_long: int,
    param_selector: str = "lm_head",
    max_examples: int = 16,
) -> Dict[str, Any]:
    """
    Compute a small ZSL proxy on a subset of parameters.

    param_selector: "lm_head" or "last_mlp"
    """
    device = batch_inputs.device
    with torch.no_grad():
        # treat each example as a whole input_seq; your training uses B=1,
        # so we simply take the first max_examples tokens or sequences.
        # We'll approximate using tokens as examples.
        inputs = batch_inputs.view(-1)  # (N,)
        targets = batch_targets.view(-1)
        N = inputs.size(0)
        num = min(max_examples, N)

    # Choose parameter tensor
    if param_selector == "lm_head":
        param = model.lm_head.weight
    elif param_selector == "last_mlp":
        # pick final block's c_fc
        for blk in reversed(model.blocks):
            if blk.mlp is not None:
                param = blk.mlp.c_fc
                break
    else:
        param = model.lm_head.weight

    # Collect per-example gradients
    grads = []
    for i in range(num):
        model.zero_grad(set_to_none=True)
        # single-token "example": input i, target i
        x_i = inputs[i:i+1]
        y_i = targets[i:i+1]
        loss_i = model(x_i, y_i, seqlens=None, ws_short=ws_short, ws_long=ws_long)
        g_i, = torch.autograd.grad(loss_i, param, retain_graph=False)
        grads.append(g_i.detach().flatten())

    G = torch.stack(grads, dim=0)  # [num, D]
    with torch.no_grad():
        norms = G.norm(dim=-1)  # [num]
        # avoid zero
        norms_clamped = norms.clamp_min(1e-12)
        G_normed = G / norms_clamped.unsqueeze(-1)

        # cosine matrix
        cos = G_normed @ G_normed.t()
        # mask diagonal
        num_f = float(num)
        mask = ~torch.eye(num, dtype=torch.bool, device=device)
        cos_off = cos[mask]
        mean_cos = float(cos_off.mean().item())
        frac_neg = float((cos_off < 0.0).float().mean().item())

        sum_g = G.sum(dim=0)
        gd = float(sum_g.norm().pow(2).item() / norms.pow(2).sum().item())

        return {
            "zsl_num_examples": int(num),
            "zsl_mean_grad_cos": mean_cos,
            "zsl_frac_negative_cos": frac_neg,
            "zsl_grad_diversity": gd,
        }

