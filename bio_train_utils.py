# bio_train_utils.py
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn


# -------------------------------
# Hard batch replay (SRS-ish)
# -------------------------------

@dataclass
class HardReplayConfig:
    enabled: bool = False
    max_size: int = 64              # max number of batches in buffer
    base_interval: int = 128        # steps between first and second review
    interval_growth: float = 2.0    # multiplier for interval after each review
    min_loss_scale: float = 1.05    # only keep batches with loss > mean * scale
    ema_beta: float = 0.99          # EMA for running train loss
    replay_fraction: float = 0.25   # fraction of grad_accum microsteps that may use replay


@dataclass
class HardBatch:
    inputs_cpu: Tensor
    targets_cpu: Tensor
    cum_seqlens_cpu: Tensor
    loss: float
    next_step: int
    seen: int = 0


class HardReplayBuffer:
    def __init__(self, cfg: HardReplayConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.buffer: List[HardBatch] = []
        self.loss_ema: Optional[float] = None

    def __len__(self) -> int:
        return len(self.buffer)

    # ---- train-loss EMA ----
    def update_loss_ema(self, batch_loss: float) -> None:
        if self.loss_ema is None:
            self.loss_ema = batch_loss
        else:
            beta = self.cfg.ema_beta
            self.loss_ema = beta * self.loss_ema + (1.0 - beta) * batch_loss

    # ---- selecting & storing batches ----
    def maybe_add(
        self,
        inputs: Tensor,
        targets: Tensor,
        cum_seqlens: Tensor,
        loss: float,
        step: int,
    ) -> None:
        """Store a batch if it's 'hard enough' relative to running loss."""
        if not self.cfg.enabled:
            return

        if self.loss_ema is not None:
            if loss < self.cfg.min_loss_scale * self.loss_ema:
                return

        hb = HardBatch(
            inputs_cpu=inputs.detach().cpu(),
            targets_cpu=targets.detach().cpu(),
            cum_seqlens_cpu=cum_seqlens.detach().cpu(),
            loss=float(loss),
            next_step=step + self.cfg.base_interval,
            seen=0,
        )
        self.buffer.append(hb)

        # Evict easiest if over capacity
        if len(self.buffer) > self.cfg.max_size:
            easiest_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i].loss)
            del self.buffer[easiest_idx]

    def sample_due(self, step: int) -> Optional[HardBatch]:
        if not self.cfg.enabled or not self.buffer:
            return None
        due_indices = [i for i, b in enumerate(self.buffer) if b.next_step <= step]
        if not due_indices:
            return None
        idx = random.choice(due_indices)
        hb = self.buffer.pop(idx)
        return hb

    def requeue(self, hb: HardBatch, new_loss: float, step: int) -> None:
        """Reinsert after review with updated loss & schedule."""
        hb.loss = float(new_loss)
        hb.seen += 1
        interval = int(self.cfg.base_interval * (self.cfg.interval_growth ** hb.seen))
        hb.next_step = step + max(interval, 1)
        self.buffer.append(hb)


# -------------------------------
# Neuron gating + tag-and-reset
# -------------------------------

@dataclass
class NeuronGatingConfig:
    enable_gating: bool = False
    gating_topk_fraction: float = 0.5  # keep top-k rows by grad EMA, zero rest
    enable_reset: bool = False
    reset_interval: int = 1000         # steps between reset sweeps
    reset_quantile: float = 0.05       # bottom q fraction of rows get reset
    grad_ema_decay: float = 0.99       # EMA decay for row-wise grad norms
    labels_to_track: tuple = ("mlp", "attn")  # param.label to include


def init_neuron_stats(
    model: nn.Module,
    device: torch.device,
    cfg: NeuronGatingConfig,
) -> Dict[nn.Parameter, Tensor]:
    """
    Create a per-parameter tensor of row-wise grad EMAs.
    Only tracks parameters with .label in cfg.labels_to_track and ndim >= 2.
    """
    stats: Dict[nn.Parameter, Tensor] = {}
    for _, p in model.named_parameters():
        label = getattr(p, "label", None)
        if label in cfg.labels_to_track and p.ndim >= 2:
            # Interpret 'row' as dim 0, which is correct for qkvo_w (attn)
            # and reasonably aligned for c_fc / c_proj (mlp).
            stats[p] = torch.zeros(p.shape[0], device=device, dtype=torch.float32)
    return stats


@torch.no_grad()
def apply_neuron_gating_and_reset(
    neuron_stats: Dict[nn.Parameter, Tensor],
    step: int,
    cfg: NeuronGatingConfig,
) -> None:
    """
    - Updates per-row grad EMAs for each tracked parameter.
    - Optionally zeros gradients for low-importance rows (gating).
    - Optionally periodically reinitializes lowest-EMA rows (reset).
    """
    if not neuron_stats:
        return

    do_gating = cfg.enable_gating
    do_reset = cfg.enable_reset and step > 0 and (step % cfg.reset_interval == 0)

    for param, row_ema in neuron_stats.items():
        if param.grad is None:
            continue

        grad = param.grad
        if grad.ndim > 2:
            grad2d = grad.view(grad.shape[0], -1)
        else:
            grad2d = grad

        # Row-wise grad norm
        row_norms = grad2d.float().pow(2).mean(dim=1).sqrt()

        # Update EMA
        beta = cfg.grad_ema_decay
        row_ema.mul_(beta).add_(row_norms, alpha=(1.0 - beta))

        # ----- Gating: keep only top-k rows -----
        if do_gating:
            num_rows = row_ema.numel()
            k = max(1, int(cfg.gating_topk_fraction * num_rows))
            if k < num_rows:
                topk_vals, topk_idx = torch.topk(row_ema, k=k, dim=0)
                mask = torch.ones_like(row_ema, dtype=torch.bool)
                mask[topk_idx] = False
                grad2d[mask] = 0.0  # zero gradients for low-importance rows

        # ----- Reset: recycle dead rows -----
        if do_reset and row_ema.numel() > 0:
            # find bottom quantile of rows
            q = cfg.reset_quantile
            if 0.0 < q < 1.0:
                threshold = torch.quantile(row_ema, q)
                dead_mask = row_ema <= threshold
            else:
                dead_mask = torch.zeros_like(row_ema, dtype=torch.bool)

            if dead_mask.any():
                # Reinitialize those rows of the parameter and reset their grad & EMA.
                p2d = param.data.view(param.shape[0], -1)
                # Small truncated normal reinit
                std = 0.02
                torch.nn.init.trunc_normal_(
                    p2d[dead_mask],
                    mean=0.0,
                    std=std,
                    a=-2 * std,
                    b=2 * std,
                )
                # Reset EMA and gradients
                alive_mask = ~dead_mask
                mean_alive = row_ema[alive_mask].mean() if alive_mask.any() else row_norms.mean()
                row_ema[dead_mask] = mean_alive
                grad2d[dead_mask] = 0.0

