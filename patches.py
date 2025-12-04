"""
ZSL Mitigation Patches for train_gpt.py

This file contains the exact code changes needed to integrate ZSL experiments.
Apply these changes to your train_gpt.py file.

Each section is marked with:
  # === ZSL PATCH: <section name> ===
  # <instructions>
  # === END ZSL PATCH ===
"""

# =============================================================================
# PATCH 1: Add after line ~45 (after ANALYSIS_MODE definitions)
# =============================================================================
"""
# === ZSL PATCH: Configuration Dataclass ===
# Add this after the ANALYSIS_MODE definitions

from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

@dataclass
class ZSLConfig:
    '''Configuration for ZSL mitigation experiments. All disabled by default.'''
    
    # Direction 1: SV Thresholding - reduce orthogonalization in Polar Express
    sv_threshold_enabled: bool = False
    sv_threshold_num_iters: int = 5  # Reduce to 3-4 for less aggressive ortho
    sv_threshold_blend_min: float = 0.5  # Minimum blend with orthogonalized result
    
    # Direction 2: Gradient Disagreement - downweight tokens in contested directions
    grad_disagree_enabled: bool = False
    grad_disagree_strength: float = 0.5  # 0=none, 1=strong downweighting
    
    # Direction 3: Momentum-Gradient Alignment - reduce momentum when grad disagrees
    momentum_align_enabled: bool = False
    momentum_align_threshold: float = 0.3  # Cosine sim below this = disagreement
    momentum_align_min_factor: float = 0.5  # Minimum momentum multiplier
    
    # Direction 4: Focal Loss - upweight hard tokens (corrected from earlier)
    focal_enabled: bool = False
    focal_gamma: float = 1.0  # Higher = more focus on hard tokens
    focal_easy_thresh: float = 0.2  # Percentile below which tokens are "easy"
    focal_easy_weight: float = 0.5  # Weight for easy tokens (downweight them)
    
    # Direction 5: DI-Adaptive Momentum - reduce momentum when DI is high  
    di_momentum_enabled: bool = False
    di_momentum_threshold: float = 0.2  # DI above this triggers reduction
    di_momentum_max_reduction: float = 0.15
    di_momentum_floor: float = 0.7
    
    # Direction 6: Extended Skip Connections
    extended_skips_enabled: bool = False
    skip_in_layers: Tuple[int, ...] = (2, 4, 6)  # Original: (4,)
    skip_out_layers: Tuple[int, ...] = (5, 7, 9)  # Original: (7,)
    
    @classmethod
    def from_env(cls) -> 'ZSLConfig':
        def get_bool(key, default):
            return os.environ.get(key, str(default)).lower() in ('1', 'true', 'yes')
        def get_float(key, default):
            return float(os.environ.get(key, str(default)))
        def get_int(key, default):
            return int(os.environ.get(key, str(default)))
        def get_tuple(key, default):
            val = os.environ.get(key, '')
            return tuple(int(x) for x in val.split(',')) if val else default
        
        return cls(
            sv_threshold_enabled=get_bool('ZSL_SV_THRESH', False),
            sv_threshold_num_iters=get_int('ZSL_SV_ITERS', 5),
            sv_threshold_blend_min=get_float('ZSL_SV_BLEND', 0.5),
            grad_disagree_enabled=get_bool('ZSL_GRAD_DISAGREE', False),
            grad_disagree_strength=get_float('ZSL_GRAD_DISAGREE_STR', 0.5),
            momentum_align_enabled=get_bool('ZSL_MOM_ALIGN', False),
            momentum_align_threshold=get_float('ZSL_MOM_ALIGN_THRESH', 0.3),
            momentum_align_min_factor=get_float('ZSL_MOM_ALIGN_MIN', 0.5),
            focal_enabled=get_bool('ZSL_FOCAL', False),
            focal_gamma=get_float('ZSL_FOCAL_GAMMA', 1.0),
            focal_easy_thresh=get_float('ZSL_FOCAL_EASY_THRESH', 0.2),
            focal_easy_weight=get_float('ZSL_FOCAL_EASY_WT', 0.5),
            di_momentum_enabled=get_bool('ZSL_DI_MOM', False),
            di_momentum_threshold=get_float('ZSL_DI_MOM_THRESH', 0.2),
            di_momentum_max_reduction=get_float('ZSL_DI_MOM_REDUCE', 0.15),
            di_momentum_floor=get_float('ZSL_DI_MOM_FLOOR', 0.7),
            extended_skips_enabled=get_bool('ZSL_EXT_SKIPS', False),
            skip_in_layers=get_tuple('ZSL_SKIP_IN', (2, 4, 6)),
            skip_out_layers=get_tuple('ZSL_SKIP_OUT', (5, 7, 9)),
        )

ZSL_CFG = ZSLConfig.from_env()

# Log active experiments
if rank == 0:  # or master_process if defined later
    active = [k for k, v in ZSL_CFG.__dict__.items() if v is True]
    if active:
        print(f"[ZSL] Active experiments: {active}")
    else:
        print("[ZSL] All experiments disabled (baseline)")

# === END ZSL PATCH ===
"""

# =============================================================================
# PATCH 2: Replace polar_express function (around line 360)
# =============================================================================
"""
# === ZSL PATCH: Modified Polar Express ===
# Replace the entire polar_express function with this version

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# Determine coefficients based on config (done at import time for torch.compile)
_polar_coeffs_to_use = polar_express_coeffs[:ZSL_CFG.sv_threshold_num_iters] if ZSL_CFG.sv_threshold_enabled else polar_express_coeffs
_sv_blend_min = ZSL_CFG.sv_threshold_blend_min if ZSL_CFG.sv_threshold_enabled else 1.0

@torch.compile(dynamic=False, fullgraph=True)
def polar_express(G: torch.Tensor):
    '''
    Polar Express Sign Method with optional SV thresholding.
    
    When sv_threshold_enabled:
    - Uses fewer Newton-Schulz iterations (configurable)
    - Blends with original normalized gradient to avoid over-committing
      in contested (low singular value) directions
    '''
    X = G.bfloat16()
    transpose = G.size(-2) > G.size(-1)
    if transpose:
        X = X.mT

    X_norm = X.norm(dim=(-2, -1), keepdim=True)
    
    # Save normalized original for potential blending
    X_original = X / (X_norm + 1e-6)
    
    X = X / (X_norm * (1 + 2e-2) + 1e-6)
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

    for a, b, c in _polar_coeffs_to_use:
        XXT(X, out=A)
        ba_plus_cAA(A, alpha=c, beta=b, out=B)
        aX_plus_BX(X, B, X, beta=a, out=C)
        X, C = C, X

    # Optional blending with original (when sv_threshold_enabled)
    if _sv_blend_min < 1.0:
        # Measure how "orthogonal" the result is
        X_frob = X.norm(dim=(-2, -1), keepdim=True)
        expected_orth = (min(X.size(-2), X.size(-1)) ** 0.5)
        orthogonality = (X_frob / (expected_orth + 1e-6)).clamp(0, 1)
        
        # Blend: more orthogonal = keep X, less = blend toward original
        blend = _sv_blend_min + (1.0 - _sv_blend_min) * orthogonality
        X = blend * X + (1.0 - blend) * X_original

    if transpose:
        X = X.mT
    return X

# === END ZSL PATCH ===
"""

# =============================================================================
# PATCH 3: Modify NorMuon.step() method (around line 436)
# =============================================================================
"""
# === ZSL PATCH: NorMuon.step with momentum-gradient alignment ===
# Replace the step() method in NorMuon class

@torch.no_grad()
def step(self, update_collector=None, param_meta=None):
    rank = dist.get_rank()
    group_infos = []
    for group in self.param_groups:
        params: list[Tensor] = group["params"]
        if not params:
            continue

        chunk_size = group["chunk_size"]
        padded_num_params = chunk_size * self.world_size

        stacked_grads = torch.empty(
            (padded_num_params, *params[0].shape),
            dtype=params[0].dtype,
            device=params[0].device,
        )
        for i, p in enumerate(params):
            stacked_grads[i].copy_(p.grad, non_blocking=True)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        grad_chunk = torch.empty_like(stacked_grads[:chunk_size])
        reduce_future = dist.reduce_scatter_tensor(
            grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
        ).get_future()
        group_infos.append(dict(grad_chunk=grad_chunk, reduce_future=reduce_future))

    all_gather_infos = []
    for group, info in zip(self.param_groups, group_infos):
        info["reduce_future"].wait()
        params = group["params"]
        grad_chunk = info["grad_chunk"]
        chunk_size = group["chunk_size"]
        padded_num_params = chunk_size * self.world_size
        start_idx = rank * chunk_size
        module_idx = start_idx if start_idx < len(params) else 0
        num_params = min(chunk_size, max(0, len(params) - start_idx))

        if "momentum_buffer" not in group:
            group["momentum_buffer"] = torch.zeros_like(grad_chunk[:num_params])
        momentum_buffer = group["momentum_buffer"]
        
        # === ZSL: Momentum-Gradient Alignment ===
        effective_momentum = group["momentum"]
        
        if ZSL_CFG.momentum_align_enabled and num_params > 0 and momentum_buffer.numel() > 0:
            grad_flat = grad_chunk[:num_params].flatten()
            mom_flat = momentum_buffer.flatten()
            grad_norm = grad_flat.norm()
            mom_norm = mom_flat.norm()
            
            if grad_norm > 1e-8 and mom_norm > 1e-8:
                alignment = (grad_flat @ mom_flat) / (grad_norm * mom_norm)
                
                if alignment < ZSL_CFG.momentum_align_threshold:
                    # Gradient disagrees with momentum - reduce momentum influence
                    # Linear interpolation: alignment=threshold -> factor=1, alignment=-1 -> factor=min
                    t = (ZSL_CFG.momentum_align_threshold - alignment) / (ZSL_CFG.momentum_align_threshold + 1)
                    factor = 1.0 - t * (1.0 - ZSL_CFG.momentum_align_min_factor)
                    effective_momentum = group["momentum"] * factor
        # === END ZSL ===
        
        momentum_buffer.lerp_(grad_chunk[:num_params], 1 - effective_momentum)
        updated_grads = grad_chunk[:num_params].lerp_(momentum_buffer, effective_momentum)

        grad_shape = updated_grads.shape
        if num_params > 0 and params[module_idx].label == 'attn':
            for p in params[module_idx:module_idx + num_params]:
                assert p.label == 'attn'
            updated_grads = updated_grads.view(4 * grad_shape[0], grad_shape[1], grad_shape[2] // 4)
        ref_param = params[module_idx]
        param_shape = ref_param.shape

        if "second_momentum_buffer" not in group:
            group["second_momentum_buffer"] = (
                torch.zeros_like(updated_grads[..., :, :1])
                if param_shape[-2] >= param_shape[-1]
                else torch.zeros_like(updated_grads[..., :1, :])
            )
        second_momentum_buffer = group["second_momentum_buffer"]

        if "param_lr" not in group:
            group["param_lr"] = (
                max(1., param_shape[-2] / param_shape[-1]) ** 0.5
                * ref_param.new_tensor(
                    [getattr(param, "lr_mul", 1.0) for param in params[module_idx:module_idx + num_params]]
                ).view(-1, 1, 1)
            )
            group["param_wd"] = ref_param.new_tensor(
                [getattr(param, "wd_mul", 1.0) for param in params[module_idx:module_idx + num_params]]
            ).view(-1, 1, 1)

        eff_lr = group["lr"] * group["param_lr"]
        eff_wd = group["lr"] * group["weight_decay"] * group["param_wd"]

        if num_params == 0:
            v_chunk = updated_grads
        else:
            v_chunk = polar_express(updated_grads)

        v_norm = v_chunk.norm(dim=(-2, -1), keepdim=True)
        v_mean = v_chunk.square().mean(dim=-1 if param_shape[-2] >= param_shape[-1] else -2, keepdim=True)
        second_momentum_buffer.lerp_(v_mean.to(dtype=ref_param.dtype), 1 - group["beta2"])
        step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
        v_chunk.mul_(step_size)
        v_norm_new = v_chunk.norm(dim=(-2, -1), keepdim=True)
        v_chunk.mul_(v_norm / v_norm_new.clamp_min_(1e-10))
        v_chunk = v_chunk.view(grad_shape)

        updated_params = torch.empty_like(grad_chunk)
        if num_params > 0:
            param_chunk = torch.stack(params[module_idx:module_idx + num_params])
        else:
            param_chunk = torch.zeros_like(v_chunk)

        mask = (v_chunk * param_chunk) >= 0
        v_chunk.addcmul_(param_chunk, (eff_wd * mask).to(ref_param.dtype))
        update_chunk = v_chunk * (-eff_lr)
        param_chunk.add_(update_chunk)

        if update_collector is not None and num_params > 0 and param_meta is not None:
            for local_idx, p in enumerate(params[module_idx:module_idx + num_params]):
                meta = param_meta.get(p)
                if meta is None:
                    continue
                u = update_chunk[local_idx]
                sum_abs = u.abs().sum().item()
                if sum_abs < 1e-8:
                    continue
                sum_signed = u.sum().item()
                update_collector.setdefault(meta["key_all"], [0.0, 0.0])
                update_collector[meta["key_all"]][0] += sum_signed
                update_collector[meta["key_all"]][1] += sum_abs
                if meta.get("key_type"):
                    update_collector.setdefault(meta["key_type"], [0.0, 0.0])
                    update_collector[meta["key_type"]][0] += sum_signed
                    update_collector[meta["key_type"]][1] += sum_abs
                if meta.get("key_layer_type"):
                    update_collector.setdefault(meta["key_layer_type"], [0.0, 0.0])
                    update_collector[meta["key_layer_type"]][0] += sum_signed
                    update_collector[meta["key_layer_type"]][1] += sum_abs

        updated_params[:num_params].copy_(param_chunk)
        if num_params < chunk_size:
            updated_params[num_params:].zero_()

        stacked_params = torch.empty(
            (padded_num_params, *param_shape),
            dtype=updated_params.dtype,
            device=updated_params.device,
        )

        gather_future = dist.all_gather_into_tensor(
            stacked_params, updated_params, async_op=True
        ).get_future()
        all_gather_infos.append(
            {"gather_future": gather_future, "stacked_params": stacked_params, "orig_params": params}
        )

    for info in all_gather_infos:
        info["gather_future"].wait()
        stacked_params = info["stacked_params"]
        orig_params = info["orig_params"]
        unstacked_params = torch.unbind(stacked_params)
        for i, p in enumerate(orig_params):
            p.copy_(unstacked_params[i], non_blocking=True)

# === END ZSL PATCH ===
"""

# =============================================================================
# PATCH 4: Modify GPT.forward() loss computation (around line 1146)
# =============================================================================
"""
# === ZSL PATCH: Token-weighted loss computation ===
# Replace the loss computation section in GPT.forward()

# Find this code:
#     per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
#
#     if self.training:
#         loss = per_token_loss.sum()
#     else:
#         loss = per_token_loss.mean()
#     return loss

# Replace with:

        per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")

        if self.training:
            weights = torch.ones_like(per_token_loss)
            
            # === ZSL Direction 2: Gradient Disagreement ===
            if ZSL_CFG.grad_disagree_enabled:
                with torch.no_grad():
                    probs = F.softmax(logits_flat, dim=-1)
                    n_tokens = probs.size(0)
                    
                    # Per-token gradient w.r.t. logits
                    g_logits = probs.clone()
                    g_logits[torch.arange(n_tokens, device=probs.device), targets_flat] -= 1.0
                    
                    # Per-vocab disagreement
                    g_sum = g_logits.sum(dim=0)
                    g_abs_sum = g_logits.abs().sum(dim=0)
                    di_vocab = 1.0 - (g_sum.abs() / (g_abs_sum + 1e-8))
                    
                    # Token conflict score
                    conflict = (g_logits.abs() * di_vocab.unsqueeze(0)).sum(dim=1)
                    conflict = conflict / (conflict.max() + 1e-8)
                    
                    # Downweight high-conflict tokens
                    weights = weights * (1.0 - ZSL_CFG.grad_disagree_strength * conflict)
            
            # === ZSL Direction 4: Focal Loss (upweight hard tokens) ===
            if ZSL_CFG.focal_enabled:
                with torch.no_grad():
                    n_tokens = per_token_loss.numel()
                    sorted_loss, _ = per_token_loss.sort()
                    
                    # Find easy threshold
                    easy_idx = max(1, int(ZSL_CFG.focal_easy_thresh * n_tokens))
                    easy_threshold = sorted_loss[easy_idx]
                    
                    # Focal weight: (1 - p_correct)^gamma
                    probs = F.softmax(logits_flat, dim=-1)
                    p_correct = probs[torch.arange(n_tokens, device=probs.device), targets_flat]
                    focal_wt = (1 - p_correct) ** ZSL_CFG.focal_gamma
                    
                    # Downweight easy tokens
                    is_easy = (per_token_loss < easy_threshold).float()
                    easy_wt = 1.0 - is_easy * (1.0 - ZSL_CFG.focal_easy_weight)
                    
                    weights = weights * focal_wt * easy_wt
            
            # Normalize weights
            if ZSL_CFG.grad_disagree_enabled or ZSL_CFG.focal_enabled:
                weights = weights / (weights.mean() + 1e-8)
            
            loss = (per_token_loss * weights.detach()).sum()
        else:
            loss = per_token_loss.mean()
        return loss

# === END ZSL PATCH ===
"""

# =============================================================================
# PATCH 5: Modify step_optimizers() for DI-adaptive momentum (around line 1635)
# =============================================================================
"""
# === ZSL PATCH: DI-Adaptive Momentum ===
# Replace step_optimizers function

def step_optimizers(step: int, optimizers, model):
    global _last_update_di
    update_collector = {}

    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)

    momentum = get_muon_momentum(step)
    
    # === ZSL Direction 5: DI-Adaptive Momentum ===
    if ZSL_CFG.di_momentum_enabled:
        prev_di = _last_update_di.get("di_update_update_all", 0.0)
        if prev_di > ZSL_CFG.di_momentum_threshold:
            excess = prev_di - ZSL_CFG.di_momentum_threshold
            max_excess = 1.0 - ZSL_CFG.di_momentum_threshold
            reduction = ZSL_CFG.di_momentum_max_reduction * (excess / max_excess)
            momentum = max(momentum - reduction, ZSL_CFG.di_momentum_floor)
    # === END ZSL ===
    
    for group in optimizers[1].param_groups:
        group["momentum"] = momentum

    if step % 2 == 0:
        optimizers[1].step(update_collector=update_collector, param_meta=param_meta_map)
        optimizers[1].zero_grad(set_to_none=True)
    else:
        optimizers[1].step(update_collector=update_collector, param_meta=param_meta_map)
        optimizers[0].step()
        model.zero_grad(set_to_none=True)
        optimizers[0].should_sync = False

    # compute DI for applied updates (Muon path)
    di = {}
    for key, (s_signed, s_abs) in update_collector.items():
        if s_abs > 1e-8:
            di[f"di_update_{key}"] = 1.0 - (abs(s_signed) / s_abs)
    _last_update_di = di

# === END ZSL PATCH ===
"""

# =============================================================================
# PATCH 6: Modify GPT forward for extended skip connections (around line 1047)
# =============================================================================
"""
# === ZSL PATCH: Extended Skip Connections ===
# Modify GPT.forward() skip connection handling

# Replace:
#     skip_connections = []
#     n = len(self.blocks) // 2
#     skip_in = [4]
#     skip_out = [7]

# With:
        skip_connections = []
        n = len(self.blocks) // 2
        
        # === ZSL Direction 6: Extended Skip Connections ===
        if ZSL_CFG.extended_skips_enabled:
            skip_in = list(ZSL_CFG.skip_in_layers)
            skip_out = list(ZSL_CFG.skip_out_layers)
        else:
            skip_in = [4]
            skip_out = [7]
        # === END ZSL ===

# Note: For extended skips, you need to ensure there are enough skip weights.
# The original code uses: skip_weights = self.scalars[: (len(self.blocks) // 2)]
# This gives 6 weights for 12 layers, which is enough for up to 6 skip connections.
# 
# If using skip_in = [2, 4, 6] and skip_out = [5, 7, 9], you need 3 skip weights.
# The indexing in the loop needs adjustment:

# Replace the loop logic:
#     for i in range(1, len(self.blocks)):
#         ...
#         if i in skip_out:
#             gate = torch.sigmoid(skip_weights[i - n])
#             x = x + gate * skip_connections.pop()
#         ...
#         if i in skip_in:
#             skip_connections.append(x)

# With:
        skip_weight_idx = 0
        for i in range(1, len(self.blocks)):
            attn_args = AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                seqlens=seqlens,
                bm_size=bm_sizes[i],
                cos=self.yarn.cos,
                sin=self.yarn.sin,
                attn_scale=self.yarn.attn_scale,
            )
            
            # Add skip connection if this is an output layer
            if i in skip_out:
                out_idx = skip_out.index(i)
                if out_idx < len(skip_connections):
                    # Use the corresponding skip_in's position for weight indexing
                    in_layer = skip_in[out_idx]
                    # Weight index based on in_layer position  
                    weight_idx = skip_in.index(in_layer) if ZSL_CFG.extended_skips_enabled else (i - n)
                    gate = torch.sigmoid(skip_weights[weight_idx])
                    x = x + gate * skip_connections[out_idx]
            
            x = self.blocks[i](x, x0, lambdas[i], attn_args)
            
            # Save skip connection if this is an input layer
            if i in skip_in:
                skip_connections.append(x)
            
            if i == backout_layer:
                x_backout = x

# === END ZSL PATCH ===
"""

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

USAGE = """
# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# Baseline (no ZSL experiments):
torchrun --nproc_per_node=8 train_gpt.py

# Direction 1: SV Thresholding (less aggressive orthogonalization)
ZSL_SV_THRESH=1 ZSL_SV_ITERS=3 torchrun --nproc_per_node=8 train_gpt.py

# Direction 2: Gradient Disagreement (downweight contested tokens)
ZSL_GRAD_DISAGREE=1 ZSL_GRAD_DISAGREE_STR=0.3 torchrun --nproc_per_node=8 train_gpt.py

# Direction 3: Momentum-Gradient Alignment
ZSL_MOM_ALIGN=1 ZSL_MOM_ALIGN_THRESH=0.2 torchrun --nproc_per_node=8 train_gpt.py

# Direction 4: Focal Loss (upweight hard tokens, downweight easy)
ZSL_FOCAL=1 ZSL_FOCAL_GAMMA=2.0 ZSL_FOCAL_EASY_WT=0.3 torchrun --nproc_per_node=8 train_gpt.py

# Direction 5: DI-Adaptive Momentum
ZSL_DI_MOM=1 ZSL_DI_MOM_THRESH=0.25 torchrun --nproc_per_node=8 train_gpt.py

# Direction 6: Extended Skip Connections
ZSL_EXT_SKIPS=1 ZSL_SKIP_IN=2,4,6 ZSL_SKIP_OUT=5,7,9 torchrun --nproc_per_node=8 train_gpt.py

# Combine experiments:
ZSL_FOCAL=1 ZSL_DI_MOM=1 ZSL_EXT_SKIPS=1 torchrun --nproc_per_node=8 train_gpt.py

# Full list of environment variables:
#   ZSL_SV_THRESH      - Enable SV thresholding (0/1)
#   ZSL_SV_ITERS       - Number of Newton-Schulz iterations (default: 5)
#   ZSL_SV_BLEND       - Minimum blend with orthogonalized (default: 0.5)
#   ZSL_GRAD_DISAGREE  - Enable gradient disagreement (0/1)
#   ZSL_GRAD_DISAGREE_STR - Strength of disagreement penalty (default: 0.5)
#   ZSL_MOM_ALIGN      - Enable momentum alignment (0/1)
#   ZSL_MOM_ALIGN_THRESH - Cosine sim threshold (default: 0.3)
#   ZSL_MOM_ALIGN_MIN  - Minimum momentum factor (default: 0.5)
#   ZSL_FOCAL          - Enable focal loss (0/1)
#   ZSL_FOCAL_GAMMA    - Focal loss gamma (default: 1.0)
#   ZSL_FOCAL_EASY_THRESH - Easy token percentile (default: 0.2)
#   ZSL_FOCAL_EASY_WT  - Weight for easy tokens (default: 0.5)
#   ZSL_DI_MOM         - Enable DI-adaptive momentum (0/1)
#   ZSL_DI_MOM_THRESH  - DI threshold for reduction (default: 0.2)
#   ZSL_DI_MOM_REDUCE  - Max momentum reduction (default: 0.15)
#   ZSL_DI_MOM_FLOOR   - Minimum momentum (default: 0.7)
#   ZSL_EXT_SKIPS      - Enable extended skips (0/1)
#   ZSL_SKIP_IN        - Skip input layers (default: 2,4,6)
#   ZSL_SKIP_OUT       - Skip output layers (default: 5,7,9)
"""

print(USAGE)