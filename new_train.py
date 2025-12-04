#!/usr/bin/env python
import os
import sys

# Read this file's code ASAP for logging
with open(sys.argv[0]) as f:
    code = f.read()

import copy
import glob
import math
import threading
import time
import uuid
from dataclasses import dataclass
from collections import defaultdict
from itertools import accumulate
from pathlib import Path
import json

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems

import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor, nn

from kernels import get_kernel  # same as original

dynamo.config.recompile_limit = 64

# -----------------------------------------------------------------------------
# Global analysis knobs
# -----------------------------------------------------------------------------

# Enable analysis-only behavior (no training math changes)
# ANALYSIS_MODE = os.environ.get("ANALYSIS_MODE", "0") not in ("0", "false", "False", "")
ANALYSIS_MODE = True 
LOG_MISTAKES_EVERY = int(os.environ.get("LOG_MISTAKES_EVERY", "100"))
# LOG_MISTAKES_EVERY = 100 

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

rank = int(os.environ["RANK"])
# Log active experiments
if rank == 0:  # or master_process if defined later
    active = [k for k, v in ZSL_CFG.__dict__.items() if v is True]
    if active:
        print(f"[ZSL] Active experiments: {active}")
    else:
        print("[ZSL] All experiments disabled (baseline)")

# -----------------------------------------------------------------------------
# FP8 custom matmul by @YouJiacheng (unchanged)
# -----------------------------------------------------------------------------

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Triton kernels for XXT etc. (unchanged)
# -----------------------------------------------------------------------------

def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]

@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)
    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def XXT_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def XXT(A: torch.Tensor, out: torch.Tensor):
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    XXT_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )
    return out

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ba_plus_cAA_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ba_plus_cAA_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )
    return out

# Polar Express coefficients (unchanged)
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# === ZSL PATCH: Modified Polar Express ===
# Replace the entire polar_express function with this version

#polar_express_coeffs = [
#    (8.156554524902461, -22.48329292557795, 15.878769915207462),
#    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
#    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
#    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
#    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
# ]

polar_express_coeffs = [
    (7.706030141118669, -22.229473232552213, 16.16727804459456),
    (3.443374785630018, -2.7282730941305497, 0.5583110302049723),
    (3.115515746565568, -2.8693827288726377, 0.75144940447043),
    (2.295274606908728, -1.9765443440388453, 0.6464378616178528),
    (1.8620648518207559, -1.21977185958322, 0.3580080821791749),
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

# -----------------------------------------------------------------------------
# Muon / NorMuon optimizer (unchanged math)
# -----------------------------------------------------------------------------

class NorMuon(torch.optim.Optimizer):
    # ... identical to your original NorMuon (no changes)
    # For brevity, omitted comments are the same.
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, beta2=0.95, custom_sizing=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        if custom_sizing and dist.get_world_size() == 8:
            param_groups = self.generate_custom_param_groups(params)
        else:
            param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)

    def reset(self):
        for group in self.param_groups:
            group["momentum_buffer"].zero_()
            group["second_momentum_buffer"].zero_()

    def generate_standard_param_groups(self, params):
        groups = defaultdict(list)
        for param in params:
            groups[param.label].append(param)

        param_groups = []
        for module_name, group_params in groups.items():
            chunk_size = (len(group_params) + self.world_size - 1) // self.world_size
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))
        return param_groups

    def generate_custom_param_groups(self, params):
        module_group_order = ['smear_gate', 'attn_gate', 'attn', 'mlp']
        params_list = list(params)
        params_list.sort(key=lambda x: module_group_order.index(x.label))

        idx = 0
        group_sizes = [1, 10, 16, 16]
        assert len(params_list) == sum(group_sizes)
        param_groups = []
        for size in group_sizes:
            chunk_size = (size + self.world_size - 1) // self.world_size
            group_params = params_list[idx: idx + size]
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))
            idx += size
        return param_groups

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

# -----------------------------------------------------------------------------
# DistAdam (unchanged math)
# -----------------------------------------------------------------------------

class DistAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        for p in params:
            chunk_size = p.size(0) // self.world_size
            exp_avg = torch.zeros_like(p[:chunk_size], dtype=torch.bfloat16, device=p.device)
            exp_avg_sq = torch.zeros_like(exp_avg)
            self.state[p] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)

        self.should_sync = False
        self._reduce_scatter_hooks = []
        self._reduce_scatter_futures = {}
        self.register_backward_hooks()

    def register_backward_hooks(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for param in params:
                hook = param.register_post_accumulate_grad_hook(self._sync_gradient)
                self._reduce_scatter_hooks.append(hook)

    @torch.compile
    @torch.no_grad()
    def _sync_gradient(self, param):
        if not self.should_sync:
            return
        grad = param.grad
        rank_size = grad.shape[0] // self.world_size
        grad_slice = torch.empty_like(grad[:rank_size])
        self._reduce_scatter_futures[param] = (
            dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future(),
            grad_slice,
        )

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        all_gather_futures: list[torch.Future] = []
        for group in reversed(self.param_groups):
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for param in reversed(group["params"]):
                if param not in self._reduce_scatter_futures:
                    continue
                fut, g_slice = self._reduce_scatter_futures[param]
                fut.wait()

                rank_size = param.shape[0] // self.world_size
                p_slice = param[rank * rank_size:(rank + 1) * rank_size]
                lr = group["lr"] * getattr(param, "lr_mul", 1.0)
                state = self.state[param]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)

                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)

                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (bias2**0.5 / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)

                all_gather_futures.append(
                    dist.all_gather_into_tensor(param, p_slice, async_op=True).get_future()
                )
        self._reduce_scatter_futures.clear()
        torch.futures.collect_all(all_gather_futures).wait()

# -----------------------------------------------------------------------------
# Model components (unchanged math, with small read-only analysis hooks)
# -----------------------------------------------------------------------------

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

# YaRN (unchanged math)
class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=self.head_dim // 4, dtype=torch.float32, device=device
        )
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim // 4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
        self.sin = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        self.angular_freq = angular_freq
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float

flash_attn_interface = get_kernel("varunneal/flash-attention-3").flash_attn_interface

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim

        assert self.hdim == self.dim
        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkvo_w = nn.Parameter(torch.empty(self.hdim, self.dim * 4))
        self.qkvo_w.label = "attn"
        with torch.no_grad():
            self.qkvo_w.view(4, self.hdim, self.dim)[:3].uniform_(-bound, bound)
            self.qkvo_w.view(4, self.hdim, self.dim)[3].zero_()

        self.attn_gate = CastedLinear(12, num_heads)
        self.attn_gate.weight.label = "attn_gate"

        if ANALYSIS_MODE:
            self.register_buffer("gate_mean_ema", torch.zeros(num_heads))
            self.register_buffer("gate_sq_mean_ema", torch.zeros(num_heads))
            self.register_buffer("gate_ema_count", torch.tensor(0.0))

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1)
        assert B == 1
        assert T % 16 == 0

        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        q, k, v = F.linear(
            x,
            self.qkvo_w.view(4, self.hdim, self.dim)[:3].flatten(end_dim=1).type_as(x),
        ).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)

        q, k = norm(q), norm(k)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)

        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)
        else:
            v = sa_lambdas[0] * v

        max_len = args.train_max_seq_len if self.training else (
            args.val_batch_size // (grad_accum_steps * world_size)
        )

        y = flash_attn_interface.flash_attn_varlen_func(
            q[0],
            k[0],
            v[0],
            cu_seqlens_q=seqlens,
            cu_seqlens_k=seqlens,
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            causal=True,
            softmax_scale=attn_scale,
            window_size=(bm_size, 0),
        )
        y = y.view(B, T, self.num_heads, self.head_dim)

        gates = torch.sigmoid(self.attn_gate(x[..., : self.attn_gate.weight.size(-1)]))  # [B, T, H]

        if ANALYSIS_MODE:
            with torch.no_grad():
                g_det = gates.detach()
                mean = g_det.mean(dim=(0, 1))
                sq_mean = (g_det * g_det).mean(dim=(0, 1))
                c = self.gate_ema_count
                new_c = c + 1.0
                alpha = 1.0 / new_c
                one_minus = 1.0 - alpha
                self.gate_mean_ema.mul_(one_minus).add_(mean, alpha=alpha)
                self.gate_sq_mean_ema.mul_(one_minus).add_(sq_mean, alpha=alpha)
                self.gate_ema_count.copy_(new_c)

        y = y * gates.view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, self.qkvo_w.view(4, self.hdim, self.dim)[3].type_as(y))
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        self.c_fc.label = "mlp"
        self.c_proj.label = "mlp"
        self.c_fc.lr_mul = 2.0

        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()

        if ANALYSIS_MODE:
            self.register_buffer("act_mean_ema", torch.zeros(hdim))
            self.register_buffer("act_sq_mean_ema", torch.zeros(hdim))
            self.register_buffer("act_ema_count", torch.tensor(0.0))

    def forward(self, x: Tensor):
        x = F.linear(x, self.c_fc.T.type_as(x))
        h = F.relu(x).square()

        if ANALYSIS_MODE:
            with torch.no_grad():
                h_det = h.detach()
                mean = h_det.mean(dim=(0, 1))
                sq_mean = (h_det * h_det).mean(dim=(0, 1))
                c = self.act_ema_count
                new_c = c + 1.0
                alpha = 1.0 / new_c
                one_minus = 1.0 - alpha
                self.act_mean_ema.mul_(one_minus).add_(mean, alpha=alpha)
                self.act_sq_mean_ema.mul_(one_minus).add_(sq_mean, alpha=alpha)
                self.act_ema_count.copy_(new_c)

        x = F.linear(h, self.c_proj.type_as(h))
        return x

class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx not in [0, 7] else None
        self.mlp = MLP(dim) if layer_idx != 0 else None

    def forward(self, x: Tensor, x0: Tensor, lambdas: Tensor, attn_args: AttnArgs):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.smear_gate = CastedLinear(12, 1)
        self.smear_gate.weight.label = "smear_gate"
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
        self.yarn = Yarn(head_dim, max_seq_len)
        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        self.lm_head = CastedLinear(
            model_dim, vocab_size, use_fp8=use_fp8, x_s=(model_dim**0.5) / 448, w_s=2**-9, grad_s=1 / 448
        )
        assert num_layers % 2 == 0
        pad = (-num_layers * 5 - 2) % dist.get_world_size()
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    -1.5 * torch.ones(num_layers),
                    *[torch.tensor([1.1, 0.0]) for _ in range(num_layers)],
                    *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
                    torch.zeros(1),
                    0.5 * torch.ones(1),
                    torch.ones(pad),
                ]
            )
        )
        for param in self.embed.parameters():
            param.lr_mul = 75.0
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.0
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0

        # compressed token-level view for logging
        self._analysis_last_token_indices = None
        self._analysis_last_topk_indices = None
        self._analysis_last_topk_logprobs = None
        self._analysis_last_targets = None
        self._analysis_last_logit_mean = None
        self._analysis_last_logit_mean_abs = None
        self._analysis_last_logit_l2 = None
        self._analysis_last_zero_sum = None
        self._analysis_last_destructive = None
        self._analysis_last_landscape = None

    @torch.no_grad()
    def _stash_analysis_view(
        self,
        logits_for_loss: torch.Tensor,
        target_seq: torch.Tensor,
        default_topk: int = 20,
        default_max_tokens: int = 4096,
    ):
        if not ANALYSIS_MODE:
            return
        if logits_for_loss.ndim != 3:
            return

        B, T, V = logits_for_loss.shape
        device = logits_for_loss.device
        logits_flat = logits_for_loss.view(-1, V)
        targets_flat = target_seq.view(-1)
        N = logits_flat.size(0)

        try:
            topk = int(os.environ.get("ANALYSIS_TOPK", str(default_topk)))
        except Exception:
            topk = default_topk
        topk = max(1, min(topk, V))

        try:
            max_tokens = int(os.environ.get("ANALYSIS_MAX_TOKENS", str(default_max_tokens)))
        except Exception:
            max_tokens = default_max_tokens
        max_tokens = max(1, max_tokens)

        if N > max_tokens:
            sample_idx = torch.randperm(N, device=device)[:max_tokens]
            logits_sampled = logits_flat.index_select(0, sample_idx)
            targets_sampled = targets_flat.index_select(0, sample_idx)
        else:
            sample_idx = torch.arange(N, device=device)
            logits_sampled = logits_flat
            targets_sampled = targets_flat

        topk_vals, topk_idx = torch.topk(logits_sampled, k=topk, dim=-1)
        # log-softmax within top-k slice (approx)
        topk_logprobs = torch.log_softmax(topk_vals.float(), dim=-1)

        self._analysis_last_token_indices = sample_idx.detach().cpu()
        self._analysis_last_topk_indices = topk_idx.detach().cpu()
        self._analysis_last_topk_logprobs = topk_logprobs.detach().cpu()
        self._analysis_last_targets = targets_sampled.detach().cpu()

    def forward(
        self,
        input_seq: Tensor,
        target_seq: Tensor,
        seqlens: Tensor,
        ws_short: int,
        ws_long: int,
        log_stats: bool = False,
    ):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [None, ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        short_bm = ws_short * args.block_size
        long_bm = ws_long * args.block_size
        bm_sizes = [
            None,
            short_bm,
            short_bm,
            short_bm,
            long_bm,
            short_bm,
            short_bm,
            None,
            short_bm,
            short_bm,
            short_bm,
            long_bm,
        ]
        assert len(bm_sizes) == len(self.blocks)

        x = self.embed(input_seq)

        skip_weights = self.scalars[: (len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)
        smear_lambda = self.scalars[5 * len(self.blocks)]
        backout_lambda = self.scalars[5 * len(self.blocks) + 1]

        smear_gate_out = smear_lambda * torch.sigmoid(
            self.smear_gate(x[1:, : self.smear_gate.weight.size(-1)])
        )
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        # skip_connections = []
        # n = len(self.blocks) // 2
        # skip_in = [4]
        # skip_out = [7]
        # === ZSL Direction 6: Extended Skip Connections ===
        skip_connections = []
        n = len(self.blocks) // 2
        
        if ZSL_CFG.extended_skips_enabled:
            skip_in = list(ZSL_CFG.skip_in_layers)
            skip_out = list(ZSL_CFG.skip_out_layers)
        else:
            skip_in = [4]
            skip_out = [7]
        # === END ZSL ===
        x_backout = None
        backout_layer = 8

        # for i in range(1, len(self.blocks)):
        #     attn_args = AttnArgs(
        #         ve=ve[i],
        #         sa_lambdas=sa_lambdas[i],
        #         seqlens=seqlens,
        #         bm_size=bm_sizes[i],
        #         cos=self.yarn.cos,
        #         sin=self.yarn.sin,
        #         attn_scale=self.yarn.attn_scale,
        #     )
        #     if i in skip_out:
        #         gate = torch.sigmoid(skip_weights[i - n])
        #         x = x + gate * skip_connections.pop()
        #     x = self.blocks[i](x, x0, lambdas[i], attn_args)
        #     if i in skip_in:
        #         skip_connections.append(x)
        #     if i == backout_layer:
        #         x_backout = x
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

        x -= backout_lambda * x_backout
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.sigmoid(logits / 7.5)
        logits_for_loss = logits.float() if not self.training else logits

        vocab_size = logits_for_loss.size(-1)
        logits_flat = logits_for_loss.view(-1, vocab_size)
        targets_flat = target_seq.view(-1)

        if ANALYSIS_MODE and log_stats and not torch._dynamo.is_compiling():
            with torch.no_grad():
                self._stash_analysis_view(logits_for_loss, target_seq)
                # lightweight logit summary statistics (zero-sum / deceleration clues)
                logit_mean_per_token = logits_for_loss.mean(dim=-1)
                self._analysis_last_logit_mean = logit_mean_per_token.mean().detach().cpu()
                self._analysis_last_logit_mean_abs = logit_mean_per_token.abs().mean().detach().cpu()
                self._analysis_last_logit_l2 = (
                    logits_for_loss.square().mean(dim=-1).sqrt().mean().detach().cpu()
                )
                # zero-sum-ish metrics over full logits
                logits_det = logits_for_loss.detach()
                logits_sum = logits_det.sum(dim=-1)
                logits_norm = logits_det.norm(dim=-1)
                eps = 1e-6
                zero_sum_residual = (logits_sum.abs() / (logits_norm + eps)).mean()
                logits_centered = logits_det - logits_det.mean(dim=-1, keepdim=True)
                centered_norm_ratio = (logits_centered.norm(dim=-1) / (logits_norm + eps)).mean()
                log_probs = F.log_softmax(logits_det.float(), dim=-1)
                probs = log_probs.exp()
                entropy = (-(probs * log_probs).sum(dim=-1)).mean()
                entropy_gap = math.log(vocab_size) - entropy
                self._analysis_last_zero_sum = {
                    "zero_sum_residual": float(zero_sum_residual.cpu()),
                    "centered_norm_ratio": float(centered_norm_ratio.cpu()),
                    "entropy_gap_vs_uniform": float(entropy_gap.cpu()),
                }
                # destructive interference approximations (sampled tokens, dense over vocab)
                try:
                    di_sample = int(os.environ.get("DI_SAMPLE_TOKENS", "256"))
                except Exception:
                    di_sample = 256
                di_sample = max(1, min(di_sample, logits_flat.size(0)))
                di_idx = torch.randperm(logits_flat.size(0), device=logits_for_loss.device)[:di_sample]
                logits_di = logits_flat.index_select(0, di_idx).float()
                targets_di = targets_flat.index_select(0, di_idx)
                log_probs_di = F.log_softmax(logits_di, dim=-1)
                probs_di = log_probs_di.exp()
                g_di = probs_di
                g_di[torch.arange(di_sample, device=g_di.device), targets_di] -= 1.0  # gradients wrt logits
                G = g_di.sum(dim=0)
                g_di_abs_sum = g_di.abs().sum()
                G_abs_sum = G.abs().sum()
                di_grad = 1.0 - (G_abs_sum / (g_di_abs_sum + 1e-12))
                delta_l = -(g_di * G).sum(dim=1)  # approx loss improvement per token under combined step
                di_loss = 1.0 - (delta_l.sum().abs() / (delta_l.abs().sum() + 1e-12))
                self._analysis_last_destructive = {
                    "destructive_grad": float(di_grad.cpu()),
                    "destructive_loss": float(di_loss.cpu()),
                }
                # approximate per-token loss landscape along update direction (alpha grid)
                lssl_sample = min(di_sample, int(os.environ.get("LSSL_SAMPLE", "32")))
                alpha_grid = [-1.0, -0.5, 0.0, 0.5, 1.0]
                dot_i = delta_l  # already -g_i·G
                selected = dot_i[:lssl_sample].detach().cpu()
                targets_sel = targets_di[:lssl_sample].detach().cpu()
                curves = [[float(a * v.item()) for a in alpha_grid] for v in selected]
                self._analysis_last_landscape = {
                    "alphas": alpha_grid,
                    "delta_l": curves,
                    "targets": [int(t) for t in targets_sel.tolist()],
                }

        #########################################################
        # ZSL PATCH: Add ZSL loss computation
        #########################################################
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
        #########################################################
        # END ZSL PATCH: Add ZSL loss computation
        #########################################################
        return loss

# -----------------------------------------------------------------------------
# Token & grad stats (approximate, analysis-only, no training effect)
# -----------------------------------------------------------------------------

def compute_token_stats_from_topk(topk_logprobs, topk_indices, targets):
    if topk_logprobs is None or topk_indices is None or targets is None:
        print(f"returning None because of topk_logprobs {topk_logprobs}; topk_indices {topk_indices} targets {targets}")
        return None
    # all on CPU
    logp = topk_logprobs  # [N, K], float32
    idx = topk_indices    # [N, K], int64
    tgt = targets.long()  # [N]

    N, K = logp.shape
    probs = logp.exp()
    topk_mass = probs.sum(dim=1)

    # predicted token & prob (top-1 within top-k slice)
    top1_idx = idx[:, 0]
    top1_p = probs[:, 0]
    correct = (top1_idx == tgt).float()

    # target prob (0 if not in top-k)
    tgt_expanded = tgt.unsqueeze(1)                # [N,1]
    match = (idx == tgt_expanded)                  # [N,K] bool
    target_p = (probs * match.float()).sum(dim=1)  # [N]
    # approximate loss with top-k slice only
    eps = 1e-12
    token_loss = -torch.log(target_p.clamp_min(eps))  # [N]

    # top2 gap if K>=2
    if K >= 2:
        top2_gap = probs[:, 0] - probs[:, 1]
    else:
        top2_gap = torch.zeros_like(probs[:, 0])

    def quantiles(x, qs):
        q_t = torch.quantile(x, torch.tensor(qs, dtype=torch.float32))
        return [float(v) for v in q_t.tolist()]

    loss_q50, loss_q90 = quantiles(token_loss, [0.5, 0.9])
    p_q50 = quantiles(target_p, [0.5])[0]

    frac_high_conf = float((top1_p >= 0.9).float().mean().item())
    frac_high_conf_wrong = float(((top1_p >= 0.9) & (correct < 0.5)).float().mean().item())

    stats = {
        "num_tokens_sampled": int(N),
        "token_loss_mean": float(token_loss.mean().item()),
        "token_loss_q50": float(loss_q50),
        "token_loss_q90": float(loss_q90),
        "accuracy": float(correct.mean().item()),
        "p_correct_mean": float(target_p.mean().item()),
        "frac_high_conf": float(frac_high_conf),
        "frac_high_conf_wrong": float(frac_high_conf_wrong),
        "top2_gap_mean_est": float(top2_gap.mean().item()),
        "topk_prob_mass_mean_est": float(topk_mass.mean().item()),
    }

    # small random sample for debugging
    sample_size = min(32, N)
    perm = torch.randperm(N)[:sample_size]
    stats["sample"] = {
        "loss": [float(v) for v in token_loss[perm].tolist()],
        "p_target": [float(v) for v in target_p[perm].tolist()],
        "correct": [int(v) for v in correct[perm].tolist()],
    }
    return stats

def compute_grad_stats(optimizers):
    stats = {}
    for opt_idx, opt in enumerate(optimizers):
        total_sq = 0.0
        max_abs = 0.0
        for group in opt.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                gf = g.float()
                total_sq += float(gf.pow(2).sum().item())
                max_abs = max(max_abs, float(gf.abs().max().item()))
        stats[f"opt{opt_idx}_grad_norm"] = math.sqrt(total_sq) if total_sq > 0 else 0.0
        stats[f"opt{opt_idx}_grad_max"] = max_abs
    return stats


def compute_destructive_by_group(model):
    """
    Per-parameter destructive interference (gradient opposition) by layer and param type.
    D = 1 - |sum g| / sum |g|
    """
    eps = 1e-12
    di = {}

    def add_metric(key, g):
        if g is None:
            return
        g_f = g.float()
        sum_abs = g_f.abs().sum()
        if sum_abs < 1e-8:
            return
        sum_signed = g_f.sum()
        di[key] = float(1.0 - (sum_signed.abs() / (sum_abs + eps)).item())

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        # type from label/name
        ptype = getattr(p, "label", None)
        if ptype is None:
            if "qkvo_w" in name:
                ptype = "attn_qkv"
            elif "attn_gate" in name:
                ptype = "attn_gate"
            elif "lm_head" in name:
                ptype = "head"
            elif "embed" in name or "value_embeds" in name:
                ptype = "embed"
            elif "gate" in name:
                ptype = "gate"
            elif "attn" in name:
                ptype = "attn"
            elif "mlp" in name:
                ptype = "mlp"
            else:
                ptype = "other"
        # layer index if present
        layer_idx = None
        if "blocks." in name:
            try:
                layer_idx = int(name.split("blocks.")[1].split(".")[0])
            except Exception:
                layer_idx = None

        add_metric("di_grad_type_" + ptype, p.grad)
        add_metric("di_grad_all", p.grad)
        if layer_idx is not None:
            add_metric(f"di_grad_layer{layer_idx}_{ptype}", p.grad)
    return di


def build_param_meta(model):
    meta = {}
    for name, p in model.named_parameters():
        ptype = getattr(p, "label", None)
        if ptype is None:
            if "qkvo_w" in name:
                ptype = "attn_qkv"
            elif "attn_gate" in name:
                ptype = "attn_gate"
            elif "lm_head" in name:
                ptype = "head"
            elif "embed" in name or "value_embeds" in name:
                ptype = "embed"
            elif "gate" in name:
                ptype = "gate"
            elif "attn" in name:
                ptype = "attn"
            elif "mlp" in name:
                ptype = "mlp"
            else:
                ptype = "other"
        layer_idx = None
        if "blocks." in name:
            try:
                layer_idx = int(name.split("blocks.")[1].split(".")[0])
            except Exception:
                layer_idx = None
        key_all = "update_all"
        key_type = f"update_type_{ptype}"
        key_layer_type = f"update_layer{layer_idx}_{ptype}" if layer_idx is not None else None
        meta[p] = {"key_all": key_all, "key_type": key_type, "key_layer_type": key_layer_type}
    return meta

# -----------------------------------------------------------------------------
# Data loader (unchanged)
# -----------------------------------------------------------------------------

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520
    assert header[1] == 1
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens

BOS_ID = 50256

class BOSFinder:
    def __init__(self, tokens: Tensor, world_size: int = 1, quickload: bool = False):
        self.tokens = tokens
        self.size = tokens.numel()
        self.quickload = quickload
        if quickload:
            self.bos_idx = (tokens[:4_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
            self.thread = None
            self.ready = threading.Event()
            self.start()
        else:
            self.bos_idx = (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.i = 0
        self.world_size = world_size
        self.batch_iter = 0

    def _load(self):
        self.bos_idx_async = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        self.bos_idx = self.bos_idx_async

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        if self.quickload and self.batch_iter == 5:
            self.get()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead of position {cur}; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(
                    self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                    cur + max_seq_len,
                    cur + num_tokens_local - cur_len + 1,
                )
                ends[r].append(end)
                cur_len += end - cur
                idx += 1
            assert cur_len == num_tokens_local + 1
        self.i = idx
        self.batch_iter += 1
        return starts, ends

class DataPreloader:
    def __init__(self, file_iter, world_size: int = 1):
        self.file_iter = file_iter
        self.world_size = world_size
        self.thread = None
        self.data = None
        self.ready = threading.Event()

    def _load(self):
        tokens = _load_data_shard(next(self.file_iter))
        self.data = (tokens, BOSFinder(tokens, self.world_size))
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        return self.data

def distributed_data_generator(
    filename_pattern: str,
    num_tokens: int,
    max_seq_len: int,
    grad_accum_steps: int = 1,
    align_to_bos: bool = True,
):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        finder = BOSFinder(tokens, world_size=world_size, quickload=True)
        preloader = DataPreloader(file_iter, world_size)
        preloader.start()
    else:
        pos = 0

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)

        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                tokens, finder = preloader.get()
                preloader.start()
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1
            cum_lengths = (end_idxs - start_idxs).cumsum(0)
        else:
            if pos + num_tokens + 1 >= len(tokens):
                tokens, pos = _load_data_shard(next(file_iter)), 0
            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local)
            _targets = buf[1:].view(num_tokens_local)
            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens

        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1 : len(cum_lengths) + 1] = cum_lengths

        new_params = yield (
            _inputs.to(device="cuda", dtype=torch.int32, non_blocking=True),
            _targets.to(device="cuda", dtype=torch.int64, non_blocking=True),
            _cum_lengths.to(device="cuda", dtype=torch.int32, non_blocking=True),
        )

        if new_params is not None:
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * grad_accum_steps) == 0
            num_tokens = new_num_tokens
            max_seq_len = new_max_seq_len
            grad_accum_steps = new_grad_accum_steps

# -----------------------------------------------------------------------------
# Hyperparameters & setup (unchanged)
# -----------------------------------------------------------------------------

@dataclass
class Hyperparameters:
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens: int = 10485760
    train_batch_size: int = 2048 * 16 * 8
    train_max_seq_len: int = 128 * 16
    val_batch_size: int = 4 * 64 * 1024 * 8
    # num_scheduled_iterations: int = 2185
    num_scheduled_iterations: int = 1200
    num_extension_iterations: int = 40
    num_iterations: int = num_scheduled_iterations + num_extension_iterations
    cooldown_frac: float = 0.50
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 250
    save_checkpoint: bool = False
    block_size: int = 128
    ws_schedule: tuple = (3, 7, 11)
    ws_final: int = 13
    ws_validate_post_yarn_ext: int = 20

args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert 8 % world_size == 0
grad_accum_steps = 8 // world_size
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = rank == 0

logfile = None
run_id = args.run_id
if master_process:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)

def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

print0(code)
print0("=" * 100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running Triton version {triton.__version__}")

def nvidia_smi():
    import subprocess
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout

print0(nvidia_smi())
print0("=" * 100)

model: nn.Module = GPT(
    vocab_size=50257,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=max(args.train_batch_size, args.val_batch_size) // (grad_accum_steps * world_size),
).cuda()
for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

param_meta_map = build_param_meta(model)

hidden_matrix_params = [
    p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n and "gate" not in n
]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]
gate_params = [p for n, p in model.named_parameters() if "gate" in n]

optimizer1 = DistAdam(
    scalar_params + head_params + embed_params,
    lr=0.008,
    betas=(0.65, 0.95),
    eps=1e-8,
    weight_decay=0.0,
)
optimizer2 = NorMuon(hidden_matrix_params + gate_params, lr=0.03, momentum=0.95, beta2=0.95, weight_decay=1.2)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

def get_lr(step: int):
    x = min(0.9999, step / args.num_scheduled_iterations)
    assert 0 <= x < 1
    lr = 1.0
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = w * 1.0 + (1 - w) * 0.1
    return lr

def get_ws(step: int):
    if step >= args.num_scheduled_iterations:
        return args.ws_final // 2, args.ws_final
    x = step / args.num_scheduled_iterations
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx] // 2, args.ws_schedule[ws_idx]

def get_muon_momentum(step: int, muon_warmup_steps=300, muon_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95):
    momentum_cd_start = args.num_iterations - muon_cooldown_steps
    if step < muon_warmup_steps:
        frac = step / muon_warmup_steps
        momentum = momentum_min + frac * (momentum_max - momentum_min)
    elif step > momentum_cd_start:
        frac = (step - momentum_cd_start) / muon_cooldown_steps
        momentum = momentum_max - frac * (momentum_max - momentum_min)
    else:
        momentum = momentum_max
    return momentum

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

# JSONL logger for mistakes
LOG_MISTAKES_EVERY = int(os.environ.get("LOG_MISTAKES_EVERY", "100"))
LOG_WARMUP_STEPS = int(os.environ.get("LOG_WARMUP_STEPS", "200"))
LOG_WARMUP_EVERY = int(os.environ.get("LOG_WARMUP_EVERY", "20"))

mistake_log_path: str | None = None
if master_process:
    os.makedirs("logs", exist_ok=True)
    mistake_log_path = f"logs/{run_id}_train_mistakes.jsonl"

# track loss trends for deceleration detection (master only)
_last_loss_mean: float | None = None
_last_loss_delta: float | None = None
_last_loss_delta_sign: float | None = None
_last_update_di: dict = {}

def log_mistakes(step: int, phase: str, details: dict | None, grad_stats: dict | None = None):
    """
    Append a single JSON record to logs/<run_id>_train_mistakes.jsonl.

    Pure logging:
    - no gradient ops
    - no in-place mutation of model state
    - safe to call under DDP (we only ever call on master_process)
    """
    if mistake_log_path is None or details is None:
        return

    record = {
        "step": int(step),
        "phase": phase,
    }
    record.update(details)
    if grad_stats:
        # grad_stats can be flat or nested; analyze_mistakes.py tolerates extra columns
        record.update(grad_stats)

    line = json.dumps(record, separators=(",", ":"))
    print(f"WRITING YOU FOOL", mistake_log_path)
    # Open and close per call so we don't keep a handle around for the whole run.
    with open(mistake_log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# torch.compile: keep enabled for memory efficiency; use a separate eager logging pass when ANALYSIS_MODE is on
if master_process and ANALYSIS_MODE:
    print("ANALYSIS_MODE: torch.compile stays on; a no-grad eager pass will capture logging payloads.")
train_model = torch.compile(model, dynamic=False, fullgraph=True)

# Eager-only helper used for logging so we can bypass torch.compile without graph breaks
@torch._dynamo.disable
@torch.no_grad()
def eager_log_forward(model, inputs, targets, cum_seqlens, ws_short, ws_long):
    return model(inputs, targets, cum_seqlens, ws_short, ws_long, log_stats=True)

# Warmup (unchanged)
warmup_steps = 30
initial_state = dict(
    model=copy.deepcopy(model.state_dict()),
    optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers],
)
train_loader = distributed_data_generator(
    args.train_files, args.train_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps
)
ws_schedule = list(args.ws_schedule) + [args.ws_final]
ws_long = ws_schedule[0]
for step in range(warmup_steps):
    inputs, targets, cum_seqlens = next(train_loader)
    ws_idx = step % len(ws_schedule)
    if ws_idx == 0:
        model.yarn.reset()
        ws_long = ws_schedule[0]
    else:
        new_ws_long = ws_schedule[ws_idx]
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long
    train_model(inputs, targets, cum_seqlens, ws_long // 2, ws_long).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.yarn.reset()
model.load_state_dict(initial_state["model"])
optimizer2.reset()
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

# Training & validation
train_loader = distributed_data_generator(
    args.train_files, args.train_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps
)
training_time_ms = 0
torch.cuda.synchronize()
t0 = time.perf_counter()
train_steps = args.num_iterations
ws_short, ws_long = get_ws(0)

for step in range(train_steps + 1):
    last_step = step == train_steps
    ws_short, new_ws_long = get_ws(step)
    if new_ws_long != ws_long:
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long

    # ----------------- VALIDATION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            ws_long = args.ws_validate_post_yarn_ext
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(
            args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False
        )
        val_loss = 0.0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, cum_seqlens = next(val_loader)
                loss = train_model(inputs, targets, cum_seqlens, ws_short, ws_long)
                val_loss += loss
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(
            f"step:{step}/{train_steps} val_loss:{val_loss:.4f} "
            f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms",
            console=True,
        )
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(
                step=step,
                code=code,
                model=model.state_dict(),
                optimizers=[opt.state_dict() for opt in optimizers],
            )
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        break

    # ----------------- TRAINING -----------------
    log_warmup = step < LOG_WARMUP_STEPS and (step % LOG_WARMUP_EVERY == 0)
    log_this_step = ANALYSIS_MODE and master_process and ((step % LOG_MISTAKES_EVERY == 0) or log_warmup)
    for idx in range(grad_accum_steps):
        if idx == grad_accum_steps - 1 and step % 2 == 1:
            optimizers[0].should_sync = True
        inputs, targets, cum_seqlens = next(train_loader)
        loss_mb = train_model(inputs, targets, cum_seqlens, ws_short, ws_long, log_stats=False) / grad_accum_steps
        loss_mb.backward()

        # --- Detailed mistake logging (analysis mode only) ---
        if log_this_step and idx == grad_accum_steps - 1:
            # Run an eager, no-grad copy of the forward just to populate analysis buffers
            _ = eager_log_forward(model, inputs, targets, cum_seqlens, ws_short, ws_long)

            details = compute_token_stats_from_topk(
                model._analysis_last_topk_logprobs,
                model._analysis_last_topk_indices,
                model._analysis_last_targets,
            )

            if details is not None:
                # loss trend signals (delta + deceleration)
                loss_mean = details.get("token_loss_mean")
                loss_delta = None
                loss_decel = None
                loss_delta_sign_flip = None
                if loss_mean is not None:
                    if _last_loss_mean is not None:
                        loss_delta = loss_mean - _last_loss_mean
                        details["loss_delta"] = loss_delta
                    if loss_delta is not None and _last_loss_delta is not None:
                        loss_decel = loss_delta - _last_loss_delta
                        details["loss_deceleration"] = loss_decel
                        # detect alternating sign (conflicting loss updates)
                        prev_sign = 0 if _last_loss_delta is None else math.copysign(1.0, _last_loss_delta) if _last_loss_delta != 0 else 0
                        cur_sign = 0 if loss_delta == 0 else math.copysign(1.0, loss_delta)
                        loss_delta_sign_flip = 1.0 if prev_sign != 0 and cur_sign != 0 and prev_sign != cur_sign else 0.0
                        details["loss_delta_sign_flip"] = loss_delta_sign_flip
                    _last_loss_mean = loss_mean
                    _last_loss_delta = loss_delta
                    _last_loss_delta_sign = loss_delta_sign_flip

                # logit balance signals (zero-sum-ish)
                if getattr(model, "_analysis_last_logit_mean", None) is not None:
                    details["logit_mean"] = float(model._analysis_last_logit_mean)
                if getattr(model, "_analysis_last_logit_mean_abs", None) is not None:
                    details["logit_mean_abs"] = float(model._analysis_last_logit_mean_abs)
                if getattr(model, "_analysis_last_logit_l2", None) is not None:
                    details["logit_l2_mean"] = float(model._analysis_last_logit_l2)
                if getattr(model, "_analysis_last_zero_sum", None) is not None:
                    details.update(model._analysis_last_zero_sum)
                if getattr(model, "_analysis_last_destructive", None) is not None:
                    details.update(model._analysis_last_destructive)
                if getattr(model, "_analysis_last_landscape", None) is not None:
                    details["loss_landscape"] = model._analysis_last_landscape
                if _last_update_di:
                    details.update(_last_update_di)

                details.update(
                    {
                        "lr": float(optimizers[0].param_groups[0]["lr"]),
                        "ws_short": ws_short,
                        "ws_long": ws_long,
                    }
                )
                grad_stats = compute_grad_stats(optimizers)
                # destructive interference by layer/type from parameter grads
                di_by_group = compute_destructive_by_group(model)
                if di_by_group:
                    details.update(di_by_group)

                log_mistakes(step=step, phase="train", details=details, grad_stats=grad_stats)
                print("LOGGING")
                # x = 1/0
            else:
                print(f"mistakes is none :(")
                x = 1/0

    step_optimizers(step, optimizers, model)

    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(
        f"step:{step+1}/{train_steps} "
        f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step+1):.2f}ms",
        console=True,
    )

print0(
    f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
    console=True,

)
dist.destroy_process_group()
