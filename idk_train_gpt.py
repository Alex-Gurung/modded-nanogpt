import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
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

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.empty(
    1, device="cuda", requires_grad=True
).backward()  # prevents a bug on some systems
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F

try:
    from analysis_utils import (
        ANALYSIS,
        compute_global_grad_and_param_norms,
        compute_optimizer_grad_stats,
        compute_layer_stats,
        zsl_gradient_probe,
    )
except Exception:
    x = 1/0
    # Fallback dummy so script still runs if analysis_utils is missing
    class _DummyAnalysis:
        enabled = False

    ANALYSIS = _DummyAnalysis()

    def compute_global_grad_and_param_norms(*args, **kwargs):
        return {}

    def compute_optimizer_grad_stats(*args, **kwargs):
        return {}

    def compute_layer_stats(*args, **kwargs):
        return {}

    def zsl_gradient_probe(*args, **kwargs):
        return {}

import json  # keep this import – used for logging later

# logger for analysis (initialized later once run_id/master_process exist)
stats_logger = None


# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
import triton
import triton.language as tl
from kernels import get_kernel
from torch import Tensor, nn

dynamo.config.recompile_limit = 64

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng


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
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
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
# Triton kernel for symmetric matrix multiplication by @byronxu99

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
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
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

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def XXT(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

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
    # This is mostly duplicated from XXT_kernel, but also loads and adds a block of A
    # Performance is slightly slower than XXT_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
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

# Computed for num_iters=5, safety_factor=2e-2, cushion=2
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile(dynamic=False, fullgraph=True) # Must use dynamic=False or else it's much slower
def polar_express(G: torch.Tensor):
    """
    Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.
    """
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)  # A = X @ X.mT
        ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X
        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# -----------------------------------------------------------------------------
# Muon optimizer

class NorMuon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).

    Differences from standard Muon:
    - Newton-Shulz is replaced with Polar Express for the orthogonalization step
    - NorMuon adds a low-rank variance estimator similar to Adafactor.
    - small 1D parameters handled here instead of in Adam
    - Cautious weight decay, a gated version of decoupled weight decay
    - Custom distributed sizing:
    The model stores all attn and mlp weights in the same shape, and then updates the view as
    needed on the forward pass. This enables attn and mlp weights to be contained within the same
    dist.reduce_scatter_tensor() call. The model architecture has been customized to enable
    (n_attn_layers+n_mlp_layers*2)%8==0 for batching across 8 GPUs with zero padding on mlp and attn.
    The scheduling is:
        1. reduce scatter smear_gate (1 param 7 padding params)
        2. reduce scatter attn_gate (10 params 6 padding params)
        3. reduce scatter attn/mlp round 1 (10 attn params 6 mlp params)
        4. reduce scatter attn/mlp round 2 (16 mlp params)
        5. wait on step 1, then compute update of 1 and schedule all gather
        6. wait on step 2, then compute update of 2 and schedule all gather
        7. wait on step 3, then compute update of 3 and schedule all gather
            GPUs receive [2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 MLP, 2 MLP, 2 MLP]
            GPUs that receive params of type attn reshape before computing update
        8. wait on 4, then compute update of 4 and schedule all gather
        9. wait for each all gather to complete and update params
    Empirically, leading with small params provides an additional 0.2s improvement.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, beta2=0.95, custom_sizing=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        # custom sizing requires 8 GPUs
        if custom_sizing and dist.get_world_size()==8:
            param_groups = self.generate_custom_param_groups(params)
        else:
            param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)

    def reset(self):
        # expose a reset for clearing buffers
        for group in self.param_groups:
            group["momentum_buffer"].zero_()
            group["second_momentum_buffer"].zero_()

    def generate_standard_param_groups(self, params):
        """
        Use this method if running on less than 8 GPU or experimenting with additional attn or mlp modules.
        Creates one param group per module.
        """
        groups = defaultdict(list)
        for param in params:
            groups[param.label].append(param)

        param_groups = []
        for module_name, group_params in groups.items():
            chunk_size = (len(group_params) + self.world_size - 1) // self.world_size
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))

        return param_groups

    def generate_custom_param_groups(self, params):
        """
        Implementation requires that a single GPU does not receive both attn
        and mlp params when a param group is split across GPUs.
        """
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

    @torch.no_grad()
    def step(self):
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, @vagrawal, and @varunneal.
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
                device=params[0].device
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
        # Second pass: wait for gradients, compute updates for the local shard of parameters,
        # and launch all async all_gather operations.
        for group, info in zip(self.param_groups, group_infos):
            info["reduce_future"].wait()

            params = group["params"]
            grad_chunk = info["grad_chunk"]
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            start_idx = rank * chunk_size
            module_idx = start_idx if start_idx < len(params) else 0

            num_params = min(chunk_size, max(0, len(params) - start_idx))  # num params for this rank

            if "momentum_buffer" not in group:
                group["momentum_buffer"]  = torch.zeros_like(grad_chunk[:num_params])
            momentum_buffer = group["momentum_buffer"]
            # Apply momentum update to the persistent momentum buffer in-place
            momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
            updated_grads = grad_chunk[:num_params].lerp_(momentum_buffer, group["momentum"])

            grad_shape = updated_grads.shape
            if params[module_idx].label == 'attn':
                # Reshape attn params from [hdim, dim*4] to [4,hdim,dim]
                for p in params[module_idx:module_idx + num_params]:
                    assert p.label == 'attn'
                updated_grads = updated_grads.view(4 * grad_shape[0], grad_shape[1], grad_shape[2] // 4)
            ref_param = params[module_idx]
            param_shape = ref_param.shape

            if "second_momentum_buffer" not in group:
                group["second_momentum_buffer"] = (torch.zeros_like(updated_grads[..., :, :1])
                    if param_shape[-2] >= param_shape[-1] else torch.zeros_like(updated_grads[..., :1, :])
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

            # Determine LR and WR
            eff_lr = group["lr"] * group["param_lr"]
            eff_wd = group["lr"] * group["weight_decay"] * group["param_wd"]

            # Compute zeropower for the entire chunk in a single, batched call.
            if num_params == 0:
                v_chunk = updated_grads
            else:
                v_chunk = polar_express(updated_grads)

            # NorMuon: second_momentum_buffer tracks squared magnitude of gradients along one dim (https://arxiv.org/pdf/2510.05491)
            v_norm = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_mean = v_chunk.square().mean(dim=-1 if param_shape[-2] >= param_shape[-1] else -2, keepdim=True)
            second_momentum_buffer.lerp_(v_mean.to(dtype=ref_param.dtype), 1 - group["beta2"])
            step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
            v_chunk.mul_(step_size)
            v_norm_new = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_chunk.mul_(v_norm / v_norm_new.clamp_min_(1e-10))

            v_chunk = v_chunk.view(grad_shape)

            updated_params = torch.empty_like(grad_chunk)
            param_chunk = torch.stack(params[module_idx:module_idx + num_params]) if num_params > 0 else torch.zeros_like(v_chunk)

            # "Cautious" weight decay (https://arxiv.org/abs/2510.12402)
            mask = (v_chunk * param_chunk) >= 0
            v_chunk.addcmul_(param_chunk, (eff_wd * mask).to(ref_param.dtype))

            param_chunk.addcmul_(v_chunk, -eff_lr)

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
                {
                    "gather_future": gather_future,
                    "stacked_params": stacked_params,
                    "orig_params": params,
                }
            )

        # Final pass: wait for all_gather to complete and copy results back into original parameter tensors.
        for info in all_gather_infos:
            info["gather_future"].wait()
            stacked_params = info["stacked_params"]
            orig_params = info["orig_params"]

            unstacked_params = torch.unbind(stacked_params)
            for i, p in enumerate(orig_params):
                p.copy_(unstacked_params[i], non_blocking=True)


class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # init state
        for p in params:
            chunk_size = p.size(0) // self.world_size
            exp_avg = torch.zeros_like(p[:chunk_size], dtype=torch.bfloat16, device=p[0].device)
            exp_avg_sq = torch.zeros_like(exp_avg)
            self.state[p] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)
        # DistributedAdam implementation by @vagrawal, @akash5474

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
            grad_slice
        )

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        all_gather_futures: list[torch.Future] = []

        for group in reversed(self.param_groups):
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for param in reversed(group['params']):
                if param not in self._reduce_scatter_futures:
                    continue

                fut, g_slice = self._reduce_scatter_futures[param]
                fut.wait()

                rank_size = param.shape[0] // self.world_size
                p_slice = param[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(param, "lr_mul", 1.0)
                state = self.state[param]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (bias2 ** 0.5 / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)

                all_gather_futures.append(dist.all_gather_into_tensor(param, p_slice, async_op=True).get_future())

        self._reduce_scatter_futures.clear()
        torch.futures.collect_all(all_gather_futures).wait()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

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
            self.weight.zero_()  # @Grad62304977 and others

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

# yarn implementation @classiclarryd
class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(
            theta.cos().to(torch.bfloat16), persistent=False
        )
        self.sin = nn.Buffer(
            theta.sin().to(torch.bfloat16), persistent=False
        )
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
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

flash_attn_interface = get_kernel('varunneal/flash-attention-3').flash_attn_interface

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim

        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std  # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        # make matrices the same shape as MLP to enable batched call in optimizer
        self.qkvo_w = nn.Parameter(torch.empty(self.hdim, self.dim * 4))
        # label module to enable custom optimizer sizing
        self.qkvo_w.label = 'attn'

        with torch.no_grad():
            self.qkvo_w.view(4, self.hdim, self.dim)[:3].uniform_(-bound, bound)  # init QKV weights
            self.qkvo_w.view(4, self.hdim, self.dim)[3].zero_()  # init output weights to zero

        # sparse gated attention to enable context based no-op by @classiclarryd
        self.attn_gate = CastedLinear(12, num_heads)
        # label module to enable custom optimizer sizing
        self.attn_gate.weight.label = 'attn_gate'

        # ---- analysis buffers for gate stats ----
        if ANALYSIS.enabled:
            # NOTE: no .cpu() here; let buffers live on the module's device
            self.register_buffer("gate_mean_ema", torch.zeros(num_heads))
            self.register_buffer("gate_sq_mean_ema", torch.zeros(num_heads))
            self.register_buffer("gate_ema_count", torch.tensor(0.0))

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1)  # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        q, k, v = F.linear(
            x,
            self.qkvo_w
                .view(4, self.hdim, self.dim)[:3]
                .flatten(end_dim=1)
                .type_as(x),
        ).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)

        q, k = norm(q), norm(k)  # QK norm @Grad62304977
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)  # @ KoszarskyB & @Grad62304977
        else:  # skip mid-layers token value embeddings by @YouJiacheng
            v = sa_lambdas[0] * v

        max_len = args.train_max_seq_len if self.training else (
            args.val_batch_size // (grad_accum_steps * world_size)
        )

        # use flash_attn over flex_attn @varunneal. flash_attn_varlen suggested by @YouJiacheng
        y = flash_attn_interface.flash_attn_varlen_func(
            q[0], k[0], v[0],
            cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
            max_seqlen_q=max_len, max_seqlen_k=max_len,
            causal=True, softmax_scale=attn_scale, window_size=(bm_size, 0),
        )
        # y is [T, num_heads, head_dim] for B==1; reshape to [B, T, H, D]
        y = y.view(B, T, self.num_heads, self.head_dim)

        # compute gates once
        gates = torch.sigmoid(
            self.attn_gate(x[..., : self.attn_gate.weight.size(-1)])
        )  # [B, T, num_heads]

        # ---- analysis: gate EMAs, done on the same device as the model ----
        if ANALYSIS.enabled:
            with torch.no_grad():
                # stay on-device; no .cpu()
                g_det = gates.detach()
                mean = g_det.mean(dim=(0, 1))          # [num_heads]
                sq_mean = (g_det * g_det).mean(dim=(0, 1))
                c = self.gate_ema_count
                new_c = c + 1.0
                alpha = 1.0 / new_c
                one_minus = 1.0 - alpha
                self.gate_mean_ema.mul_(one_minus).add_(mean, alpha=alpha)
                self.gate_sq_mean_ema.mul_(one_minus).add_(sq_mean, alpha=alpha)
                self.gate_ema_count.copy_(new_c)

        # apply gate exactly once
        y = y * gates.view(B, T, self.num_heads, 1)

        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)  # re-assemble all head outputs side by side
        y = F.linear(y, self.qkvo_w.view(4, self.hdim, self.dim)[3].type_as(y))
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        # make matrices the same shape to enable batched call in optimizer
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        # label modules to enable custom optimizer sizing
        self.c_fc.label = 'mlp'
        self.c_proj.label = 'mlp'
        # corrective factor to account for transpose
        self.c_fc.lr_mul = 2.

        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std  # improved init scale by @YouJiacheng
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()  # zero init suggested by @Grad62304977

        # ---- analysis buffers ----
        if ANALYSIS.enabled:
            hdim = 4 * dim
            # also on the module's device (no .cpu())
            self.register_buffer("act_mean_ema", torch.zeros(hdim))
            self.register_buffer("act_sq_mean_ema", torch.zeros(hdim))
            self.register_buffer("act_ema_count", torch.tensor(0.0))

    def forward(self, x: Tensor):
        x = F.linear(x, self.c_fc.T.type_as(x))
        h = F.relu(x).square()

        if ANALYSIS.enabled:
            with torch.no_grad():
                # h: [B, T, hdim] on the same device as the model
                h_det = h.detach()
                # average across batch & time
                mean = h_det.mean(dim=(0, 1))       # [hdim]
                sq_mean = (h_det * h_det).mean(dim=(0, 1))

                # update running averages in-place
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
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx not in [0, 7] else None
        # skip MLP blocks for first MLP layer by @EmelyanenkoK
        self.mlp = MLP(dim) if layer_idx != 0 else None

    def forward(self, x: Tensor, x0: Tensor, lambdas: Tensor, attn_args: AttnArgs):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.smear_gate = CastedLinear(12, 1)
        # label modules to enable custom optimizer sizing
        self.smear_gate.weight.label = 'smear_gate'
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
        self.yarn = Yarn(head_dim, max_seq_len)
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=use_fp8, x_s=(model_dim**0.5)/448, w_s=2**-9, grad_s=1/448)
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        pad = (-num_layers * 5 - 2) % dist.get_world_size()
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    -1.5
                    * torch.ones(num_layers),  # skip_weights -> σ(-1.5) ≈ 0.18
                    *[
                        torch.tensor([1.1, 0.0]) for _ in range(num_layers) 
                    ],  # block lambdas. 1.1 init such that layer i weight is i^(num_layers-i). 
                        # ~3x higher weight to layer 1 compared to 12 at init.
                    *[
                        torch.tensor([0.5, 0.5]) for _ in range(num_layers)
                    ],  # SA lambdas
                    torch.zeros(1), # smear_lambda
                    0.5*torch.ones(1), # backout_lambda
                    torch.ones(pad),
                ]
            )
        )
        # set learning rates
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0


        self._log_stats = False
        self._last_logits_for_stats = None
        self._last_targets_for_stats = None

        # --- analysis buffers (for token-level stats) ---

        # Old-style (full logits) analysis fields — kept for compatibility.
        # We set these to None and no longer populate them by default.
        self._analysis_last_logits = None          # [N, V] or [B, T, V] in the old code
        self._analysis_last_targets = None         # [N] or [B, T]

        # New compressed representation:
        # - token_indices: flat positions into [B*T]
        # - topk_indices: token ids of top-k predictions per sampled token
        # - topk_logprobs: log-probs within the top-k slice
        # - targets: ground-truth token ids for those sampled tokens
        self._analysis_last_token_indices = None   # [Nsamp]
        self._analysis_last_topk_indices = None    # [Nsamp, K]
        self._analysis_last_topk_logprobs = None   # [Nsamp, K]
        self._analysis_last_targets = None         # [Nsamp]



    @torch.no_grad()
    def _stash_analysis_view(
        self,
        logits_for_loss: torch.Tensor,
        target_seq: torch.Tensor,
        *,
        default_topk: int = 20,
        default_max_tokens: int = 4096,
    ):
        """
        Compress the batch into a logging-friendly view:

        - Flatten [B, T, V] -> [N, V] (B is usually 1 here).
        - Optionally subsample up to `max_tokens` tokens.
        - For each sampled token, store top-k log-probs + indices and the target id.
        - Everything is detached and moved to CPU for logging.

        This is *pure logging* (no grad), and only runs when ANALYSIS.enabled and
        not inside a torch.compile graph.
        """
        if logits_for_loss.ndim != 3:
            # Expect [B, T, V]
            return

        B, T, V = logits_for_loss.shape
        device = logits_for_loss.device

        # Flatten over batch and time
        logits_flat = logits_for_loss.view(-1, V)     # [N, V]
        targets_flat = target_seq.view(-1)            # [N]
        N = logits_flat.size(0)

        # Allow overrides via env vars if you want to tweak without editing code
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

        # Subsample tokens for logging if needed
        if N > max_tokens:
            # Random subset of positions
            sample_idx = torch.randperm(N, device=device)[:max_tokens]
            logits_sampled = logits_flat.index_select(0, sample_idx)
            targets_sampled = targets_flat.index_select(0, sample_idx)
        else:
            sample_idx = torch.arange(N, device=device)
            logits_sampled = logits_flat
            targets_sampled = targets_flat

        # Compute top-k logits per sampled token
        # Shape: [Nsamp, topk]
        topk_vals, topk_idx = torch.topk(logits_sampled, k=topk, dim=-1)  # both on `device`

        # Turn them into log-probs *over the top-k slice* (normalized among the topk only).
        # This avoids materializing full [Nsamp, V] float32 log_probs.
        topk_logprobs = torch.log_softmax(topk_vals.float(), dim=-1)      # [Nsamp, topk], float32

        # Stash to CPU for external analysis/logging
        self._analysis_last_token_indices = sample_idx.detach().cpu()        # [Nsamp]
        self._analysis_last_topk_indices = topk_idx.detach().cpu()          # [Nsamp, topk]
        self._analysis_last_topk_logprobs = topk_logprobs.detach().cpu()    # [Nsamp, topk]
        self._analysis_last_targets = targets_sampled.detach().cpu()        # [Nsamp]


    def forward(
        self,
        input_seq: Tensor,
        target_seq: Tensor,
        seqlens: Tensor,
        ws_short: int,
        ws_long: int,
        log_stats: bool = False,
    ):
        # clear logging state for this call
        self._log_stats = log_stats
        self._last_logits_for_stats = None
        self._last_targets_for_stats = None

        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings
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
        lambdas = self.scalars[1 * len(self.blocks) : 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks) : 5 * len(self.blocks)].view(-1, 2)
        smear_lambda = self.scalars[5 * len(self.blocks)]
        backout_lambda = self.scalars[5 * len(self.blocks) + 1]

        # smear token embed forward 1 position
        smear_gate_out = smear_lambda * torch.sigmoid(
            self.smear_gate(x[1:, : self.smear_gate.weight.size(-1)])
        )
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        # skip connection from layer 4 to layer 7
        skip_connections = []
        n = len(self.blocks) // 2
        skip_in = [4]
        skip_out = [7]

        x_backout = None
        backout_layer = 8
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
            if i in skip_out:
                gate = torch.sigmoid(skip_weights[i - n])  # in (0,1)
                x = x + gate * skip_connections.pop()
            x = self.blocks[i](x, x0, lambdas[i], attn_args)
            if i in skip_in:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        # back out contributions from early layers used only for context
        x -= backout_lambda * x_backout
        x = norm(x)
        logits = self.lm_head(x)
        # same tanh-like softcapping as original
        logits = 30 * torch.sigmoid(logits / 7.5)
        logits_for_loss = logits.float() if not self.training else logits

        # --- analysis: stash a *compressed* view of this batch ---
        # We:
        #   - optionally subsample tokens
        #   - keep only top-k log-probs + indices per token
        #   - move just that to CPU
        # if ANALYSIS.enabled and not torch._dynamo.is_compiling():
        if ANALYSIS.enabled and not torch._dynamo.is_compiling():
            self._stash_analysis_view(logits_for_loss, target_seq)

        # === ORIGINAL TRAINING LOSS (unchanged gradients) ===
        loss = F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            target_seq.view(-1),
            reduction="sum" if self.training else "mean",
        )

        return loss


def compute_token_stats_from_logits(
    logits_for_loss: torch.Tensor,
    target_seq: torch.Tensor,
    chunk_size: int = 256,
    *,
    sample_topk: int = 20,
) -> dict:
    """
    Memory-safe token-level stats.

    - Works directly on logits (GPU or CPU).
    - Never materializes full [N, V] log_probs or probs.
    - Processes in chunks along the token dimension.
    - Additionally logs top-k predictions for a small random sample of tokens.

    Args:
        logits_for_loss: [B, T, V] or [N, V] tensor of logits.
        target_seq:      [B, T] or [N] tensor of target token ids.
        chunk_size:      how many tokens per chunk when computing stats.
        sample_topk:     if > 0, for the sampled tokens in `stats["sample"]`,
                         also record their top-k logprobs + indices.
    """
    with torch.no_grad():
        # Flatten over batch/time to [N, V]; work on whatever device logits are on
        vocab_size = logits_for_loss.size(-1)
        logits_flat = logits_for_loss.view(-1, vocab_size)
        targets_flat = target_seq.view(-1)

        N = logits_flat.size(0)

        # Collect per-token scalars (small 1D tensors)
        loss_chunks = []
        p_target_chunks = []
        entropy_chunks = []
        top1_p_chunks = []
        top2_p_chunks = []
        correct_chunks = []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)

            logits_chunk = logits_flat[start:end]          # [C, V]
            targets_chunk = targets_flat[start:end]        # [C]

            # [C, V]
            log_probs_chunk = F.log_softmax(logits_chunk, dim=-1)
            probs_chunk = log_probs_chunk.exp()

            # per-token target logp / p
            idx_local = torch.arange(end - start, device=logits_chunk.device)
            target_logp_chunk = log_probs_chunk[idx_local, targets_chunk]
            target_p_chunk = target_logp_chunk.exp()
            loss_chunk = -target_logp_chunk                 # cross-entropy per token

            # entropy per token
            entropy_chunk = -(probs_chunk * log_probs_chunk).sum(dim=-1)

            # top-2 and correctness
            top_vals, top_idx = probs_chunk.topk(2, dim=-1)
            top1 = top_idx[:, 0]
            top1_p = top_vals[:, 0]
            top2_p = top_vals[:, 1]
            correct_chunk = (top1 == targets_chunk).float()

            loss_chunks.append(loss_chunk)
            p_target_chunks.append(target_p_chunk)
            entropy_chunks.append(entropy_chunk)
            top1_p_chunks.append(top1_p)
            top2_p_chunks.append(top2_p)
            correct_chunks.append(correct_chunk)

            # (log_probs_chunk, probs_chunk, etc. go out of scope here)

        # Concatenate per-token scalars (still on same device as logits)
        token_loss = torch.cat(loss_chunks, dim=0)       # [N]
        target_p = torch.cat(p_target_chunks, dim=0)
        entropy = torch.cat(entropy_chunks, dim=0)
        top1_p = torch.cat(top1_p_chunks, dim=0)
        top2_p = torch.cat(top2_p_chunks, dim=0)
        correct = torch.cat(correct_chunks, dim=0)

        # Quantiles (on same device, then moved to CPU)
        def quantiles(x, qs):
            q_t = torch.quantile(x, torch.tensor(qs, device=x.device))
            return [float(v) for v in q_t.cpu().tolist()]

        loss_q50, loss_q90, loss_q99 = quantiles(token_loss, [0.5, 0.9, 0.99])
        p_q10, p_q50, p_q90 = quantiles(target_p, [0.1, 0.5, 0.9])

        # High-confidence behaviour
        high_conf = (top1_p >= 0.9)
        frac_high_conf = float(high_conf.float().mean().item())
        frac_high_conf_wrong = float(((high_conf) & (correct < 0.5)).float().mean().item())

        overconf_wrong_frac = frac_high_conf_wrong
        underconf_right_frac = float(((correct > 0.5) & (target_p <= 0.5)).float().mean().item())

        stats = {
            "num_tokens": int(token_loss.numel()),
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
            "frac_solved_95": float((target_p > 0.95).float().mean().item()),
            "frac_solved_99": float((target_p > 0.99).float().mean().item()),
            "entropy_mean": float(entropy.mean().item()),
            "entropy_std": float(entropy.std(unbiased=False).item()),
            "top2_gap_mean": float((top1_p - top2_p).mean().item()),
            "top2_gap_std": float((top1_p - top2_p).std(unbiased=False).item()),
            "overconf_wrong_frac": float(overconf_wrong_frac),
            "underconf_right_frac": float(underconf_right_frac),
            # these names are what analyze_mistakes.py expects
            "token_loss_mean": float(token_loss.mean().item()),
            "token_loss_q50": float(loss_q50),
            "token_loss_q90": float(loss_q90),
            "accuracy": float(correct.mean().item()),
            "p_correct_mean": float(target_p.mean().item()),
            "frac_high_conf": float(frac_high_conf),
            "frac_high_conf_wrong": float(frac_high_conf_wrong),
        }

        # Tiny random sample for debugging (unchanged)
        sample_size = min(32, token_loss.numel())
        perm = torch.randperm(token_loss.numel(), device=token_loss.device)[:sample_size]

        sample_dict = {
            "loss": [float(v) for v in token_loss[perm].cpu().tolist()],
            "p_target": [float(v) for v in target_p[perm].cpu().tolist()],
            "entropy": [float(v) for v in entropy[perm].cpu().tolist()],
            "correct": [int(v) for v in correct[perm].cpu().tolist()],
        }

        # --- NEW: top-k predictions for the same sampled tokens ---
        if sample_topk is not None and sample_topk > 0 and vocab_size > 0 and sample_size > 0:
            k = min(sample_topk, vocab_size)

            # [S, V] logits for the sampled tokens
            logits_sample = logits_flat[perm]                        # same device as logits
            log_probs_sample = F.log_softmax(logits_sample.float(), dim=-1)  # [S, V] float32

            topk_vals, topk_idx = torch.topk(log_probs_sample, k=k, dim=-1)  # [S, k]

            sample_dict["topk_indices"] = [
                [int(t) for t in row] for row in topk_idx.cpu().tolist()
            ]
            sample_dict["topk_logprobs"] = [
                [float(v) for v in row] for row in topk_vals.cpu().tolist()
            ]

        stats["sample"] = sample_dict

        return stats


# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

BOS_ID = 50256

class BOSFinder:
    # Helper for getting sequences that start at the beginning of documents by @varunneal based on work by @classiclarryd
    def __init__(self, tokens: Tensor, world_size: int = 1, quickload: bool = False):
        # Precompute BOS positions once per shard
        self.tokens=tokens
        self.size = tokens.numel()
        self.quickload = quickload
        if quickload:
            # only scan first 4 million tokens, then kickoff async thread to scan rest
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
        # if quickload was used, repoint to the full dataset after 5 batches
        if self.quickload and self.batch_iter==5:
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
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
                idx += 1

            assert cur_len == num_tokens_local + 1
        self.i = idx
        self.batch_iter+=1
        return starts, ends

class DataPreloader:
    # Helper for asynchronously loading next shard and indexing bos tokens
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

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        finder = BOSFinder(tokens, world_size=world_size, quickload=True)
        preloader = DataPreloader(file_iter, world_size)
        preloader.start()
    else:
        pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)  # median doc length is ~400

        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                tokens, finder = preloader.get()
                preloader.start()
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens


        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        new_params = yield (
            _inputs.to(device="cuda", dtype=torch.int32, non_blocking=True),
            _targets.to(device="cuda", dtype=torch.int64, non_blocking=True),
            _cum_lengths.to(device="cuda", dtype=torch.int32, non_blocking=True)
        )

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len, grad_accum_steps) via .send()
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens
            max_seq_len = new_max_seq_len
            grad_accum_steps = new_grad_accum_steps


# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files: str = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files: str = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens: int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_batch_size: int = 2048 * 16 * 8
    # train_batch_size: int = 2048 * 16
    # train_batch_size: int = 2048 * 16 * 8 * 2
    train_max_seq_len: int = 128 * 16
    # train_max_seq_len: int = 128 * 8
    val_batch_size: int = 4 * 64 * 1024 * 8
    # optimization
    # num_scheduled_iterations: int = 2185  # number of steps to complete lr and ws schedule
    num_scheduled_iterations: int = 1000 # number of steps to complete lr and ws schedule
    # num_scheduled_iterations: int = 500 # number of steps to complete lr and ws schedule
    num_extension_iterations: int = 40  # number of steps to continue training at final lr and ws
    num_iterations: int = num_scheduled_iterations + num_extension_iterations
    cooldown_frac: float = 0.50  # fraction of num_scheduled_iterations spent cooling down the learning rate
    # evaluation and logging
    run_id: str = f"{uuid.uuid4()}"
    # val_loss_every: int = 250  # every how many steps to evaluate val loss? 0 for only at the end
    val_loss_every: int = 100 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint: bool = False
    # attention masking
    block_size: int = 128
    ws_schedule: tuple = (3, 7, 11)
    ws_final: int = 13 # increase final validation ws, used for YaRN extension and short window size @classiclarryd
    ws_validate_post_yarn_ext: int = 20 # extend long windows out even further after applying YaRN



args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert 8 % world_size == 0, "world_size must be a divisor of 8"
grad_accum_steps = 8 // world_size
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = args.run_id
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# Optional secondary analysis logger (currently unused, but safe to keep around)
if ANALYSIS.enabled and master_process:
    os.makedirs("logs", exist_ok=True)
    stats_logger = open(f"logs/{run_id}_analysis.jsonl", "a", buffering=1)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running Triton version {triton.__version__}")

def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

model: nn.Module = GPT(
    vocab_size=50257,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=max(args.train_batch_size, args.val_batch_size) // (grad_accum_steps * world_size)
).cuda()
for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n and "gate" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]
gate_params = [p for n, p in model.named_parameters() if "gate" in n]

# init the optimizer(s)
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = DistAdam(
    scalar_params + head_params + embed_params,
    lr=0.008,
    betas=(0.65, 0.95),
    eps=1e-8,
    weight_decay=0.0,
)
optimizer2 = NorMuon(hidden_matrix_params + gate_params, lr=0.03, momentum=0.95, beta2=0.95, weight_decay=1.2)
# optimizer2 = NorMuon(hidden_matrix_params + gate_params, lr=0.06, momentum=0.95, beta2=0.95, weight_decay=1.2)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: flat, then linear decay, then flat
def get_lr(step: int):
    x = min(0.9999, step / args.num_scheduled_iterations)
    assert 0 <= x < 1
    lr = 1.0
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = w * 1.0 + (1 - w) * 0.1
    return lr

def get_ws(step: int):
    # set short window size to half of long window size
    # Higher ws on "extension" steps
    if step >= args.num_scheduled_iterations:
        return args.ws_final // 2, args.ws_final
    x = step / args.num_scheduled_iterations
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx] // 2, args.ws_schedule[ws_idx]

def get_muon_momentum(step: int, muon_warmup_steps=300, muon_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95):
    # warmup phase: linearly increase momentum from min to max
    # cooldown phase: linearly decrease momentum from max to min
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

def step_optimizers(step: int, optimizers, model):
    # update lr
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)

    # set muon momentum based on step
    momentum = get_muon_momentum(step)
    for group in optimizers[1].param_groups:
        group["momentum"] = momentum

    # on even steps, only step Muon params
    # on odd steps, step all params
    if step%2==0:
        optimizers[1].step()
        optimizers[1].zero_grad(set_to_none=True)
    else:
        for optimizer in optimizers:
            optimizer.step()
        model.zero_grad(set_to_none=True)
        # disable sync in the next training step for the adam optimizer
        optimizers[0].should_sync = False

LOG_MISTAKES_EVERY = int(os.environ.get("LOG_MISTAKES_EVERY", "100"))

def compute_grad_stats(optimizers):
    """Aggregate basic gradient stats per optimizer (per-rank)."""
    stats = {}
    for opt_idx, opt in enumerate(optimizers):
        total_sq = 0.0
        max_abs = 0.0
        for group in opt.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach().cpu()
                gf = g.float()
                total_sq += float(gf.pow(2).sum().item())
                max_abs = max(max_abs, float(gf.abs().max().item()))
        stats[f"opt{opt_idx}_grad_norm"] = math.sqrt(total_sq) if total_sq > 0 else 0.0
        stats[f"opt{opt_idx}_grad_max"] = max_abs
    return stats

mistake_logger = None
mistake_log_path = None

if master_process:
    os.makedirs("logs", exist_ok=True)
    mistake_log_path = f"logs/{run_id}_train_mistakes.jsonl"
    # line-buffered so you see updates even if job crashes
    mistake_logger = open(mistake_log_path, "a", buffering=1)

def log_mistakes(
    step: int,
    phase: str,
    logits: torch.Tensor | None,
    targets: torch.Tensor | None,
    grad_stats: dict | None = None,
):
    """Write one JSONL record with token-level and grad-level stats."""
    if mistake_logger is None or logits is None or targets is None:
        return

    # all of this is OUTSIDE the compiled graph
    token_stats = compute_token_stats_from_logits(logits, targets)

    rec = {
        "step": int(step),
        "phase": str(phase),
        **token_stats,
    }
    if grad_stats is not None:
        rec.update(grad_stats)

    mistake_logger.write(json.dumps(rec) + "\n")
    mistake_logger.flush()


model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)
if not ANALYSIS.enabled:
    model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)
else:
    print0("ANALYSIS_MODE: torch.compile is disabled for richer logging", console=True)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 30
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
train_loader = distributed_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
ws_schedule = list(args.ws_schedule) + [args.ws_final]
ws_long = ws_schedule[0]
for step in range(warmup_steps):
    inputs, targets, cum_seqlens = next(train_loader)
    # each window size is a new graph, need to warm up each with Yarn.attn_scale
    ws_idx = step % len(ws_schedule)
    if ws_idx==0:
        model.yarn.reset()
        ws_long = ws_schedule[0]
    else:
        new_ws_long = ws_schedule[ws_idx]
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long
    model(inputs, targets, cum_seqlens, ws_long//2, ws_long).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.yarn.reset() # rotary buffer is not stored in state_dict
model.load_state_dict(initial_state["model"])
optimizer2.reset() # muon momentum buffers not in state dict
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
ws_short, ws_long = get_ws(0)
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    # should we log detailed stats this step?
    log_this_step = (
        ANALYSIS.enabled
        and master_process
        and (step % LOG_MISTAKES_EVERY == 0)
    )
    ws_short, new_ws_long = get_ws(step)
    if new_ws_long != ws_long:
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            ws_long = args.ws_validate_post_yarn_ext
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for val_idx in range(val_steps):
                inputs, targets, cum_seqlens = next(val_loader)
                # only log stats for the first val microbatch when requested
                log_stats = log_this_step and (val_idx == 0)
                loss_batch = model(
                    inputs,
                    targets,
                    cum_seqlens,
                    ws_short,
                    ws_long,
                    log_stats=log_stats,
                )
                val_loss += loss_batch
                if (
                    ANALYSIS.enabled
                    and master_process
                    and val_idx == 0
                    and stats_logger is not None
                    and model._analysis_last_logits is not None
                    and model._analysis_last_targets is not None
                ):
                    record = {
                        "type": "val_step",
                        "step": int(step),
                    }
                    token_stats = compute_token_stats_from_logits(
                        model._analysis_last_logits,
                        model._analysis_last_targets,
                    )
                    record.update(token_stats)
                    stats_logger.write(json.dumps(record) + "\n")
                    model._analysis_last_logits = None
                    model._analysis_last_targets = None
        # after the val loop, if we logged stats, write a "val" row
        if log_this_step and master_process and model._last_logits_for_stats is not None:
            log_mistakes(
                step=step,
                phase="val",
                logits=model._last_logits_for_stats,
                targets=model._last_targets_for_stats,
                grad_stats=None,
            )
            model._last_logits_for_stats = None
            model._last_targets_for_stats = None

        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    last_inputs = None
    last_targets = None
    last_cum_seqlens = None

    for idx in range(grad_accum_steps):
        if idx == grad_accum_steps - 1 and step % 2 == 1:
            optimizers[0].should_sync = True

        inputs, targets, cum_seqlens = next(train_loader)
        last_inputs, last_targets, last_cum_seqlens = inputs, targets, cum_seqlens

        # only ask the model to stash logits on the last microstep we care about
        log_stats = log_this_step and (idx == grad_accum_steps - 1)

        (model(inputs, targets, cum_seqlens, ws_short, ws_long, log_stats=log_stats) / grad_accum_steps).backward()

    # after backward, if we're logging this step, compute grad stats + token stats
    if log_this_step and master_process and model._last_logits_for_stats is not None:
        grad_stats = compute_grad_stats(optimizers)
        log_mistakes(
            step=step,
            phase="train",
            logits=model._last_logits_for_stats,
            targets=model._last_targets_for_stats,
            grad_stats=grad_stats,
        )
        model._last_logits_for_stats = None
        model._last_targets_for_stats = None

    # ---- Analysis logging before stepping ----
    if ANALYSIS.enabled and master_process and (step % ANALYSIS.log_every == 0):
        record = {
            "type": "train_step",
            "step": int(step),
        }
        # global loss for this step (approx: using last microbatch's loss/num_tokens)
        with torch.no_grad():
            # use token-level stats from last forward
            if model._analysis_last_logits is not None and model._analysis_last_targets is not None:
                token_stats = compute_token_stats_from_logits(
                    model._analysis_last_logits,
                    model._analysis_last_targets,
                )
                record.update(token_stats)
                # clear to avoid stale
                model._analysis_last_logits = None
                model._analysis_last_targets = None

        # gradient & layer stats (need grads before step)
        record.update(compute_global_grad_and_param_norms(model))
        record.update(compute_optimizer_grad_stats(optimizers))
        record["layers"] = compute_layer_stats(model)

        # occasional ZSL probe
        if ANALYSIS.zsl_every > 0 and (step % ANALYSIS.zsl_every == 0):
            try:
                zsl_stats = zsl_gradient_probe(
                    model,
                    last_inputs,
                    last_targets,
                    last_cum_seqlens,
                    ws_short,
                    ws_long,
                    param_selector="lm_head",
                    max_examples=ANALYSIS.zsl_batch_size,
                )
                record.update(zsl_stats)
            except RuntimeError as e:
                record["zsl_error"] = str(e)

        if stats_logger is not None:
            stats_logger.write(json.dumps(record) + "\n")

    step_optimizers(step, optimizers, model)

    # logging (unchanged)
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

if ANALYSIS.enabled and master_process and stats_logger is not None:
    stats_logger.close()

if master_process and mistake_logger is not None:
    mistake_logger.close()

dist.destroy_process_group()
