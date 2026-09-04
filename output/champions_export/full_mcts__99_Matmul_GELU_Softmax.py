import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/99_Matmul_GELU_Softmax_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.jit
def _matmul_gelu_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K,
                        stride_xm, stride_xk, stride_wn, stride_wk,
                        stride_om, stride_on,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        GROUP_M: tl.constexpr, EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(x_ptrs)
            b = tl.load(w_ptrs)
        else:
            k_rem = K - k * BLOCK_K
            a = tl.load(x_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
            b = tl.load(w_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        acc = tl.dot(a, b, acc=acc, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    bias = tl.load(b_ptr + offs_n)
    acc = acc + bias

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    z = acc * inv_sqrt2
    t = 1.0 / (1.0 + 0.3275911 * tl.abs(z))
    erf_val = 1.0 - (t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))) * tl.exp(-z * z)
    erf_val = tl.where(z >= 0, erf_val, -erf_val)
    gelu = 0.5 * acc * (1.0 + erf_val)

    # Store
    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, gelu, mask=mask_mn)


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, M, N,
                    stride_xm, stride_xn, stride_om, stride_on,
                    BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    # Single-pass softmax: load entire row, compute max, exp, sum, normalize
    x = tl.load(x_ptr + pid * stride_xm + offs_n * stride_xn, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    x = x - m
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_ptr + pid * stride_om + offs_n * stride_on, y, mask=mask)


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    w = _weights['linear.weight']  # [N, K]
    b = _weights['linear.bias']    # [N]
    M, K = x.shape
    N = w.shape[0]

    # --- Kernel 1: GEMM + bias + GELU ---
    out_gelu = torch.empty(M, N, device=x.device, dtype=torch.float32)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    EVEN_K = (K % BLOCK_K == 0)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _matmul_gelu_kernel[grid](
        x, w, b, out_gelu, M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out_gelu.stride(0), out_gelu.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, EVEN_K=EVEN_K,
        num_stages=3, num_warps=8,
    )

    # --- Kernel 2: Softmax (single pass, full row) ---
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)
    BLOCK_N_SM = 8192
    grid_sm = (M,)
    _softmax_kernel[grid_sm](
        out_gelu, out, M, N,
        out_gelu.stride(0), out_gelu.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N_SM,
        num_warps=16,
    )

    return out
