import torch
import torch.nn as nn
import triton
import triton.language as tl

class Model(nn.Module):
    """
    Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/12_Gemm_Multiply_LeakyReLU_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)

@triton.jit
def gemm_mul_leaky_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    multiplier, negative_slope,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
):
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

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K and EVEN_M and EVEN_N:
            a = tl.load(x_ptrs)
            b = tl.load(w_ptrs)
        elif EVEN_K:
            a = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)
        else:
            k_mask = (k * BLOCK_K + offs_k) < K
            a = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc=acc, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if EVEN_N:
        b_bias = tl.load(b_ptr + offs_n)
    else:
        b_bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b_bias[None, :]

    acc = acc * multiplier

    acc = tl.where(acc >= 0.0, acc, acc * negative_slope)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if EVEN_M and EVEN_N:
        tl.store(out_ptrs, acc)
    else:
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    gemm_weight = _weights['gemm.weight']  # [out_features, in_features]
    gemm_bias = _weights['gemm.bias']      # [out_features]

    M, K = x.shape
    N = gemm_weight.shape[0]

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    EVEN_M = (M % BLOCK_M) == 0
    EVEN_N = (N % BLOCK_N) == 0
    EVEN_K = (K % BLOCK_K) == 0

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    gemm_mul_leaky_kernel[grid](
        x, gemm_weight, gemm_bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        gemm_weight.stride(0), gemm_weight.stride(1),
        out.stride(0), out.stride(1),
        multiplier, negative_slope,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_K=EVEN_K,
        num_stages=3, num_warps=8,
    )

    return out
