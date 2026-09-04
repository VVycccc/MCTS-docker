import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def accel_l2_12_Gemm_Multiply_LeakyReLU_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_wk: tl.constexpr, stride_wn: tl.constexpr,
    stride_om: tl.constexpr, stride_on: tl.constexpr,
    multiplier, negative_slope,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + offs_k

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn

        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)

        accumulator = tl.dot(x, w, acc=accumulator, allow_tf32=True)

    bias = tl.load(b_ptr + offs_n).to(tl.float32)
    accumulator += bias[None, :]

    accumulator *= multiplier

    accumulator = tl.where(accumulator >= 0, accumulator, accumulator * negative_slope)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, accumulator)


batch_size = 1024
in_features  = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/12_Gemm_Multiply_LeakyReLU_weights.pt"
_weights = None
_weight_t = None
_bias = None

def run(x, *args):
    global _weights, _weight_t, _bias
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _weights = {k: v.to(x.device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
        w = _weights['gemm.weight']
        b = _weights['gemm.bias']
        _weight_t = w.T.contiguous()
        _bias = b.contiguous()

    M, K = x.shape
    N = _weight_t.shape[1]

    x = x.contiguous()
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    accel_l2_12_Gemm_Multiply_LeakyReLU_kernel[grid](
        x, _weight_t, _bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        _weight_t.stride(0), _weight_t.stride(1),
        out.stride(0), out.stride(1),
        multiplier, negative_slope,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_M=GROUP_M,
        num_stages=3,
        num_warps=8,
    )

    return out
