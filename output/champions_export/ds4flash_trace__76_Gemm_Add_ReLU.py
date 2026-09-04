import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/76_Gemm_Add_ReLU_weights.pt"
_weights = None
_device = None


@triton.jit
def _linear_bias_relu_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=a_mask,
            other=0.0,
        )
        # b[k, n] = w[n, k]
        b = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
            mask=b_mask,
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]
    acc = tl.maximum(acc, 0.0)

    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc,
        mask=y_mask,
    )


def run(x):
    global _weights, _device

    if _weights is None or _device != str(x.device):
        _weights = {
            k: v.to(x.device).contiguous()
            for k, v in torch.load(_weights_path, map_location="cpu", weights_only=True).items()
        }
        _device = str(x.device)

    bias = _weights["bias"]
    w = _weights["gemm.weight"]

    x = x.contiguous()
    M, K = x.shape
    N = w.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _linear_bias_relu_kernel[grid](
        x, w, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    return y
