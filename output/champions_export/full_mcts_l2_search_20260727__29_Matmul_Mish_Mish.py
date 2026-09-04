import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/29_Matmul_Mish_Mish_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_mish_mish_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
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
        k_rem = K - k * BLOCK_K
        k_mask = offs_k < k_rem
        a = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc=acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # bias
    b_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(b_ptr + b_offs, mask=b_offs < N, other=0.0)
    acc = acc + bias[None, :]

    # mish 1: x * tanh(softplus(x)),  softplus(x) = max(x,0) + log(1+exp(-|x|))
    sp = tl.maximum(acc, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(acc)))
    e2 = tl.exp(2.0 * sp)
    th = 1.0 - 2.0 / (e2 + 1.0)
    acc = acc * th

    # mish 2
    sp2 = tl.maximum(acc, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(acc)))
    e2b = tl.exp(2.0 * sp2)
    th2 = 1.0 - 2.0 / (e2b + 1.0)
    acc = acc * th2

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    linear_weight = _weights['linear.weight']  # [N, K]
    linear_bias = _weights['linear.bias']      # [N]

    M, K = x.shape
    N = linear_weight.shape[0]

    y = torch.empty(M, N, device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    matmul_mish_mish_kernel[grid](
        x, linear_weight, linear_bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        linear_weight.stride(0), linear_weight.stride(1),
        y.stride(0), y.stride(1),
    )
    return y
