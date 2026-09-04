import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel(
    x_ptr, w_ptr, bias_ptr, add_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
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

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, w, acc=acc, allow_tf32=True)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(bias_ptr + offs_n)
    add_val = tl.load(add_ptr + offs_n)
    acc = acc + bias[None, :] + add_val[None, :]

    # Swish: sigmoid(x) * x
    x_val = acc
    acc = tl.sigmoid(x_val) * x_val

    # Tanh: 2 * sigmoid(2*x) - 1
    acc = 2.0 * tl.sigmoid(2.0 * acc) - 1.0

    # GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
    x_val = acc
    acc = 0.5 * x_val * (1.0 + tl.erf(x_val * 0.7071067811865476))

    # Hardtanh: clamp to [-1, 1]
    acc = tl.maximum(acc, -1.0)
    acc = tl.minimum(acc, 1.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc)


_weights = None
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/95_Matmul_Add_Swish_Tanh_GELU_Hardtanh_weights.pt"

def run(x):
    global _weights
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _weights = {k: v.to(x.device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

    matmul_weight = _weights['matmul.weight']
    matmul_bias = _weights['matmul.bias']
    add_value = _weights['add_value']

    M, K = x.shape
    N = matmul_weight.shape[0]

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    fused_kernel[grid](
        x, matmul_weight, matmul_bias, add_value, out,
        M, N, K,
        x.stride(0), x.stride(1),
        matmul_weight.stride(0), matmul_weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_stages=3, num_warps=8,
    )

    return out
