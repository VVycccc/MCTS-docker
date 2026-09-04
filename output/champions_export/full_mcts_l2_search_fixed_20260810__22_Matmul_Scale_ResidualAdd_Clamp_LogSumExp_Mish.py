import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.jit
def fused_gemm_lse_kernel(x_ptr, w_ptr, b_ptr, pmax_ptr, psum_ptr,
                          num_n_tiles, scale_factor, clamp_min, clamp_max,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K // BLOCK_K):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, w, acc=acc, allow_tf32=True)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    b = tl.load(b_ptr + offs_n)
    acc += b[None, :]

    acc = acc * scale_factor
    acc = acc + acc
    acc = tl.minimum(tl.maximum(acc, clamp_min), clamp_max)

    local_max = tl.max(acc, axis=1)
    local_sum = tl.sum(tl.exp(acc - local_max[:, None]), axis=1)

    pmax_ptrs = pmax_ptr + offs_m * num_n_tiles + pid_n
    psum_ptrs = psum_ptr + offs_m * num_n_tiles + pid_n
    tl.store(pmax_ptrs, local_max)
    tl.store(psum_ptrs, local_sum)


@triton.jit
def lse_reduce_mish_kernel(pmax_ptr, psum_ptr, out_ptr, num_n_tiles,
                           BLOCK_M: tl.constexpr, M: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    running_max = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for pid_n in range(0, num_n_tiles):
        pmax_ptrs = pmax_ptr + offs_m * num_n_tiles + pid_n
        psum_ptrs = psum_ptr + offs_m * num_n_tiles + pid_n
        local_max = tl.load(pmax_ptrs)
        local_sum = tl.load(psum_ptrs)

        new_max = tl.maximum(running_max, local_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + local_sum * tl.exp(local_max - new_max)
        running_max = new_max

    lse = tl.log(running_sum) + running_max

    sp = tl.where(lse > 20.0, lse, tl.log(1.0 + tl.exp(lse)))
    tanh_sp = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    mish_val = lse * tanh_sp
    result = lse * mish_val

    tl.store(out_ptr + offs_m, result)


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    matmul_weight = _weights['matmul.weight']
    matmul_bias = _weights['matmul.bias']

    x = x.contiguous()
    M, K = x.shape
    N = matmul_weight.shape[0]

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    num_n_tiles = triton.cdiv(N, BLOCK_N)

    pmax = torch.empty(M, num_n_tiles, device=x.device, dtype=torch.float32)
    psum = torch.empty(M, num_n_tiles, device=x.device, dtype=torch.float32)
    final_out = torch.empty(M, 1, device=x.device, dtype=torch.float32)

    grid1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    fused_gemm_lse_kernel[grid1](x, matmul_weight, matmul_bias, pmax, psum,
                                 num_n_tiles, 2.0, -10.0, 10.0,
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                 M=M, N=N, K=K,
                                 num_warps=8, num_stages=3)

    grid2 = (triton.cdiv(M, BLOCK_M),)
    lse_reduce_mish_kernel[grid2](pmax, psum, final_out.view(-1), num_n_tiles,
                                 BLOCK_M=BLOCK_M, M=M)

    return final_out
