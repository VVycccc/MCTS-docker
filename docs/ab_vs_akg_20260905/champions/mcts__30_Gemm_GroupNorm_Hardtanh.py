import torch
import triton
import triton.language as tl

_weights = None
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/30_Gemm_GroupNorm_Hardtanh_weights.pt"

def _init_weights(device):
    global _weights
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}


@triton.jit
def gemm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr):
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * K + k_offs[None, :]
        w_ptrs = w_ptr + offs_n[None, :] * K + k_offs[:, None]
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, w, acc=acc, allow_tf32=True)

    b = tl.load(b_ptr + offs_n)
    acc += b[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc)


@triton.jit
def gn_ht_kernel(x_ptr, gn_w_ptr, gn_b_ptr, out_ptr, M, N,
                 group_size: tl.constexpr, BLOCK: tl.constexpr,
                 HT_MIN: tl.constexpr, HT_MAX: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    start = pid_g * group_size
    offs = tl.arange(0, BLOCK)

    # Pass 1: mean
    mean = 0.0
    for i in range(0, group_size, BLOCK):
        idx = start + i + offs
        val = tl.load(x_ptr + pid_m * N + idx)
        mean += tl.sum(val)
    mean = mean / group_size

    # Pass 2: var
    var = 0.0
    for i in range(0, group_size, BLOCK):
        idx = start + i + offs
        val = tl.load(x_ptr + pid_m * N + idx)
        diff = val - mean
        var += tl.sum(diff * diff)
    var = var / group_size

    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # Pass 3: normalize + affine + hardtanh
    for i in range(0, group_size, BLOCK):
        idx = start + i + offs
        val = tl.load(x_ptr + pid_m * N + idx)
        normalized = (val - mean) * rstd
        w = tl.load(gn_w_ptr + idx)
        b = tl.load(gn_b_ptr + idx)
        out = normalized * w + b
        out = tl.maximum(out, HT_MIN)
        out = tl.minimum(out, HT_MAX)
        tl.store(out_ptr + pid_m * N + idx, out)


def run(x):
    global _weights
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)

    M, K = x.shape
    N = _weights['gemm.weight'].shape[0]
    num_groups = 16
    group_size = N // num_groups

    gemm_weight = _weights['gemm.weight']
    gemm_bias = _weights['gemm.bias']
    gn_weight = _weights['group_norm.weight']
    gn_bias = _weights['group_norm.bias']

    # GEMM
    gemm_out = torch.empty(M, N, device=x.device, dtype=torch.float32)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    gemm_kernel[grid](x, gemm_weight, gemm_bias, gemm_out, M, N, K,
                      BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                      GROUP_M=GROUP_M, num_stages=3, num_warps=8)

    # GroupNorm + HardTanh
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)
    BLOCK = 64
    grid_gn = (M, num_groups)
    gn_ht_kernel[grid_gn](gemm_out, gn_weight, gn_bias, out, M, N,
                          group_size=group_size, BLOCK=BLOCK,
                          HT_MIN=-2.0, HT_MAX=2.0)

    return out