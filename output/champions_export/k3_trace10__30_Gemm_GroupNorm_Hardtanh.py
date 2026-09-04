import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/30_Gemm_GroupNorm_Hardtanh_weights.pt"
_W = None


def _get_weights(device):
    global _W
    if _W is None or str(next(iter(_W.values())).device) != str(device):
        _W = {k: v.to(device).contiguous() for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    return _W


@triton.jit
def gemm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, sum_ptr, sq_ptr,
    M, N, K, G, D,
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

    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(x_ptrs)
        b = tl.load(w_ptrs)
        acc = tl.dot(a, tl.trans(b), acc, allow_tf32=True)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    bias = tl.load(b_ptr + offs_n)
    acc = acc + bias[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc)

    # GroupNorm 融合：累加本 tile 所属 (row, group) 的 sum / sumsq
    # (BLOCK_N 整除 D，故一个 tile 完整落在单个 group 内)
    g = (pid_n * BLOCK_N) // D
    part_sum = tl.sum(acc, axis=1)
    part_sq = tl.sum(acc * acc, axis=1)
    tl.atomic_add(sum_ptr + offs_m * G + g, part_sum)
    tl.atomic_add(sq_ptr + offs_m * G + g, part_sq)


@triton.jit
def norm_hardtanh_kernel(
    x_ptr, sum_ptr, sq_ptr, w_ptr, b_ptr, y_ptr,
    C, G, eps, min_val, max_val,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # 每个 program 处理一个 (row, group)：统计量已由 GEMM epilogue 累加好
    pid = tl.program_id(0)
    row = pid // G
    g = pid % G

    s = tl.load(sum_ptr + row * G + g)
    sq = tl.load(sq_ptr + row * G + g)
    mean = s / D
    var = sq / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    base = row * C + g * D
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + g * D + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + g * D + offs, mask=mask, other=0.0)
    y = (x - mean) * inv_std * w + b
    y = tl.minimum(tl.maximum(y, min_val), max_val)
    tl.store(y_ptr + base + offs, y, mask=mask)


def run(x):
    weights = _get_weights(x.device)
    gemm_weight = weights['gemm.weight']      # [out_features, in_features]
    gemm_bias = weights['gemm.bias']          # [out_features]
    gn_weight = weights['group_norm.weight']  # [out_features]
    gn_bias = weights['group_norm.bias']      # [out_features]

    num_groups = 16
    hardtanh_min = -2.0
    hardtanh_max = 2.0
    eps = 1e-5

    x = x.contiguous()
    M, K = x.shape
    N = gemm_weight.shape[0]
    D = N // num_groups

    # 1) 融合 GEMM: y = x @ W^T + b，并原子累加每 (row, group) 的 sum/sumsq
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    gsum = torch.zeros((M, num_groups), device=x.device, dtype=torch.float32)
    gsq = torch.zeros((M, num_groups), device=x.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 32, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    gemm_kernel[grid](x, gemm_weight, gemm_bias, y, gsum, gsq, M, N, K,
                      num_groups, D,
                      BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
                      num_warps=8, num_stages=3)

    # 2) 融合 GroupNorm + HardTanh（统计量已由 GEMM 产出，单遍完成）
    out = torch.empty_like(y)
    norm_hardtanh_kernel[(M * num_groups,)](
        y, gsum, gsq, gn_weight, gn_bias, out, N, num_groups, eps,
        hardtanh_min, hardtanh_max,
        D=D, BLOCK_D=triton.next_power_of_2(D),
    )
    return out
