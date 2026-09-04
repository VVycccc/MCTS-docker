import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/23_Conv3d_GroupNorm_Mean_weights.pt"
_weights = None

def _init_weights(device):
    global _weights
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_IN, C_OUT, D, H, W, OD, OH, OW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CIN: tl.constexpr,
    KK: tl.constexpr,
    KK_TOTAL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(C_OUT, BLOCK_M)
    num_pid_n = tl.cdiv(B * OD * OH * OW, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    oc_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    oc_mask = oc_offs < C_OUT

    pos_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    total_pos = B * OD * OH * OW
    pos_mask = pos_offs < total_pos

    ow = pos_offs % OW
    tmp = pos_offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    b = tmp // OD

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, KK_TOTAL, BLOCK_K):
        kk_offs = k_start + tl.arange(0, BLOCK_K)
        kk_mask = kk_offs < KK_TOTAL

        kw = kk_offs % KK
        tmp_k = kk_offs // KK
        kh = tmp_k % KK
        tmp_k = tmp_k // KK
        kd = tmp_k % KK
        ic = tmp_k // KK

        w_idx = oc_offs[:, None] * KK_TOTAL + kk_offs[None, :]
        w_val = tl.load(w_ptr + w_idx, mask=oc_mask[:, None] & kk_mask[None, :], other=0.0).to(tl.float32)

        in_d = od[None, :] + kd[:, None]
        in_h = oh[None, :] + kh[:, None]
        in_w = ow[None, :] + kw[:, None]
        x_idx = b[None, :] * (C_IN * D * H * W) + ic[:, None] * (D * H * W) + in_d * (H * W) + in_h * W + in_w
        x_val = tl.load(x_ptr + x_idx, mask=kk_mask[:, None] & pos_mask[None, :], other=0.0).to(tl.float32)

        acc = tl.dot(w_val, x_val, acc=acc, allow_tf32=True)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    acc += bias[:, None]

    out_idx = b[None, :] * (C_OUT * OD * OH * OW) + oc_offs[:, None] * (OD * OH * OW) + od[None, :] * (OH * OW) + oh[None, :] * OW + ow[None, :]
    tl.store(out_ptr + out_idx, acc, mask=oc_mask[:, None] & pos_mask[None, :])


@triton.jit
def gn_stats_kernel(
    x_ptr, mean_ptr, var_ptr,
    B, C_OUT, OD, OH, OW, NUM_GROUPS, CPG,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    total_elem = CPG * OD * OH * OW
    num_iters = tl.cdiv(total_elem, BLOCK_SIZE)

    sum_val = 0.0
    sum_sq = 0.0
    for i in range(num_iters):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elem

        ow = offs % OW
        tmp = offs // OW
        oh = tmp % OH
        tmp = tmp // OH
        od = tmp % OD
        c_local = tmp // OD

        c = g * CPG + c_local
        idx = b * C_OUT * OD * OH * OW + c * OD * OH * OW + od * OH * OW + oh * OW + ow
        x_val = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x_val, axis=0)
        sum_sq += tl.sum(x_val * x_val, axis=0)

    mean = sum_val / total_elem
    var = sum_sq / total_elem - mean * mean

    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, var)


@triton.jit
def fused_gn_mean_kernel(
    x_ptr, mean_ptr, var_ptr, gn_w_ptr, gn_b_ptr, out_ptr,
    B, C_OUT, OD, OH, OW, NUM_GROUPS, CPG, TOTAL_ELEM,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid

    sum_val = 0.0
    num_iters = tl.cdiv(TOTAL_ELEM, BLOCK_SIZE)

    for i in range(num_iters):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < TOTAL_ELEM

        ow = offs % OW
        tmp = offs // OW
        oh = tmp % OH
        tmp = tmp // OH
        od = tmp % OD
        c = tmp // OD

        g = c // CPG
        stats_idx = b * NUM_GROUPS + g

        mean = tl.load(mean_ptr + stats_idx, mask=mask, other=0.0)
        var = tl.load(var_ptr + stats_idx, mask=mask, other=0.0)
        gn_w = tl.load(gn_w_ptr + c, mask=mask, other=0.0).to(tl.float32)
        gn_b = tl.load(gn_b_ptr + c, mask=mask, other=0.0).to(tl.float32)

        x_idx = b * TOTAL_ELEM + offs
        x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0).to(tl.float32)

        normalized = (x_val - mean) / tl.sqrt(var + EPS)
        out = normalized * gn_w + gn_b
        sum_val += tl.sum(out, axis=0)

    result = sum_val / TOTAL_ELEM
    tl.store(out_ptr + b, result)


def run(x):
    global _weights
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)

    conv_weight = _weights['conv.weight']
    conv_bias = _weights['conv.bias']
    gn_weight = _weights['group_norm.weight']
    gn_bias = _weights['group_norm.bias']

    B = x.shape[0]
    C_IN = x.shape[1]
    D = x.shape[2]
    H = x.shape[3]
    W = x.shape[4]
    C_OUT = conv_weight.shape[0]
    K = conv_weight.shape[2]
    NUM_GROUPS = 8
    CPG = C_OUT // NUM_GROUPS

    OD = D - K + 1
    OH = H - K + 1
    OW = W - K + 1

    KK_TOTAL = C_IN * K * K * K

    conv_out = torch.empty(B, C_OUT, OD, OH, OW, device=x.device, dtype=torch.float32)

    BLOCK_M = 16
    BLOCK_N = 32
    BLOCK_K = 32

    total_pos = B * OD * OH * OW
    grid_conv = (triton.cdiv(C_OUT, BLOCK_M) * triton.cdiv(total_pos, BLOCK_N),)
    conv3d_kernel[grid_conv](
        x, conv_weight, conv_bias, conv_out,
        B, C_IN, C_OUT, D, H, W, OD, OH, OW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        CIN=C_IN, KK=K, KK_TOTAL=KK_TOTAL,
        num_warps=4, num_stages=3,
    )

    mean_buf = torch.empty(B, NUM_GROUPS, device=x.device, dtype=torch.float32)
    var_buf = torch.empty(B, NUM_GROUPS, device=x.device, dtype=torch.float32)
    grid_gn_stats = (B * NUM_GROUPS,)
    gn_stats_kernel[grid_gn_stats](
        conv_out, mean_buf, var_buf,
        B, C_OUT, OD, OH, OW, NUM_GROUPS, CPG,
        BLOCK_SIZE=1024,
    )

    result = torch.empty(B, device=x.device, dtype=torch.float32)
    TOTAL_ELEM = C_OUT * OD * OH * OW
    grid_fused = (B,)
    fused_gn_mean_kernel[grid_fused](
        conv_out, mean_buf, var_buf, gn_weight, gn_bias, result,
        B, C_OUT, OD, OH, OW, NUM_GROUPS, CPG, TOTAL_ELEM,
        EPS=1e-5, BLOCK_SIZE=1024,
    )

    return result
