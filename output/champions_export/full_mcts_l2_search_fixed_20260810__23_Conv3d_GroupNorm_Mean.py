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
def conv_stats_kernel(
    x_ptr, w_ptr, b_ptr,
    S_ptr, Q_ptr,
    B, C_IN, C_OUT, D, H, W, OD, OH, OW,
    SPATIAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CIN: tl.constexpr,
    KK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(SPATIAL, BLOCK_SIZE)
    bc = pid // num_tiles
    tile_id = pid % num_tiles

    c = bc % C_OUT
    b = bc // C_OUT

    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < SPATIAL

    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    od = tmp // OH

    bias_val = tl.load(b_ptr + c).to(tl.float32)
    acc = tl.where(mask, bias_val, 0.0)

    for ic in range(CIN):
        for kd in range(KK):
            for kh in range(KK):
                for kw in range(KK):
                    in_d = od + kd
                    in_h = oh + kh
                    in_w = ow + kw
                    x_idx = b * C_IN * D * H * W + ic * D * H * W + in_d * H * W + in_h * W + in_w
                    w_idx = c * CIN * KK * KK * KK + ic * KK * KK * KK + kd * KK * KK + kh * KK + kw
                    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0).to(tl.float32)
                    w_val = tl.load(w_ptr + w_idx).to(tl.float32)
                    acc += x_val * w_val

    partial_sum = tl.sum(acc, axis=0)
    partial_sq = tl.sum(acc * acc, axis=0)
    tl.atomic_add(S_ptr + b * C_OUT + c, partial_sum)
    tl.atomic_add(Q_ptr + b * C_OUT + c, partial_sq)


@triton.jit
def final_kernel(
    S_ptr, Q_ptr, gn_w_ptr, gn_b_ptr, out_ptr,
    C_OUT, NUM_GROUPS, CPG,
    S_VOL: tl.constexpr,
    N_TOTAL: tl.constexpr,
    N_G: tl.constexpr,
    EPS: tl.constexpr,
    COUT: tl.constexpr,
    NUMGROUPS: tl.constexpr,
    CPGC: tl.constexpr,
):
    b = tl.program_id(0)

    result = 0.0
    for g in range(NUMGROUPS):
        sum_s = 0.0
        sum_q = 0.0
        for ci in range(CPGC):
            c = g * CPGC + ci
            s_val = tl.load(S_ptr + b * C_OUT + c)
            q_val = tl.load(Q_ptr + b * C_OUT + c)
            sum_s += s_val
            sum_q += q_val
        mu_g = sum_s / N_G
        var_g = sum_q / N_G - mu_g * mu_g
        inv_std = 1.0 / tl.sqrt(var_g + EPS)

        group_contrib = 0.0
        for ci in range(CPGC):
            c = g * CPGC + ci
            s_val = tl.load(S_ptr + b * C_OUT + c)
            gn_w = tl.load(gn_w_ptr + c).to(tl.float32)
            group_contrib += gn_w * (s_val - S_VOL * mu_g)
        result += inv_std * group_contrib

    bias_sum = 0.0
    for c in range(COUT):
        gn_b = tl.load(gn_b_ptr + c).to(tl.float32)
        bias_sum += gn_b
    result += bias_sum * S_VOL

    result = result / N_TOTAL
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
    SPATIAL = OD * OH * OW
    S_VOL = SPATIAL
    N_TOTAL = C_OUT * S_VOL
    N_G = CPG * S_VOL

    BLOCK = 512

    S_buf = torch.zeros(B, C_OUT, device=x.device, dtype=torch.float32)
    Q_buf = torch.zeros(B, C_OUT, device=x.device, dtype=torch.float32)
    num_tiles = triton.cdiv(SPATIAL, BLOCK)
    grid_stats = (B * C_OUT * num_tiles,)
    conv_stats_kernel[grid_stats](
        x, conv_weight, conv_bias,
        S_buf, Q_buf,
        B, C_IN, C_OUT, D, H, W, OD, OH, OW,
        SPATIAL=SPATIAL, BLOCK_SIZE=BLOCK, CIN=C_IN, KK=K,
    )

    result = torch.empty(B, device=x.device, dtype=torch.float32)
    final_kernel[(B,)](
        S_buf, Q_buf, gn_weight, gn_bias, result,
        C_OUT, NUM_GROUPS, CPG,
        S_VOL=S_VOL, N_TOTAL=N_TOTAL, N_G=N_G,
        EPS=1e-5, COUT=C_OUT, NUMGROUPS=NUM_GROUPS, CPGC=CPG,
    )

    return result
