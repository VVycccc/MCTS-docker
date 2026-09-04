import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/21_Conv2d_Add_Scale_Sigmoid_GroupNorm_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.jit
def conv2d_kernel(x_ptr, w_ptr, cb_ptr, bias_ptr, scale_ptr, out_ptr,
                  N, IC, OC, H, W, OH, OW,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_spatial = tl.program_id(1)

    num_pid_m = tl.cdiv(OC, BLOCK_M)
    pid_n = pid // num_pid_m
    pid_m = pid % num_pid_m

    # Output channel tile
    oc_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    oc_mask = oc_offs < OC

    # Spatial tile
    spatial_offs = pid_spatial * BLOCK_N + tl.arange(0, BLOCK_N)
    spatial_mask = spatial_offs < (OH * OW)

    # Decode spatial offsets to (oh, ow)
    oh_idx_s = spatial_offs // OW
    ow_idx_s = spatial_offs % OW

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension (IC * 9)
    IC9 = IC * 9
    for k_start in range(0, IC9, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < IC9

        # Decode k_offs into (ic, kh, kw)
        ic_idx = k_offs // 9
        khkw = k_offs % 9
        kh_idx = khkw // 3
        kw_idx = khkw % 3

        # Load weight tile: (BLOCK_M, BLOCK_K)
        w_offs = oc_offs[:, None] * IC9 + k_offs[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=oc_mask[:, None] & k_mask[None, :], other=0.0)

        # Compute input indices: (BLOCK_K, BLOCK_N)
        ih_idx = kh_idx[:, None] + oh_idx_s[None, :]
        iw_idx = kw_idx[:, None] + ow_idx_s[None, :]

        x_offs = pid_n * IC * H * W + ic_idx[:, None] * H * W + ih_idx * W + iw_idx
        x_tile = tl.load(x_ptr + x_offs, mask=k_mask[:, None] & spatial_mask[None, :], other=0.0)

        # Matrix multiply
        acc = tl.dot(w_tile, x_tile, acc=acc, allow_tf32=True)

    # Add conv bias
    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += cb[:, None]

    # Fused epilogue: add bias, scale, sigmoid
    b_val = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    s_val = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_val[:, None]
    acc = acc * s_val[:, None]
    acc = 1.0 / (1.0 + tl.exp(-acc))

    # Store output
    out_offs = pid_n * OC * OH * OW + oc_offs[:, None] * OH * OW + spatial_offs[None, :]
    tl.store(out_ptr + out_offs, acc, mask=oc_mask[:, None] & spatial_mask[None, :])


@triton.jit
def add_scale_sigmoid_kernel(x_ptr, bias_ptr, scale_ptr, out_ptr,
                             total, spatial, OC,
                             BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    oc_idx = (offs // spatial) % OC

    x_val = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b_val = tl.load(bias_ptr + oc_idx, mask=mask, other=0.0)
    s_val = tl.load(scale_ptr + oc_idx, mask=mask, other=0.0)

    y = x_val + b_val
    y = y * s_val
    y = 1.0 / (1.0 + tl.exp(-y))

    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def gn_stats_kernel(x_ptr, mean_ptr, var_ptr,
                    NG, CPG, OC, OH, OW,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    n_idx = pid // NG
    g_idx = pid % NG

    total_elem = CPG * OH * OW
    spatial = OH * OW
    num_iters = tl.cdiv(total_elem, BLOCK)

    sum_val = 0.0
    sum_sq = 0.0
    for i in range(num_iters):
        off = i * BLOCK
        idx = off + tl.arange(0, BLOCK)
        mask = idx < total_elem

        ow_idx = idx % OW
        tmp = idx // OW
        oh_idx = tmp % OH
        c_local = tmp // OH

        c_idx = g_idx * CPG + c_local
        full_idx = n_idx * OC * spatial + c_idx * spatial + oh_idx * OW + ow_idx

        x_val = tl.load(x_ptr + full_idx, mask=mask, other=0.0)
        sum_val += tl.sum(x_val, axis=0)
        sum_sq += tl.sum(x_val * x_val, axis=0)

    mean = sum_val / total_elem
    var = sum_sq / total_elem - mean * mean

    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, var)


@triton.jit
def gn_norm_kernel(x_ptr, mean_ptr, var_ptr, w_ptr, b_ptr, out_ptr,
                   total, spatial, NG, CPG, OC, EPS,
                   BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    oc_idx = (offs // spatial) % OC
    n_idx = offs // (OC * spatial)
    g_idx = oc_idx // CPG

    x_val = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + n_idx * NG + g_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + n_idx * NG + g_idx, mask=mask, other=0.0)
    w_val = tl.load(w_ptr + oc_idx, mask=mask, other=0.0)
    b_val = tl.load(b_ptr + oc_idx, mask=mask, other=0.0)

    y = (x_val - mean) / tl.sqrt(var + EPS)
    y = y * w_val + b_val

    tl.store(out_ptr + offs, y, mask=mask)


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    x = x.contiguous()
    N, IC, H, W = x.shape
    OC = 32
    K = 3
    OH = H - K + 1
    OW = W - K + 1
    NG = 8
    CPG = OC // NG
    EPS = 1e-5

    conv_weight = _weights['conv.weight']
    conv_bias = _weights['conv.bias']
    bias = _weights['bias'].reshape(-1)
    scale = _weights['scale'].reshape(-1)
    gn_weight = _weights['group_norm.weight']
    gn_bias = _weights['group_norm.bias']

    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_K = 32
    total_conv = N * OC * OH * OW
    spatial = OH * OW

    mid_out = torch.empty(N, OC, OH, OW, device=x.device, dtype=torch.float32)
    grid1 = (N * triton.cdiv(OC, BLOCK_M), triton.cdiv(spatial, BLOCK_N))
    conv2d_kernel[grid1](x, conv_weight, conv_bias, bias, scale, mid_out,
                         N, IC, OC, H, W, OH, OW,
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                         num_stages=3, num_warps=4)

    BLOCK = 256
    mean = torch.empty(N, NG, device=x.device, dtype=torch.float32)
    var = torch.empty(N, NG, device=x.device, dtype=torch.float32)
    grid3 = (N * NG,)
    gn_stats_kernel[grid3](mid_out, mean, var,
                           NG, CPG, OC, OH, OW, BLOCK=BLOCK)

    out = torch.empty_like(mid_out)
    grid4 = (triton.cdiv(total_conv, BLOCK),)
    gn_norm_kernel[grid4](mid_out, mean, var, gn_weight, gn_bias, out,
                          total_conv, spatial, NG, CPG, OC, EPS, BLOCK=BLOCK)

    return out
