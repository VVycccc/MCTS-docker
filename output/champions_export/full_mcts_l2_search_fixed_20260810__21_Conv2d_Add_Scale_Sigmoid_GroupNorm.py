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
def conv_add_scale_sigmoid_kernel(x_ptr, w_ptr, conv_b_ptr, bias_ptr, scale_ptr, out_ptr,
                                  N: tl.constexpr, IC: tl.constexpr, OC: tl.constexpr,
                                  H: tl.constexpr, W: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
                                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    total_n = N * OH * OW
    mask_n = offs_n < total_n

    ow_idx = offs_n % OW
    tmp = offs_n // OW
    oh_idx = tmp % OH
    n_idx = tmp // OH

    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)

    K = IC * 9
    HW = H * W
    num_k = tl.cdiv(K, BLOCK_K)
    for k_iter in range(num_k):
        k_start = k_iter * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        ic_k = offs_k // 9
        rem_k = offs_k % 9
        kh_k = rem_k // 3
        kw_k = rem_k % 3

        w_ptrs = w_ptr + offs_m[:, None] * K + offs_k[None, :]
        w_tile = tl.load(w_ptrs, mask=mask_k[None, :], other=0.0)

        ih = oh_idx[None, :] + kh_k[:, None]
        iw = ow_idx[None, :] + kw_k[:, None]

        x_ptrs = x_ptr + n_idx[None, :] * IC * HW + ic_k[:, None] * HW + ih * W + iw
        x_tile = tl.load(x_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)

        acc = tl.dot(w_tile, x_tile, acc=acc, allow_tf32=True)

    conv_b = tl.load(conv_b_ptr + offs_m)
    acc = acc + conv_b[:, None]

    b_val = tl.load(bias_ptr + offs_m)
    s_val = tl.load(scale_ptr + offs_m)
    y = acc + b_val[:, None]
    y = y * s_val[:, None]
    y = tl.sigmoid(y)

    out_ptrs = out_ptr + n_idx[None, :] * OC * OH * OW + offs_m[:, None] * OH * OW + oh_idx[None, :] * OW + ow_idx[None, :]
    tl.store(out_ptrs, y, mask=mask_n[None, :])


@triton.jit
def gn_stats_kernel(x_ptr, mean_ptr, var_ptr,
                    NG: tl.constexpr, CPG: tl.constexpr, OC: tl.constexpr,
                    OH: tl.constexpr, OW: tl.constexpr,
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
                   total: tl.constexpr, spatial: tl.constexpr, NG: tl.constexpr,
                   CPG: tl.constexpr, OC: tl.constexpr, EPS: tl.constexpr,
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

    conv_weight = _weights['conv.weight'].reshape(OC, IC * 9).contiguous()
    conv_bias = _weights['conv.bias']
    bias = _weights['bias'].reshape(-1)
    scale = _weights['scale'].reshape(-1)
    gn_weight = _weights['group_norm.weight']
    gn_bias = _weights['group_norm.bias']

    total_conv = N * OC * OH * OW
    spatial = OH * OW

    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_K = 32
    total_n = N * OH * OW

    mid_out = torch.empty(N, OC, OH, OW, device=x.device, dtype=torch.float32)
    grid1 = (triton.cdiv(total_n, BLOCK_N),)
    conv_add_scale_sigmoid_kernel[grid1](x, conv_weight, conv_bias, bias, scale, mid_out,
                                         N=N, IC=IC, OC=OC, H=H, W=W, OH=OH, OW=OW,
                                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                         num_warps=8, num_stages=2)

    BLOCK = 4096
    mean = torch.empty(N, NG, device=x.device, dtype=torch.float32)
    var = torch.empty(N, NG, device=x.device, dtype=torch.float32)
    grid3 = (N * NG,)
    gn_stats_kernel[grid3](mid_out, mean, var,
                           NG=NG, CPG=CPG, OC=OC, OH=OH, OW=OW, BLOCK=BLOCK,
                           num_warps=8, num_stages=3)

    out = torch.empty_like(mid_out)
    grid4 = (triton.cdiv(total_conv, BLOCK),)
    gn_norm_kernel[grid4](mid_out, mean, var, gn_weight, gn_bias, out,
                          total=total_conv, spatial=spatial, NG=NG, CPG=CPG, OC=OC, EPS=EPS, BLOCK=BLOCK,
                          num_warps=8)

    return out
