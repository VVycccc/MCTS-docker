import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/1_Conv2D_ReLU_BiasAdd_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['N', 'IC', 'OC', 'OH', 'OW'],
)
@triton.jit
def conv_relu_bias_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, IC, OC, OH, OW, IH, IW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    spatial_size = OH * OW
    N_total = N * spatial_size
    num_pid_n = tl.cdiv(N_total, BLOCK_N)
    pid_n = pid % num_pid_n
    pid_m = pid // num_pid_n

    oc_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    oc_mask = oc_offs < OC

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N_total

    batch_idx = n_offs // spatial_size
    spatial_idx = n_offs % spatial_size
    oh_idx = spatial_idx // OW
    ow_idx = spatial_idx % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K = IC * KH * KW
    for k_blk in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k_blk * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        ic_idx = k_offs // (KH * KW)
        kh_kw_idx = k_offs % (KH * KW)
        kh_idx = kh_kw_idx // KW
        kw_idx = kh_kw_idx % KW

        w_offs = oc_offs[:, None] * K + k_offs[None, :]
        w_val = tl.load(w_ptr + w_offs, mask=oc_mask[:, None] & k_mask[None, :], other=0.0)

        ih_idx = oh_idx[None, :] + kh_idx[:, None]
        iw_idx = ow_idx[None, :] + kw_idx[:, None]
        x_offs = batch_idx[None, :] * IC * IH * IW + ic_idx[:, None] * IH * IW + ih_idx * IW + iw_idx
        x_val = tl.load(x_ptr + x_offs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc = tl.dot(w_val, x_val, acc=acc, allow_tf32=True)

    cb_val = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += cb_val[:, None]
    acc = tl.maximum(acc, 0.0)
    acc += b_val[:, None]

    out_offs = batch_idx[None, :] * OC * spatial_size + oc_offs[:, None] * spatial_size + spatial_idx[None, :]
    tl.store(out_ptr + out_offs, acc, mask=oc_mask[:, None] & n_mask[None, :])

def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    bias = _weights['bias']
    conv_bias = _weights['conv.bias']
    conv_weight = _weights['conv.weight']

    N, IC, IH, IW = x.shape
    OC, _, KH, KW = conv_weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty(N, OC, OH, OW, device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(OC, meta['BLOCK_M']) * triton.cdiv(N * OH * OW, meta['BLOCK_N']),)

    conv_relu_bias_kernel[grid](
        x, conv_weight, conv_bias, bias, out,
        N, IC, OC, OH, OW, IH, IW,
        KH=KH, KW=KW,
    )

    return out
