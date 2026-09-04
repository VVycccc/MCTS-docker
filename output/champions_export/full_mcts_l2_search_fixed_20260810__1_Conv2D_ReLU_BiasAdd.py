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

@triton.jit
def conv_relu_bias_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, IC, OC, OH, OW, IH, IW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_hw_tiles = tl.cdiv(OH * OW, BLOCK_HW)
    num_oc_tiles = tl.cdiv(OC, BLOCK_OC)

    pid_hw = pid % num_hw_tiles
    tmp = pid // num_hw_tiles
    pid_oc = tmp % num_oc_tiles
    pid_n = tmp // num_oc_tiles

    # Output channel indices
    oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = oc < OC

    # Flattened spatial indices
    hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = hw < (OH * OW)
    oh = hw // OW
    ow = hw % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # im2col + GEMM: loop over IC*KH*KW in chunks of BLOCK_K
    K = IC * KH * KW
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_idx < K

        # Decode linear k -> (ic, kh, kw)
        ic = k_idx // (KH * KW)
        kh_kw = k_idx % (KH * KW)
        kh = kh_kw // KW
        kw = kh_kw % KW

        # Input block: (BLOCK_K, BLOCK_HW) via im2col gather
        ih = oh[None, :] + kh[:, None]
        iw = ow[None, :] + kw[:, None]
        x_offs = pid_n * IC * IH * IW + ic[:, None] * IH * IW + ih * IW + iw
        x_val = tl.load(x_ptr + x_offs, mask=mask_k[:, None] & mask_hw[None, :], other=0.0)

        # Weight block: (BLOCK_OC, BLOCK_K)
        w_offs = oc[:, None] * K + ic[None, :] * KH * KW + kh[None, :] * KW + kw[None, :]
        w_val = tl.load(w_ptr + w_offs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)

        acc = tl.dot(w_val, x_val, acc=acc, allow_tf32=True)

    # Conv bias -> ReLU -> channel bias
    cb_val = tl.load(cb_ptr + oc, mask=mask_oc, other=0.0)
    b_val = tl.load(b_ptr + oc, mask=mask_oc, other=0.0)
    acc += cb_val[:, None]
    acc = tl.maximum(acc, 0.0)
    acc += b_val[:, None]

    # Store output
    out_offs = pid_n * OC * OH * OW + oc[:, None] * OH * OW + oh[None, :] * OW + ow[None, :]
    tl.store(out_ptr + out_offs, acc, mask=mask_oc[:, None] & mask_hw[None, :])

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

    BLOCK_OC = 64
    BLOCK_HW = 128
    BLOCK_K = 32
    num_oc_tiles = triton.cdiv(OC, BLOCK_OC)
    num_hw_tiles = triton.cdiv(OH * OW, BLOCK_HW)
    grid = (N * num_oc_tiles * num_hw_tiles,)

    conv_relu_bias_kernel[grid](
        x, conv_weight, conv_bias, bias, out,
        N, IC, OC, OH, OW, IH, IW,
        KH=KH, KW=KW,
        BLOCK_OC=BLOCK_OC, BLOCK_HW=BLOCK_HW, BLOCK_K=BLOCK_K,
        num_stages=3, num_warps=8,
    )

    return out
