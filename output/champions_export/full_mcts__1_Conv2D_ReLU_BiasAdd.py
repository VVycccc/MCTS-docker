import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/1_Conv2D_ReLU_BiasAdd_weights.pt"
_weights = None

def _init_weights(device):
    global _weights
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

@triton.jit
def conv_relu_bias_kernel(
    x_ptr, w_ptr, cb_ptr, bias_ptr, out_ptr,
    N, C_OUT, H, W, OH, OW,
    C_IN: tl.constexpr, K: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr, BLOCK_R: tl.constexpr,
):
    pid_n = tl.program_id(1)
    pid = tl.program_id(0)

    num_oc_blocks = tl.cdiv(C_OUT, BLOCK_OC)
    num_hw_blocks = tl.cdiv(OH * OW, BLOCK_HW)

    oc_block = pid // num_hw_blocks
    hw_block = pid % num_hw_blocks

    oc_start = oc_block * BLOCK_OC
    hw_start = hw_block * BLOCK_HW

    oc_offs = tl.arange(0, BLOCK_OC)
    hw_offs = tl.arange(0, BLOCK_HW)

    hw_idx = hw_start + hw_offs
    oh = hw_idx // OW
    ow = hw_idx % OW
    hw_mask = hw_idx < (OH * OW)
    oc_mask = (oc_start + oc_offs) < C_OUT

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    R = C_IN * K * K
    for r_start in range(0, R, BLOCK_R):
        r_offs = tl.arange(0, BLOCK_R)
        r_idx = r_start + r_offs

        ic = r_idx // (K * K)
        khk = r_idx % (K * K)
        kh = khk // K
        kw = khk % K

        r_mask = r_idx < R

        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]
        x_idx = pid_n * C_IN * H * W + ic[None, :] * H * W + ih * W + iw
        x_mask = hw_mask[:, None] & r_mask[None, :]
        x_patch = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

        w_idx = r_idx[:, None] * C_OUT + (oc_start + oc_offs)[None, :]
        w_mask = r_mask[:, None] & oc_mask[None, :]
        w_slice = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x_patch, w_slice, acc=acc, allow_tf32=True)

    cb = tl.load(cb_ptr + oc_start + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + cb[None, :]
    acc = tl.maximum(acc, 0.0)
    bias_val = tl.load(bias_ptr + oc_start + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias_val[None, :]

    out_idx = pid_n * C_OUT * OH * OW + (oc_start + oc_offs)[None, :] * (OH * OW) + oh[:, None] * OW + ow[:, None]
    out_mask = hw_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)

def run(x):
    global _weights
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)

    bias = _weights['bias']
    conv_bias = _weights['conv.bias']
    conv_weight = _weights['conv.weight']

    N, C_IN, H, W = x.shape
    C_OUT, _, K, _ = conv_weight.shape
    OH = H - K + 1
    OW = W - K + 1

    out = torch.empty(N, C_OUT, OH, OW, device=x.device, dtype=x.dtype)

    conv_weight = conv_weight.reshape(C_OUT, -1).t().contiguous()
    BLOCK_OC = 128
    BLOCK_HW = 128
    BLOCK_R = 32

    num_oc_blocks = triton.cdiv(C_OUT, BLOCK_OC)
    num_hw_blocks = triton.cdiv(OH * OW, BLOCK_HW)
    grid = (num_oc_blocks * num_hw_blocks, N)

    conv_relu_bias_kernel[grid](
        x, conv_weight, conv_bias, bias, out,
        N, C_OUT, H, W, OH, OW,
        C_IN=C_IN, K=K,
        BLOCK_OC=BLOCK_OC, BLOCK_HW=BLOCK_HW, BLOCK_R=BLOCK_R,
        num_stages=3, num_warps=8,
    )

    return out
