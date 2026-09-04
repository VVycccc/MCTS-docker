import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/17_Conv2d_InstanceNorm_Divide_weights.pt"
_W = None

def _init_weights(device):
    global _W
    w = torch.load(_weights_path, map_location='cpu', weights_only=True)
    _W = {k: v.to(device) for k, v in w.items()}

@triton.jit
def conv2d_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                  N, IC, OC, H, W, OH, OW,
                  BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_noc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_oc_blocks = (OC + BLOCK_OC - 1) // BLOCK_OC
    n = pid_noc // num_oc_blocks
    oc_block = pid_noc % num_oc_blocks

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < (OH * OW)

    oh = offs_hw // OW
    ow = offs_hw % OW

    offs_oc = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    K = IC * 9
    k_idx = tl.arange(0, BLOCK_K)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + k_idx
        mask_k = k < K

        ic_k = k // 9
        kh_k = (k % 9) // 3
        kw_k = k % 3

        ih = oh[None, :] + kh_k[:, None]
        iw = ow[None, :] + kw_k[:, None]

        x_ptrs = x_ptr + n * IC * H * W + ic_k[:, None] * H * W + ih * W + iw
        x_tile = tl.load(x_ptrs, mask=mask_k[:, None] & mask_hw[None, :], other=0.0)

        w_ptrs = w_ptr + offs_oc[:, None] * IC * 9 + k[None, :]
        w_tile = tl.load(w_ptrs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)

        acc = tl.dot(w_tile, x_tile, acc=acc, allow_tf32=True)

    b_ptrs = b_ptr + offs_oc
    b_val = tl.load(b_ptrs, mask=mask_oc, other=0.0)
    acc += b_val[:, None]

    out_ptrs = out_ptr + n * OC * OH * OW + offs_oc[:, None] * OH * OW + offs_hw[None, :]
    tl.store(out_ptrs, acc, mask=mask_oc[:, None] & mask_hw[None, :])


@triton.jit
def fused_norm_divide_kernel(x_ptr, out_ptr,
                             N, OC, OH, OW, divide_by,
                             BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // OC
    c = pid % OC

    total = OH * OW
    base_idx = n * OC * OH * OW + c * OH * OW

    acc_sum = 0.0
    acc_sq = 0.0

    for off in range(0, total, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < total
        x_val = tl.load(x_ptr + base_idx + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x_val)
        acc_sq += tl.sum(x_val * x_val)

    mean = acc_sum / total
    var = acc_sq / total - mean * mean
    eps = 1e-5
    rstd = 1.0 / tl.sqrt(var + eps)
    inv_div = 1.0 / divide_by

    for off in range(0, total, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < total
        x_val = tl.load(x_ptr + base_idx + offs, mask=mask, other=0.0)
        y_val = (x_val - mean) * rstd * inv_div
        tl.store(out_ptr + base_idx + offs, y_val, mask=mask)


def run(x):
    global _W
    if _W is None or str(next(iter(_W.values())).device) != str(x.device):
        _init_weights(x.device)

    conv_weight = _W['conv.weight']
    conv_bias = _W['conv.bias']

    N, IC, H, W = x.shape
    OC = conv_weight.shape[0]
    KH = conv_weight.shape[2]
    OH = H - KH + 1
    OW = W - KH + 1
    divide_by = 2.0

    conv_out = torch.empty(N, OC, OH, OW, device=x.device, dtype=torch.float32)

    BLOCK_OC = 16
    BLOCK_HW = 256
    BLOCK_K = 16
    grid = (N * triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_HW))
    conv2d_kernel[grid](
        x, conv_weight, conv_bias, conv_out,
        N, IC, OC, H, W, OH, OW,
        BLOCK_OC=BLOCK_OC, BLOCK_HW=BLOCK_HW, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2
    )

    out = torch.empty(N, OC, OH, OW, device=x.device, dtype=torch.float32)
    BLOCK_SIZE = 256
    grid_nd = (N * OC,)
    fused_norm_divide_kernel[grid_nd](
        conv_out, out,
        N, OC, OH, OW, divide_by,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )

    return out
