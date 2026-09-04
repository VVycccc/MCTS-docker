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
def conv2d_kernel(x_ptr, w_ptr, b_ptr, out_ptr, sum_ptr, sumsq_ptr,
                  N, IC, OC, H, W, OH, OW,
                  BLOCK_HW: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    n = pid_n
    oc_start = pid_oc * BLOCK_OC
    hw_start = pid_hw * BLOCK_HW

    offs_hw = hw_start + tl.arange(0, BLOCK_HW)
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)

    ow = offs_hw % OW
    oh = offs_hw // OW
    mask_hw = offs_hw < OH * OW
    mask_oc = offs_oc < OC

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    K = IC * 9
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        kw = offs_k % 3
        kh = (offs_k // 3) % 3
        ic = offs_k // 9

        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]
        x_ptrs = x_ptr + n * IC * H * W + ic[None, :] * H * W + ih * W + iw
        a = tl.load(x_ptrs, mask=mask_hw[:, None], other=0.0)

        w_ptrs = w_ptr + offs_oc[None, :] * (IC * 9) + offs_k[:, None]
        b = tl.load(w_ptrs, mask=mask_oc[None, :], other=0.0)

        acc = tl.dot(a, b, acc=acc, allow_tf32=True)

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += bias[None, :]

    # Fuse InstanceNorm: accumulate per-channel sum and sum_sq via atomic_add
    acc_masked = tl.where(mask_hw[:, None], acc, 0.0)
    partial_sum = tl.sum(acc_masked, axis=0)
    partial_sumsq = tl.sum(acc_masked * acc_masked, axis=0)
    tl.atomic_add(sum_ptr + n * OC + offs_oc, partial_sum, mask=mask_oc)
    tl.atomic_add(sumsq_ptr + n * OC + offs_oc, partial_sumsq, mask=mask_oc)

    out_ptrs = out_ptr + n * OC * OH * OW + offs_oc[None, :] * OH * OW + offs_hw[:, None]
    tl.store(out_ptrs, acc, mask=mask_hw[:, None] & mask_oc[None, :])


@triton.jit
def fused_norm_divide_kernel(x_ptr, out_ptr, sum_ptr, sumsq_ptr,
                             N, OC, OH, OW, divide_by,
                             BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // OC
    c = pid % OC

    total = OH * OW
    base = n * OC * OH * OW + c * OH * OW

    # Read pre-computed sum and sum_sq from conv kernel epilogue
    acc_sum = tl.load(sum_ptr + n * OC + c)
    acc_sq = tl.load(sumsq_ptr + n * OC + c)

    mean = acc_sum / total
    var = acc_sq / total - mean * mean
    eps = 1e-5
    rstd = 1.0 / tl.sqrt(var + eps)
    inv_div = rstd / divide_by

    # Single pass: normalize and divide
    for off in range(0, total, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < total
        x_val = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y_val = (x_val - mean) * inv_div
        tl.store(out_ptr + base + offs, y_val, mask=mask)


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
    sum_buf = torch.zeros(N, OC, device=x.device, dtype=torch.float32)
    sumsq_buf = torch.zeros(N, OC, device=x.device, dtype=torch.float32)

    BLOCK_HW = 128
    BLOCK_OC = 128
    BLOCK_K = 32
    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_HW))
    conv2d_kernel[grid](
        x, conv_weight, conv_bias, conv_out, sum_buf, sumsq_buf,
        N, IC, OC, H, W, OH, OW,
        BLOCK_HW=BLOCK_HW, BLOCK_OC=BLOCK_OC, BLOCK_K=BLOCK_K,
        num_stages=3, num_warps=8,
    )

    out = torch.empty(N, OC, OH, OW, device=x.device, dtype=torch.float32)
    BLOCK_SIZE = 2048
    grid_nd = (N * OC,)
    fused_norm_divide_kernel[grid_nd](
        conv_out, out, sum_buf, sumsq_buf,
        N, OC, OH, OW, divide_by,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    return out
