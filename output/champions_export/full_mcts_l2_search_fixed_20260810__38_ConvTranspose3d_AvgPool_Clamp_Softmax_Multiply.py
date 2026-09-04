import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.jit
def avg_pool3d_kernel(x_ptr, out_ptr, B, C, D, H, W, OD, OH, OW, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_w = tl.cdiv(OW, BLOCK)
    pid_w = pid % num_w
    pid_h = (pid // num_w) % OH
    pid_d = (pid // (num_w * OH)) % OD
    pid_c = (pid // (num_w * OH * OD)) % C
    pid_b = pid // (num_w * OH * OD * C)

    offs = pid_w * BLOCK + tl.arange(0, BLOCK)
    mask = offs < OW

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for kd in range(2):
        for kh in range(2):
            for kw in range(2):
                in_d = pid_d * 2 + kd
                in_h = pid_h * 2 + kh
                in_w = offs * 2 + kw
                idx = (((pid_b * C + pid_c) * D + in_d) * H + in_h) * W + in_w
                val = tl.load(x_ptr + idx, mask=mask, other=0.0)
                acc += val
    acc = acc * 0.125
    out_idx = (((pid_b * C + pid_c) * OD + pid_d) * OH + pid_h) * OW + offs
    tl.store(out_ptr + out_idx, acc, mask=mask)


@triton.jit
def conv_transpose3d_clamp_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                                   B, C_in, C_out, D_in, H_in, W_in,
                                   D_out, H_out, W_out,
                                   BLOCK_W: tl.constexpr, BLOCK_OC: tl.constexpr):
    pid = tl.program_id(0)
    num_w = tl.cdiv(W_out, BLOCK_W)
    num_oc = tl.cdiv(C_out, BLOCK_OC)
    pid_w = pid % num_w
    pid_h = (pid // num_w) % H_out
    pid_d = (pid // (num_w * H_out)) % D_out
    pid_oc = (pid // (num_w * H_out * D_out)) % num_oc
    pid_b = pid // (num_w * H_out * D_out * num_oc)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_out

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < C_out

    acc = tl.zeros([BLOCK_OC, BLOCK_W], dtype=tl.float32)

    for ic in range(C_in):
        for kd in range(3):
            for kh in range(3):
                for kw in range(3):
                    id_num = pid_d + 1 - kd
                    ih_num = pid_h + 1 - kh
                    iw_num = offs_w + 1 - kw

                    id_ok = (id_num >= 0) & (id_num < D_in * 2) & ((id_num % 2) == 0)
                    ih_ok = (ih_num >= 0) & (ih_num < H_in * 2) & ((ih_num % 2) == 0)
                    iw_ok = (iw_num >= 0) & (iw_num < W_in * 2) & ((iw_num % 2) == 0)

                    valid = id_ok & ih_ok & iw_ok & mask_w

                    id_val = id_num // 2
                    ih_val = ih_num // 2
                    iw_val = iw_num // 2

                    x_idx = (((pid_b * C_in + ic) * D_in + id_val) * H_in + ih_val) * W_in + iw_val
                    x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)

                    w_idx = (((ic * C_out + offs_oc) * 3 + kd) * 3 + kh) * 3 + kw
                    w_val = tl.load(w_ptr + w_idx, mask=mask_oc, other=0.0)

                    acc += w_val[:, None] * x_val[None, :]

    b_val = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += b_val[:, None]

    # Fused clamp [0, 1]
    acc = tl.maximum(tl.minimum(acc, 1.0), 0.0)

    out_oc_base = (((pid_b * C_out + offs_oc) * D_out + pid_d) * H_out + pid_h) * W_out
    out_idx = out_oc_base[:, None] + offs_w[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask_oc[:, None] & mask_w[None, :])


@triton.jit
def softmax_multiply_kernel(x_ptr, scale_ptr, out_ptr, B, C, spatial, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid_c = pid % C
    pid_b = pid // C

    base = (pid_b * C + pid_c) * spatial

    max_val = -1.0e38
    for off in range(0, spatial, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < spatial
        val = tl.load(x_ptr + base + offs, mask=mask, other=-1.0e38)
        block_max = tl.max(val, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = 0.0
    for off in range(0, spatial, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < spatial
        val = tl.load(x_ptr + base + offs, mask=mask, other=-1.0e38)
        sum_exp += tl.sum(tl.exp(val - max_val), axis=0)

    scale_val = tl.load(scale_ptr + pid_c)

    for off in range(0, spatial, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < spatial
        val = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        result = tl.exp(val - max_val) / sum_exp * scale_val
        tl.store(out_ptr + base + offs, result, mask=mask)


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    device = x.device
    B, C_in, D, H, W = x.shape

    BLOCK = 64

    # Step 1: avg_pool3d (kernel_size=2)
    OD_p, OH_p, OW_p = D // 2, H // 2, W // 2
    pooled = torch.empty(B, C_in, OD_p, OH_p, OW_p, device=device, dtype=torch.float32)
    grid = (B * C_in * OD_p * OH_p * triton.cdiv(OW_p, BLOCK),)
    avg_pool3d_kernel[grid](x, pooled, B, C_in, D, H, W, OD_p, OH_p, OW_p, BLOCK=BLOCK)

    # Step 2: conv_transpose3d (k=3, s=2, p=1, op=1) + fused clamp [0, 1]
    conv_w = _weights['conv_transpose.weight']
    conv_b = _weights['conv_transpose.bias']
    C_out = conv_w.shape[1]
    D_in, H_in, W_in = OD_p, OH_p, OW_p
    D_out = (D_in - 1) * 2 - 2 * 1 + 3 + 1
    H_out = (H_in - 1) * 2 - 2 * 1 + 3 + 1
    W_out = (W_in - 1) * 2 - 2 * 1 + 3 + 1

    BLOCK_W = 32
    BLOCK_OC = 16

    clamped = torch.empty(B, C_out, D_out, H_out, W_out, device=device, dtype=torch.float32)
    grid = (B * triton.cdiv(C_out, BLOCK_OC) * D_out * H_out * triton.cdiv(W_out, BLOCK_W),)
    conv_transpose3d_clamp_kernel[grid](
        pooled, conv_w, conv_b, clamped,
        B, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        BLOCK_W=BLOCK_W, BLOCK_OC=BLOCK_OC
    )

    # Step 3: softmax over flattened spatial dims + fused multiply by scale
    spatial = D_out * H_out * W_out
    scale = _weights['scale'].view(-1)
    output = torch.empty_like(clamped)
    grid = (B * C_out,)
    softmax_multiply_kernel[grid](clamped, scale, output, B, C_out, spatial, BLOCK=BLOCK)

    return output
