import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling_weights.pt"
_weights = None

def _init_weights(device):
    global _weights
    w = torch.load(_weights_path, map_location='cpu', weights_only=True)
    _weights = {k: v.to(device).contiguous() for k, v in w.items()}

@triton.jit
def kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    N, IC, OC, H, W, OH, OW,
    BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_tile = tl.program_id(2)

    num_ow_blocks = tl.cdiv(OW, BLOCK_OW)
    oh_block = pid_tile // num_ow_blocks
    ow_block = pid_tile % num_ow_blocks

    # Flatten (oh, ow) into 1D tile for tl.dot
    tile_offs = tl.arange(0, TILE_SIZE)
    oh = oh_block * BLOCK_OH + tile_offs // BLOCK_OW
    ow = ow_block * BLOCK_OW + tile_offs % BLOCK_OW
    oh_mask = oh < OH
    ow_mask = ow < OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Accumulator: [TILE_SIZE, BLOCK_OC]
    acc = tl.zeros((TILE_SIZE, BLOCK_OC), dtype=tl.float32)
    b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b[None, :]

    # ConvTranspose2d via tl.dot over IC dimension
    for ic_block in tl.range(0, tl.cdiv(IC, BLOCK_IC), num_stages=3):
        ic_offs = ic_block * BLOCK_IC + tl.arange(0, BLOCK_IC)
        ic_valid = ic_offs < IC

        for kh in range(3):
            for kw in range(3):
                ih_num = oh + 1 - kh
                iw_num = ow + 1 - kw
                ih = ih_num // 2
                iw = iw_num // 2
                valid = (ih_num >= 0) & (ih * 2 == ih_num) & (ih < H) & \
                        (iw_num >= 0) & (iw * 2 == iw_num) & (iw < W) & \
                        oh_mask & ow_mask

                ih_safe = tl.where(valid, ih, 0)
                iw_safe = tl.where(valid, iw, 0)

                # Load input: [TILE_SIZE, BLOCK_IC]
                x_ptrs = x_ptr + pid_n * IC * H * W + ic_offs[None, :] * H * W + (ih_safe * W + iw_safe)[:, None]
                x_vals = tl.load(x_ptrs, mask=valid[:, None] & ic_valid[None, :], other=0.0)

                # Load weight: [BLOCK_IC, BLOCK_OC]
                w_ptrs = weight_ptr + ic_offs[:, None] * OC * 9 + oc_offs[None, :] * 9 + kh * 3 + kw
                w_vals = tl.load(w_ptrs, mask=ic_valid[:, None] & oc_mask[None, :], other=0.0)

                acc = tl.dot(x_vals, w_vals, acc=acc, allow_tf32=True)

    # Mish: x * tanh(softplus(x))
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    mish = acc * (2.0 * tl.sigmoid(2.0 * sp) - 1.0)

    # Add 0.5, Hardtanh [-1, 1], Scale by 2
    y = mish + 0.5
    y = tl.maximum(-1.0, tl.minimum(1.0, y))
    y = y * 2.0

    valid_out = oh_mask & ow_mask
    out_ptrs = output_ptr + pid_n * OC * OH * OW + oc_offs[None, :] * OH * OW + (oh * OW + ow)[:, None]
    tl.store(out_ptrs, y, mask=valid_out[:, None] & oc_mask[None, :])

def run(x):
    global _weights
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)

    weight = _weights['conv_transpose.weight']
    bias = _weights['conv_transpose.bias']

    x = x.contiguous()
    N, IC, H, W = x.shape
    OC = weight.shape[1]
    stride = 2
    padding = 1
    output_padding = 1
    K = 3

    OH = (H - 1) * stride - 2 * padding + K + output_padding
    OW = (W - 1) * stride - 2 * padding + K + output_padding

    output = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OH = 8
    BLOCK_OW = 16
    BLOCK_OC = 64
    BLOCK_IC = 32
    TILE_SIZE = BLOCK_OH * BLOCK_OW

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH, BLOCK_OH) * triton.cdiv(OW, BLOCK_OW))

    kernel[grid](
        x, weight, bias, output,
        N, IC, OC, H, W, OH, OW,
        BLOCK_OH=BLOCK_OH, BLOCK_OW=BLOCK_OW,
        BLOCK_OC=BLOCK_OC, BLOCK_IC=BLOCK_IC,
        TILE_SIZE=TILE_SIZE,
        num_warps=8, num_stages=2,
    )

    return output
