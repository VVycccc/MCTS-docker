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
    BLOCK_OC: tl.constexpr, BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr,
    BLOCK_IC: tl.constexpr, BLOCK_SPATIAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_tile = tl.program_id(2)

    num_ow_blocks = tl.cdiv(OW, BLOCK_OW)
    oh_block = pid_tile // num_ow_blocks
    ow_block = pid_tile % num_ow_blocks

    # Flattened output spatial positions
    pos = tl.arange(0, BLOCK_SPATIAL)
    oh_pos = pos // BLOCK_OW + oh_block * BLOCK_OH
    ow_pos = pos % BLOCK_OW + ow_block * BLOCK_OW
    spatial_mask = (oh_pos < OH) & (ow_pos < OW)

    # Output channel offsets
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Input channel offsets
    ic_offs = tl.arange(0, BLOCK_IC)
    ic_mask = ic_offs < IC

    # Initialize accumulator with bias
    bias_val = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    bias_val_f32 = tl.cast(bias_val, tl.float32)
    acc = tl.zeros((BLOCK_OC, BLOCK_SPATIAL), dtype=tl.float32)
    acc += bias_val_f32[:, None]

    # ConvTranspose2d via im2col + GEMM
    # weight_rearranged layout: [9, OC, IC] where index = kh*3+kw
    for kh_kw in range(9):
        kh = kh_kw // 3
        kw = kh_kw % 3

        # Compute input indices for each output position
        ih_num = oh_pos + 1 - kh
        iw_num = ow_pos + 1 - kw
        ih = ih_num // 2
        iw = iw_num // 2

        ih_valid = (ih_num >= 0) & (ih_num % 2 == 0) & (ih < H)
        iw_valid = (iw_num >= 0) & (iw_num % 2 == 0) & (iw < W)
        valid = ih_valid & iw_valid & spatial_mask

        # im2col: gather input [BLOCK_IC, BLOCK_SPATIAL]
        x_ptrs = x_ptr + pid_n * IC * H * W + ic_offs[:, None] * H * W + ih[None, :] * W + iw[None, :]
        x_mask = ic_mask[:, None] & valid[None, :]
        x_val = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load weight tile [BLOCK_OC, BLOCK_IC]
        w_ptrs = weight_ptr + kh_kw * OC * IC + oc_offs[:, None] * IC + ic_offs[None, :]
        w_mask = oc_mask[:, None] & ic_mask[None, :]
        w_val = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # GEMM via Tensor Core
        acc = tl.dot(w_val, x_val, acc=acc, allow_tf32=True)

    # Mish: x * tanh(softplus(x))
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(tl.minimum(acc, 20.0))))
    mish = acc * (2.0 * tl.sigmoid(2.0 * sp) - 1.0)

    # Add 0.5
    y = mish + 0.5

    # Hardtanh [-1, 1]
    y = tl.maximum(-1.0, tl.minimum(1.0, y))

    # Scale by 2
    y = y * 2.0

    # Store output
    out_ptrs = output_ptr + pid_n * OC * OH * OW + oc_offs[:, None] * OH * OW + oh_pos[None, :] * OW + ow_pos[None, :]
    tl.store(out_ptrs, y, mask=oc_mask[:, None] & spatial_mask[None, :])

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

    # Pre-arrange weight: [IC, OC, 3, 3] → [9, OC, IC] for im2col GEMM
    weight_rearranged = weight.permute(2, 3, 1, 0).reshape(9, OC, IC).contiguous()
    if weight_rearranged.dtype != x.dtype:
        weight_rearranged = weight_rearranged.to(x.dtype)

    output = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_OH = 8
    BLOCK_OW = 8
    BLOCK_SPATIAL = BLOCK_OH * BLOCK_OW  # 64
    BLOCK_IC = triton.next_power_of_2(IC)

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH, BLOCK_OH) * triton.cdiv(OW, BLOCK_OW))

    kernel[grid](
        x, weight_rearranged, bias, output,
        N, IC, OC, H, W, OH, OW,
        BLOCK_OC=BLOCK_OC, BLOCK_OH=BLOCK_OH, BLOCK_OW=BLOCK_OW,
        BLOCK_IC=BLOCK_IC, BLOCK_SPATIAL=BLOCK_SPATIAL,
        num_warps=4, num_stages=2,
    )

    return output
