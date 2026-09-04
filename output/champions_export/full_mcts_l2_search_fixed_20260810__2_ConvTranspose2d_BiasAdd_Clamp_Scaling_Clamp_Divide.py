import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device).contiguous() for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)

@triton.jit
def conv_transpose2d_kernel(
    x_ptr, weight_ptr, conv_bias_ptr, bias_ptr, output_ptr,
    B, IC: tl.constexpr, OC, H, W, OH, OW,
    K: tl.constexpr, S: tl.constexpr, P: tl.constexpr,
    SCALING,
    BLOCK_SPATIAL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid = tl.program_id(0)

    num_spatial = OH * OW
    num_spatial_blocks = tl.cdiv(num_spatial, BLOCK_SPATIAL)
    num_oc_blocks = tl.cdiv(OC, BLOCK_OC)

    blocks_per_batch = num_spatial_blocks * num_oc_blocks
    b = pid // blocks_per_batch
    pid_in_batch = pid % blocks_per_batch

    spatial_block = pid_in_batch // num_oc_blocks
    oc_block = pid_in_batch % num_oc_blocks

    # Spatial positions
    spatial_offs = spatial_block * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    spatial_mask = spatial_offs < num_spatial

    oy = spatial_offs // OW
    ox = spatial_offs % OW

    # Output channels
    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SPATIAL, BLOCK_OC), dtype=tl.float32)

    # IC offsets
    ic_offs_1d = tl.arange(0, BLOCK_IC)

    # Loop over kernel positions
    for ky in range(K):
        for kx in range(K):
            iy_raw = oy + P - ky
            ix_raw = ox + P - kx
            cond = (iy_raw % S == 0) & (ix_raw % S == 0)
            iy = iy_raw // S
            ix = ix_raw // S
            in_bounds = cond & (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)

            spatial_idx = iy * W + ix  # (BLOCK_SPATIAL,)

            for ic_base in range(0, IC, BLOCK_IC):
                ic_offs = ic_base + ic_offs_1d
                ic_mask = ic_offs < IC

                # Input offsets: (BLOCK_SPATIAL, BLOCK_IC)
                x_offsets = b * IC * H * W + ic_offs[None, :] * (H * W) + spatial_idx[:, None]
                x_mask = spatial_mask[:, None] & ic_mask[None, :] & in_bounds[:, None]
                x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0).to(tl.float32)

                # Weight offsets: (BLOCK_IC, BLOCK_OC)
                w_offsets = ic_offs[:, None] * (OC * K * K) + oc_offs[None, :] * (K * K) + (ky * K + kx)
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(weight_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

                # Dot: (BLOCK_SPATIAL, BLOCK_IC) @ (BLOCK_IC, BLOCK_OC) = (BLOCK_SPATIAL, BLOCK_OC)
                acc = tl.dot(x_vals, w_vals, acc=acc, allow_tf32=True)

    # Add conv bias and regular bias
    conv_bias_vals = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    bias_vals = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    acc = acc + (conv_bias_vals + bias_vals)[None, :]

    # Clamp, scale, clamp, divide
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 1.0)
    acc = acc * SCALING
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 1.0)
    acc = acc / SCALING

    # Store
    out_offsets = b * OC * OH * OW + oc_offs[None, :] * (OH * OW) + spatial_offs[:, None]
    out_mask = spatial_mask[:, None] & oc_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)

def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    x = x.contiguous()

    bias = _weights['bias']
    conv_transpose_bias = _weights['conv_transpose.bias']
    conv_transpose_weight = _weights['conv_transpose.weight']

    B, IC, H, W = x.shape
    OC = conv_transpose_weight.shape[1]
    K = conv_transpose_weight.shape[2]
    S = 2
    P = 1
    OP = 1
    SCALING = 2.0

    OH = (H - 1) * S - 2 * P + K + OP
    OW = (W - 1) * S - 2 * P + K + OP

    output = torch.empty(B, OC, OH, OW, device=x.device, dtype=torch.float32)

    # Choose block sizes based on dimensions
    def next_pow2(n):
        p = 1
        while p < n:
            p *= 2
        return p

    BLOCK_OC = min(64, max(16, next_pow2(OC)))
    BLOCK_IC = min(32, max(16, next_pow2(IC)))
    BLOCK_SPATIAL = 64

    num_spatial_blocks = triton.cdiv(OH * OW, BLOCK_SPATIAL)
    num_oc_blocks = triton.cdiv(OC, BLOCK_OC)
    grid = (B * num_spatial_blocks * num_oc_blocks,)

    conv_transpose2d_kernel[grid](
        x, conv_transpose_weight, conv_transpose_bias, bias, output,
        B, IC, OC, H, W, OH, OW,
        K, S, P,
        SCALING,
        BLOCK_SPATIAL,
        BLOCK_OC,
        BLOCK_IC,
        num_warps=4,
        num_stages=3,
    )

    return output
