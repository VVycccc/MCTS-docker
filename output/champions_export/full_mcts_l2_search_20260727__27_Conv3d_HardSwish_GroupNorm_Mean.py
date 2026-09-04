import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/27_Conv3d_HardSwish_GroupNorm_Mean_weights.pt"
_weights = None

def _init_weights(device):
    global _weights
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

@triton.jit
def hardswish_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_plus_3 = x + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    y = x * relu6 / 6.0
    tl.store(out_ptr + offs, y, mask=mask)

def run(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)

    conv_weight = _weights['conv.weight']
    conv_bias = _weights['conv.bias']
    gn_weight = _weights['group_norm.weight']
    gn_bias = _weights['group_norm.bias']

    B, C_in, D, H, W = x.shape
    C_out, _, kD, kH, kW = conv_weight.shape

    D_out = D - kD + 1
    H_out = H - kH + 1
    W_out = W - kW + 1

    # Step 1: Conv3D — naive loop over kernel taps, matmul per tap
    conv_out = torch.zeros(B, C_out, D_out, H_out, W_out, device=x.device, dtype=torch.float32)
    for kd in range(kD):
        for kh in range(kH):
            for kw in range(kW):
                x_slice = x[:, :, kd:kd + D_out, kh:kh + H_out, kw:kw + W_out]
                w_slice = conv_weight[:, :, kd, kh, kw]
                x_p = x_slice.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, C_in)
                contrib = (x_p @ w_slice.t()).reshape(B, D_out, H_out, W_out, C_out).permute(0, 4, 1, 2, 3)
                conv_out += contrib
    conv_out = conv_out + conv_bias.view(1, C_out, 1, 1, 1)

    # Step 2: HardSwish via Triton kernel
    conv_flat = conv_out.contiguous().view(-1)
    N = conv_flat.numel()
    hw_out = torch.empty_like(conv_flat)
    BLOCK = 32
    grid = (triton.cdiv(N, BLOCK),)
    hardswish_kernel[grid](conv_flat, hw_out, N, BLOCK=BLOCK)
    hw_out = hw_out.view(B, C_out, D_out, H_out, W_out)

    # Step 3: GroupNorm (num_groups=4)
    num_groups = 4
    channels_per_group = C_out // num_groups
    x_gn = hw_out.reshape(B, num_groups, channels_per_group, D_out, H_out, W_out)
    mean = x_gn.mean(dim=[2, 3, 4, 5], keepdim=True)
    var = ((x_gn - mean) ** 2).mean(dim=[2, 3, 4, 5], keepdim=True)
    x_normed = (x_gn - mean) / torch.sqrt(var + 1e-5)
    x_normed = x_normed.reshape(B, C_out, D_out, H_out, W_out)
    gn_out = x_normed * gn_weight.view(1, C_out, 1, 1, 1) + gn_bias.view(1, C_out, 1, 1, 1)

    # Step 4: Mean over spatial dims
    result = gn_out.mean(dim=[2, 3, 4])

    return result
