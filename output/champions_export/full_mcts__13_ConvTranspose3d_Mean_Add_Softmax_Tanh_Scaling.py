import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_mean_add_softmax_tanh_scale_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    scaling_factor,
    B, C, D, H, W,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    y_stride_b, y_stride_c, y_stride_d, y_stride_h, y_stride_w,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    total_hw = H * W
    b = pid // total_hw
    hw_idx = pid % total_hw
    h = hw_idx // W
    w = hw_idx % W

    c_offsets = tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    x_base = b * stride_b + h * stride_h + w * stride_w

    # Mean over D for each channel c
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        offsets = x_base + c_offsets[:, None] * stride_c + d_offsets[None, :] * stride_d
        mask = c_mask[:, None] & d_mask[None, :]
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(x_vals, axis=1)

    mean_vals = acc / D

    # Add bias
    bias_vals = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0)
    vals = mean_vals + bias_vals

    # Softmax over C
    max_val = tl.max(vals, axis=0)
    exp_vals = tl.exp(vals - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_out = exp_vals / sum_exp

    # Tanh + Scale
    tanh_out = 2.0 / (1.0 + tl.exp(-2.0 * softmax_out)) - 1.0
    result = tanh_out * scaling_factor

    # Store
    y_base = b * y_stride_b + h * y_stride_h + w * y_stride_w
    y_offsets = y_base + c_offsets * y_stride_c
    tl.store(y_ptr + y_offsets, result, mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        torch.manual_seed(0)
        conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.weight = nn.Parameter(conv_transpose.weight.clone())
        self.bias_conv = nn.Parameter(conv_transpose.bias.clone()) if conv_transpose.bias is not None else None
        self.bias_add = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # 1. ConvTranspose3d via PyTorch
        x = torch.nn.functional.conv_transpose3d(
            x, self.weight, self.bias_conv,
            stride=self.stride, padding=self.padding
        )

        B, C, D, H, W = x.shape
        y_out = torch.empty((B, C, 1, H, W), dtype=x.dtype, device=x.device)
        bias_flat = self.bias_add.reshape(-1)

        BLOCK_C = triton.next_power_of_2(C)
        BLOCK_D = triton.next_power_of_2(D)

        grid = (B * H * W,)
        fused_mean_add_softmax_tanh_scale_kernel[grid](
            x, bias_flat, y_out, self.scaling_factor,
            B, C, D, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            y_out.stride(0), y_out.stride(1), y_out.stride(2), y_out.stride(3), y_out.stride(4),
            BLOCK_C=BLOCK_C,
            BLOCK_D=BLOCK_D,
        )

        return y_out

# --- DirecTune shim (.pt weights aligned, cached) ---
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a series of operations:
    1. Transposed 3D convolution
    2. Mean pooling (across depth)
    3. Addition
    4. Softmax (across channels)
    5. Tanh activation
    6. Scaling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))  # Broadcastable bias over channels
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)                            # (B, C, D, H, W)
        x = x.mean(dim=2, keepdim=True)                       # Mean pool over depth dim (D)
        x = x + self.bias                                     # Bias add per channel
        x = torch.softmax(x, dim=1)                           # Softmax over channels
        x = torch.tanh(x)                                     # Nonlinearity
        x = x * self.scaling_factor                           # Scaling
        return x

# === Test config ===
batch_size = 16
in_channels  = 16  
out_channels = 64  
depth = 32; height = width = 128  
kernel_size  = 3
stride       = 1  
padding = 1
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scaling_factor]

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling_weights.pt"
_MODEL = None
def run(x, *args):
    global _MODEL
    if _MODEL is None:
        torch.manual_seed(0)
        _MODEL = ModelNew(*get_init_inputs())
        _ref = Model(*get_init_inputs())
        _ref.load_state_dict(torch.load(_weights_path, map_location='cpu', weights_only=True))
        _rp = list(_ref.parameters()); _np = list(_MODEL.parameters())
        for _pn, _pr in zip(_np, _rp):
            if _pn.shape == _pr.shape:
                _pn.data.copy_(_pr.data)
        _MODEL = _MODEL.to(x.device).eval()
    return _MODEL(x, *args)
