import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def scale_min_kernel(
    input_ptr, output_ptr,
    B, H, W,
    scale_factor,
    in_b_stride, in_c_stride, in_h_stride, in_w_stride,
    out_b_stride, out_h_stride, out_w_stride,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # 每个 program 处理一个 位置
    pid = tl.program_id(0)
    b = pid // (H * W)
    hw = pid % (H * W)
    h = hw // W
    w = hw % W

    # 沿通道维度取最小值（同时乘以 scale_factor）
    min_val = float('inf')
    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        mask = c_offs < C
        vals = tl.load(
            input_ptr + b * in_b_stride + c_offs * in_c_stride + h * in_h_stride + w * in_w_stride,
            mask=mask, other=float('inf')
        )
        vals = vals * scale_factor
        block_min = tl.min(vals, axis=0)
        min_val = tl.minimum(min_val, block_min)

    tl.store(
        output_ptr + b * out_b_stride + h * out_h_stride + w * out_w_stride,
        min_val
    )


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        torch.manual_seed(0)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone())
        self.scale_factor = scale_factor

    def forward(self, x):
        # 使用 F.conv2d 计算卷积
        conv_out = F.conv2d(x, self.weight, self.bias)
        B, C, H, W = conv_out.shape

        # 分配输出张量 (B, 1, H, W)
        output = torch.empty((B, 1, H, W), dtype=conv_out.dtype, device=conv_out.device)

        # 启动 scale + min kernel
        BLOCK_C = 128
        grid = (B * H * W,)
        scale_min_kernel[grid](
            conv_out, output,
            B, H, W,
            self.scale_factor,
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
            output.stride(0), output.stride(2), output.stride(3),
            C=C,
            BLOCK_C=BLOCK_C,
        )

        return output

# --- DirecTune shim (.pt weights aligned, cached) ---
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(x)
        x = x * self.scale_factor
        x = torch.min(x, dim=1, keepdim=True)[0]  # Minimum along channel dimension
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/32_Conv2d_Scaling_Min_weights.pt"
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
