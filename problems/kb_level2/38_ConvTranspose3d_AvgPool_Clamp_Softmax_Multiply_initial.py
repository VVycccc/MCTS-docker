import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs average pooling, 3D transposed convolution, clamping,
    spatial softmax, and multiplication by a learnable scale.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1)                     # flatten spatial dims
        x = torch.softmax(x, dim=2)
        x = x.view(b, c, d, h, w)
        x = x * self.scale
        return x

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 32, 64, 64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]

# Frozen weights (loaded from .pt at module init):
#   scale: [1, 64, 1, 1, 1] (float32)
#   conv_transpose.weight: [32, 64, 3, 3, 3] (float32)
#   conv_transpose.bias: [64] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    conv_transpose_bias = _weights['conv_transpose.bias']  # [64]
    conv_transpose_weight = _weights['conv_transpose.weight']  # [32, 64, 3, 3, 3]
    scale = _weights['scale']  # [1, 64, 1, 1, 1]

    x = _torch.nn.functional.avg_pool3d(x, kernel_size=2)
    x = _torch.nn.functional.conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias)
    x = _torch.clamp(x, 0.0, 1.0)
    x = _torch.softmax(x, dim=-1)
    x = x * scale
    return x

# --- UNIVERSAL RUN (验证用，调 Model.forward) ---
_model_cache = None
_model_device = None

def run(x):
    global _model_cache, _model_device
    if _model_cache is None or _model_device != str(x.device):
        _model_cache = Model(*get_init_inputs())
        _model_cache.load_state_dict(_torch.load(_weights_path, map_location='cpu', weights_only=True))
        _model_cache = _model_cache.to(x.device).eval()
        _model_device = str(x.device)
    return _model_cache(x)
