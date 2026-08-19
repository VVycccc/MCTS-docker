import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies LeakyReLU, multiplies by a learnable parameter, 
    applies LeakyReLU again, and performs a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.leaky_relu(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        x = self.max_pool(x)
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape]
# Frozen weights (loaded from .pt at module init):
#   multiplier: [32, 1, 1, 1] (float32)
#   conv_transpose.weight: [16, 32, 3, 3, 3] (float32)
#   conv_transpose.bias: [32] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    conv_transpose_bias = _weights['conv_transpose.bias']  # [32]
    conv_transpose_weight = _weights['conv_transpose.weight']  # [16, 32, 3, 3, 3]
    multiplier = _weights['multiplier']  # [32, 1, 1, 1]

    x = _torch.nn.functional.conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias)
    x = _torch.nn.functional.leaky_relu(x)
    x = x * multiplier
    x = _torch.nn.functional.leaky_relu(x)
    x = _torch.nn.functional.max_pool3d(x, kernel_size=2)
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
