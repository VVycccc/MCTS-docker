import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = x + self.bias
        x = x * self.scale
        x = torch.sigmoid(x)
        x = self.group_norm(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]
# Frozen weights (loaded from .pt at module init):
#   bias: [32, 1, 1] (float32)
#   scale: [32, 1, 1] (float32)
#   conv.weight: [32, 8, 3, 3] (float32)
#   conv.bias: [32] (float32)
#   group_norm.weight: [32] (float32)
#   group_norm.bias: [32] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/21_Conv2d_Add_Scale_Sigmoid_GroupNorm_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    bias = _weights['bias']  # [32, 1, 1]
    conv_bias = _weights['conv.bias']  # [32]
    conv_weight = _weights['conv.weight']  # [32, 8, 3, 3]
    group_norm_bias = _weights['group_norm.bias']  # [32]
    group_norm_weight = _weights['group_norm.weight']  # [32]
    scale = _weights['scale']  # [32, 1, 1]

    x = _torch.nn.functional.conv2d(x, conv_weight, conv_bias)
    x = x + bias
    x = x * scale
    x = _torch.sigmoid(x)
    x = _torch.nn.functional.group_norm(x, num_groups=1, weight=group_norm_weight, bias=group_norm_bias)
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
