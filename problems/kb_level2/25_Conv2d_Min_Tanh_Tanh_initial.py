import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.min(x, dim=1, keepdim=True)[0] # Apply minimum operation along the channel dimension
        x = torch.tanh(x)
        x = torch.tanh(x)
        return x

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# Frozen weights (loaded from .pt at module init):
#   conv.weight: [64, 16, 3, 3] (float32)
#   conv.bias: [64] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/25_Conv2d_Min_Tanh_Tanh_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    conv_bias = _weights['conv.bias']  # [64]
    conv_weight = _weights['conv.weight']  # [64, 16, 3, 3]

    x = _torch.nn.functional.conv2d(x, conv_weight, conv_bias)
    x = _torch.min(x, dim=1, keepdim=True)[0] # Apply minimum operation along the channel dimension
    x = _torch.tanh(x)
    x = _torch.tanh(x)
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
