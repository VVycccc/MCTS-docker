import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, x):
        x = self.conv(x)
        x = x - self.subtract1_value
        x = torch.tanh(x)
        x = x - self.subtract2_value
        x = self.avgpool(x)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
# Frozen weights (loaded from .pt at module init):
#   conv.weight: [128, 64, 3, 3] (float32)
#   conv.bias: [128] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/46_Conv2d_Subtract_Tanh_Subtract_AvgPool_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    conv_bias = _weights['conv.bias']  # [128]
    conv_weight = _weights['conv.weight']  # [128, 64, 3, 3]

    x = _torch.nn.functional.conv2d(x, conv_weight, conv_bias)
    x = x - 0.5
    x = _torch.tanh(x)
    x = x - 0.2
    x = _torch.nn.functional.avg_pool2d(x, kernel_size=2)
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
