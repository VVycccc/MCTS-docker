import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, LogSumExp, HardSwish, subtraction, clamp operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1)) 

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x * torch.sigmoid(x + 3) / 6
        x = x - self.bias
        x = torch.clamp(x, min=-1, max=1)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1)  

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]

# Frozen weights (loaded from .pt at module init):
#   bias: [1, 1, 1, 1] (float32)
#   conv_transpose.weight: [3, 16, 3, 3, 3] (float32)
#   conv_transpose.bias: [16] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    bias = _weights['bias']  # [1, 1, 1, 1]
    conv_transpose_bias = _weights['conv_transpose.bias']  # [16]
    conv_transpose_weight = _weights['conv_transpose.weight']  # [3, 16, 3, 3, 3]

    x = _torch.nn.functional.conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias)
    x = _torch.logsumexp(x, dim=1, keepdim=True)
    x = _torch.sigmoid(x)
    x = x - bias
    x = _torch.clamp(x, min=-1, max=1)
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
