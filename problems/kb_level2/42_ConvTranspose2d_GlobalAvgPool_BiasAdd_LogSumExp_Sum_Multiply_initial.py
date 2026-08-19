import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, global average pooling, adds a bias, applies log-sum-exp, sum, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
        x = x + self.bias
        x = torch.logsumexp(x, dim=1, keepdim=True)  # Log-sum-exp
        x = torch.sum(x, dim=(2, 3))  # Sum
        x = x * 10.0  # Multiplication
        return x

batch_size = 16
in_channels = 64
out_channels = 128
height = width = 512
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
# Frozen weights (loaded from .pt at module init):
#   bias: [128, 1, 1] (float32)
#   conv_transpose.weight: [64, 128, 3, 3] (float32)
#   conv_transpose.bias: [128] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    bias = _weights['bias']  # [128, 1, 1]
    conv_transpose_bias = _weights['conv_transpose.bias']  # [128]
    conv_transpose_weight = _weights['conv_transpose.weight']  # [64, 128, 3, 3]

    x = _torch.nn.functional.conv_transpose2d(x, conv_transpose_weight, conv_transpose_bias)
    x = _torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
    x = x + bias
    x = _torch.logsumexp(x, dim=1, keepdim=True)  # Log-sum-exp
    x = _torch.sum(x, dim=(2, 3))  # Sum
    x = x * 10.0  # Multiplication
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
