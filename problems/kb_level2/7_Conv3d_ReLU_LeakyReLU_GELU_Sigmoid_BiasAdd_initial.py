import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = torch.nn.functional.gelu(x)
        x = torch.sigmoid(x)
        x = x + self.bias
        return x

batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
# Frozen weights (loaded from .pt at module init):
#   bias: [32, 1, 1, 1] (float32)
#   conv.weight: [32, 8, 3, 3, 3] (float32)
#   conv.bias: [32] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    bias = _weights['bias']  # [32, 1, 1, 1]
    conv_bias = _weights['conv.bias']  # [32]
    conv_weight = _weights['conv.weight']  # [32, 8, 3, 3, 3]

    x = _torch.nn.functional.conv3d(x, conv_weight, conv_bias)
    x = _torch.relu(x)
    x = _torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    x = _torch.nn.functional.gelu(x)
    x = _torch.sigmoid(x)
    x = x + bias
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
