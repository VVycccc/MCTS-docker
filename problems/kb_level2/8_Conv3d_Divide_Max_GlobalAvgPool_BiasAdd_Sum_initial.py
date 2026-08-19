import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = self.conv(x)
        x = x / self.divisor
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x

batch_size   = 128  
in_channels  = 8            
out_channels = 16  
depth = 16; height = width = 64 
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
# Frozen weights (loaded from .pt at module init):
#   bias: [16, 1, 1, 1] (float32)
#   conv.weight: [16, 8, 3, 3, 3] (float32)
#   conv.bias: [16] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    bias = _weights['bias']  # [16, 1, 1, 1]
    conv_bias = _weights['conv.bias']  # [16]
    conv_weight = _weights['conv.weight']  # [16, 8, 3, 3, 3]

    x = _torch.nn.functional.conv3d(x, conv_weight, conv_bias)
    x = x / 2.0
    x = _torch.nn.functional.max_pool3d(x, kernel_size=2)
    x = x + bias
    x = _torch.sum(x, dim=1)
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
