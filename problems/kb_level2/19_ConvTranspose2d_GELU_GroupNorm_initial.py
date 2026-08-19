import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.nn.functional.gelu(x)
        x = self.group_norm(x)
        return x

batch_size   = 128  
in_channels  = 64  
out_channels = 64  
height = width = 256  
kernel_size  = 3
stride       = 1
groups = 8
num_groups = 8

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]
# Frozen weights (loaded from .pt at module init):
#   conv_transpose.weight: [64, 64, 3, 3] (float32)
#   conv_transpose.bias: [64] (float32)
#   group_norm.weight: [64] (float32)
#   group_norm.bias: [64] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/19_ConvTranspose2d_GELU_GroupNorm_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    conv_transpose_bias = _weights['conv_transpose.bias']  # [64]
    conv_transpose_weight = _weights['conv_transpose.weight']  # [64, 64, 3, 3]
    group_norm_bias = _weights['group_norm.bias']  # [64]
    group_norm_weight = _weights['group_norm.weight']  # [64]

    x = _torch.nn.functional.conv_transpose2d(x, conv_transpose_weight, conv_transpose_bias)
    x = _torch.nn.functional.gelu(x)
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
