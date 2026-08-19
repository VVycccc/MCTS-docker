import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that performs:
    1. Conv3D
    2. HardSwish activation
    3. GroupNorm  
    4. Mean pooling across spatial dimensions
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)                             # (B, C, D, H, W)
        x = F.hardswish(x)                           # Nonlinear activation
        x = self.group_norm(x)                       # Normalization over channels
        x = torch.mean(x, dim=[2, 3, 4])             # Mean over spatial dims → (B, C)
        return x

# === Test config ===
batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# Frozen weights (loaded from .pt at module init):
#   conv.weight: [16, 3, 4, 4, 4] (float32)
#   conv.bias: [16] (float32)
#   group_norm.weight: [16] (float32)
#   group_norm.bias: [16] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/27_Conv3d_HardSwish_GroupNorm_Mean_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    conv_bias = _weights['conv.bias']  # [16]
    conv_weight = _weights['conv.weight']  # [16, 3, 4, 4, 4]
    group_norm_bias = _weights['group_norm.bias']  # [16]
    group_norm_weight = _weights['group_norm.weight']  # [16]

    x = _torch.nn.functional.conv3d(x, conv_weight, conv_bias)
    x = _torch.nn.functional.group_norm(x, num_groups=1, weight=group_norm_weight, bias=group_norm_bias)
    x = _torch.mean(x, dim=[2, 3, 4])             # Mean over spatial dims → (B, C)
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
