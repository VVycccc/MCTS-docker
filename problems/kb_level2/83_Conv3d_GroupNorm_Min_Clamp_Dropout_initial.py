import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, minimum, clamp, and dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = torch.min(x, torch.tensor(min_value, device=x.device))
        x = torch.clamp(x, min=min_value, max=max_value)
        x = self.dropout(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
groups = 8
min_value = 0.0
max_value = 1.0
dropout_p = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p]
# Frozen weights (loaded from .pt at module init):
#   conv.weight: [16, 3, 3, 3, 3] (float32)
#   conv.bias: [16] (float32)
#   norm.weight: [16] (float32)
#   norm.bias: [16] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/83_Conv3d_GroupNorm_Min_Clamp_Dropout_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    conv_bias = _weights['conv.bias']  # [16]
    conv_weight = _weights['conv.weight']  # [16, 3, 3, 3, 3]
    norm_bias = _weights['norm.bias']  # [16]
    norm_weight = _weights['norm.weight']  # [16]

    x = _torch.nn.functional.conv3d(x, conv_weight, conv_bias)
    x = _torch.nn.functional.group_norm(x, num_groups=1, weight=norm_weight, bias=norm_bias)
    x = _torch.min(x, _torch.tensor(min_value, device=x.device))
    x = _torch.clamp(x, min=min_value, max=max_value)
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
