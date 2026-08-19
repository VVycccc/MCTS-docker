import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = x.mean(dim=[1, 2, 3, 4]) # Compute mean across all dimensions except batch
        return x

batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]
# Frozen weights (loaded from .pt at module init):
#   conv.weight: [24, 3, 3, 3, 3] (float32)
#   conv.bias: [24] (float32)
#   group_norm.weight: [24] (float32)
#   group_norm.bias: [24] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/23_Conv3d_GroupNorm_Mean_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    conv_bias = _weights['conv.bias']  # [24]
    conv_weight = _weights['conv.weight']  # [24, 3, 3, 3, 3]
    group_norm_bias = _weights['group_norm.bias']  # [24]
    group_norm_weight = _weights['group_norm.weight']  # [24]

    x = _torch.nn.functional.conv3d(x, conv_weight, conv_bias)
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
