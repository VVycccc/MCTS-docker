import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.instance_norm(x)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        x = x * self.multiplier
        x = torch.max(x, dim=1)[0]
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]
# Frozen weights (loaded from .pt at module init):
#   multiplier: [16, 1, 1, 1] (float32)
#   conv.weight: [16, 3, 3, 3, 3] (float32)
#   conv.bias: [16] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max_weights.pt"
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
    multiplier = _weights['multiplier']  # [16, 1, 1, 1]

    x = _torch.nn.functional.conv3d(x, conv_weight, conv_bias)
    x = x * multiplier
    x = _torch.clamp(x, -1.0, 1.0)
    x = x * multiplier
    x = _torch.max(x, dim=1)[0]
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
