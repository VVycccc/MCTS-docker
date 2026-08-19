import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, applies Group Normalization, Tanh, HardSwish, 
    Residual Addition, and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Group Normalization
        x_norm = self.group_norm(x_conv)
        # Tanh
        x_tanh = self.tanh(x_norm)
        # HardSwish
        x_hard_swish = self.hard_swish(x_tanh)
        # Residual Addition
        x_res = x_conv + x_hard_swish
        # LogSumExp
        x_logsumexp = torch.logsumexp(x_res, dim=1, keepdim=True)
        return x_logsumexp

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
groups = 16

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]
# Frozen weights (loaded from .pt at module init):
#   conv.weight: [64, 8, 3, 3] (float32)
#   conv.bias: [64] (float32)
#   group_norm.weight: [64] (float32)
#   group_norm.bias: [64] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    conv_bias = _weights['conv.bias']  # [64]
    conv_weight = _weights['conv.weight']  # [64, 8, 3, 3]
    group_norm_bias = _weights['group_norm.bias']  # [64]
    group_norm_weight = _weights['group_norm.weight']  # [64]

    x = _torch.nn.functional.conv2d(x, conv_weight, conv_bias)
    x = _torch.nn.functional.group_norm(x, num_groups=1, weight=group_norm_weight, bias=group_norm_bias)
    x_tanh = _torch.tanh(x_norm)
    x_logsumexp = _torch.logsumexp(x_res, dim=1, keepdim=True)
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
