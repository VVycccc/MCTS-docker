import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, batch normalization, tanh activation, max pooling, and group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.tanh(x)
        x = self.max_pool(x)
        x = self.group_norm(x)
        return x

batch_size = 512
in_channels  = 64  
out_channels = 128  
height = width = 2048  
kernel_size  = 5
stride       = 1  
padding      = 1
groups       = 8
num_groups   = 8
height, width = 32, 32

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]
# Frozen weights (loaded from .pt at module init):
#   conv_transpose.weight: [64, 128, 5, 5] (float32)
#   conv_transpose.bias: [128] (float32)
#   batch_norm.weight: [128] (float32)
#   batch_norm.bias: [128] (float32)
#   batch_norm.running_mean: [128] (float32)
#   batch_norm.running_var: [128] (float32)
#   batch_norm.num_batches_tracked: [] (int64)
#   group_norm.weight: [128] (float32)
#   group_norm.bias: [128] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    batch_norm_bias = _weights['batch_norm.bias']  # [128]
    batch_norm_num_batches_tracked = _weights['batch_norm.num_batches_tracked']  # []
    batch_norm_running_mean = _weights['batch_norm.running_mean']  # [128]
    batch_norm_running_var = _weights['batch_norm.running_var']  # [128]
    batch_norm_weight = _weights['batch_norm.weight']  # [128]
    conv_transpose_bias = _weights['conv_transpose.bias']  # [128]
    conv_transpose_weight = _weights['conv_transpose.weight']  # [64, 128, 5, 5]
    group_norm_bias = _weights['group_norm.bias']  # [128]
    group_norm_weight = _weights['group_norm.weight']  # [128]

    x = _torch.nn.functional.conv_transpose2d(x, conv_transpose_weight, conv_transpose_bias)
    x = _torch.nn.functional.batch_norm(x, running_mean=None, running_var=None, weight=batch_norm_weight, bias=batch_norm_bias, training=True)
    x = _torch.tanh(x)
    x = _torch.nn.functional.max_pool2d(x, kernel_size=2)
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
