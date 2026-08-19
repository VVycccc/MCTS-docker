import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)  # Subtract mean along spatial dimensions
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
# Frozen weights (loaded from .pt at module init):
#   conv_transpose.weight: [16, 32, 3, 3, 3] (float32)
#   conv_transpose.bias: [32] (float32)
#   batch_norm.weight: [32] (float32)
#   batch_norm.bias: [32] (float32)
#   batch_norm.running_mean: [32] (float32)
#   batch_norm.running_var: [32] (float32)
#   batch_norm.num_batches_tracked: [] (int64)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/15_ConvTranspose3d_BatchNorm_Subtract_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    batch_norm_bias = _weights['batch_norm.bias']  # [32]
    batch_norm_num_batches_tracked = _weights['batch_norm.num_batches_tracked']  # []
    batch_norm_running_mean = _weights['batch_norm.running_mean']  # [32]
    batch_norm_running_var = _weights['batch_norm.running_var']  # [32]
    batch_norm_weight = _weights['batch_norm.weight']  # [32]
    conv_transpose_bias = _weights['conv_transpose.bias']  # [32]
    conv_transpose_weight = _weights['conv_transpose.weight']  # [16, 32, 3, 3, 3]

    x = _torch.nn.functional.conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias)
    x = _torch.nn.functional.batch_norm(x, running_mean=None, running_var=None, weight=batch_norm_weight, bias=batch_norm_bias, training=True)
    x = x - _torch.mean(x, dim=(2, 3, 4), keepdim=True)  # Subtract mean along spatial dimensions
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
