import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, scales the output, applies batch normalization, 
    and then performs global average pooling. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale_factor
        x = self.batch_norm(x)
        x = self.global_avg_pool(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
# Frozen weights (loaded from .pt at module init):
#   conv_transpose.weight: [64, 128, 5, 5, 5] (float32)
#   conv_transpose.bias: [128] (float32)
#   batch_norm.weight: [128] (float32)
#   batch_norm.bias: [128] (float32)
#   batch_norm.running_mean: [128] (float32)
#   batch_norm.running_var: [128] (float32)
#   batch_norm.num_batches_tracked: [] (int64)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool_weights.pt"
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
    conv_transpose_weight = _weights['conv_transpose.weight']  # [64, 128, 5, 5, 5]

    x = _torch.nn.functional.conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias)
    x = x * 2.0
    x = _torch.nn.functional.batch_norm(x, running_mean=None, running_var=None, weight=batch_norm_weight, bias=batch_norm_bias, training=True)
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
