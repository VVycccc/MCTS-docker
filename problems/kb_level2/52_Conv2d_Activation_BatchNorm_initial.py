import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies activation, and then applies Batch Normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        x = torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x)
        x = self.bn(x)
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# Frozen weights (loaded from .pt at module init):
#   conv.weight: [128, 64, 3, 3] (float32)
#   conv.bias: [128] (float32)
#   bn.weight: [128] (float32)
#   bn.bias: [128] (float32)
#   bn.running_mean: [128] (float32)
#   bn.running_var: [128] (float32)
#   bn.num_batches_tracked: [] (int64)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/52_Conv2d_Activation_BatchNorm_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    bn_bias = _weights['bn.bias']  # [128]
    bn_num_batches_tracked = _weights['bn.num_batches_tracked']  # []
    bn_running_mean = _weights['bn.running_mean']  # [128]
    bn_running_var = _weights['bn.running_var']  # [128]
    bn_weight = _weights['bn.weight']  # [128]
    conv_bias = _weights['conv.bias']  # [128]
    conv_weight = _weights['conv.weight']  # [128, 64, 3, 3]

    x = _torch.nn.functional.conv2d(x, conv_weight, conv_bias)
    x = _torch.tanh(x)
    x = _torch.nn.functional.batch_norm(x, running_mean=None, running_var=None, weight=bn_weight, bias=bn_bias, training=True)
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
