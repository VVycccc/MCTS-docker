import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.min(x, dim=1, keepdim=True)[0]  # Minimum operation along channel dimension
        x = torch.sum(x, dim=2, keepdim=True)  # Sum operation along height dimension
        x = torch.nn.functional.gelu(x)  # GELU activation
        x = x + self.bias
        return x

batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
# Frozen weights (loaded from .pt at module init):
#   bias: [1, 1, 1] (float32)
#   conv_transpose.weight: [64, 128, 3, 3] (float32)
#   conv_transpose.bias: [128] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/36_ConvTranspose2d_Min_Sum_GELU_Add_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    bias = _weights['bias']  # [1, 1, 1]
    conv_transpose_bias = _weights['conv_transpose.bias']  # [128]
    conv_transpose_weight = _weights['conv_transpose.weight']  # [64, 128, 3, 3]

    x = _torch.nn.functional.conv_transpose2d(x, conv_transpose_weight, conv_transpose_bias)
    x = _torch.min(x, dim=1, keepdim=True)[0]  # Minimum operation along channel dimension
    x = _torch.sum(x, dim=2, keepdim=True)  # Sum operation along height dimension
    x = _torch.nn.functional.gelu(x)  # GELU activation
    x = x + bias
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
