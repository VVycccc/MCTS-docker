import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = self.fc(x)
        x = self.gn(x)
        x = self.leaky_relu(x)
        x = x + x
        return x


batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_groups]
# Frozen weights (loaded from .pt at module init):
#   fc.weight: [8192, 8192] (float32)
#   fc.bias: [8192] (float32)
#   gn.weight: [8192] (float32)
#   gn.bias: [8192] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/62_Matmul_GroupNorm_LeakyReLU_Sum_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    fc_bias = _weights['fc.bias']  # [8192]
    fc_weight = _weights['fc.weight']  # [8192, 8192]
    gn_bias = _weights['gn.bias']  # [8192]
    gn_weight = _weights['gn.weight']  # [8192]

    x = _torch.nn.functional.linear(x, fc_weight, fc_bias)
    x = _torch.nn.functional.group_norm(x, num_groups=1, weight=gn_weight, bias=gn_bias)
    x = _torch.nn.functional.leaky_relu(x)
    x = x + x
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
