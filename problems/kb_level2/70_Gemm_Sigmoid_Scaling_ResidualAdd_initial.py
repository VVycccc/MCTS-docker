import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = self.gemm(x)
        original_x = x
        x = torch.sigmoid(x)
        x = x * self.scaling_factor
        x = x + original_x
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
# Frozen weights (loaded from .pt at module init):
#   gemm.weight: [8192, 8192] (float32)
#   gemm.bias: [8192] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/70_Gemm_Sigmoid_Scaling_ResidualAdd_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    gemm_bias = _weights['gemm.bias']  # [8192]
    gemm_weight = _weights['gemm.weight']  # [8192, 8192]

    x = _torch.nn.functional.linear(x, gemm_weight, gemm_bias)
    x = _torch.sigmoid(x)
    x = x * 2.0
    x = x + original_x
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
