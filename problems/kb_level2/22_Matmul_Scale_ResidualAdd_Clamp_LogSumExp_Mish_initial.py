import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = self.matmul(x)
        x = x * self.scale_factor
        x = x + x
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x * torch.nn.functional.mish(x)  # Mish activation
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]
# Frozen weights (loaded from .pt at module init):
#   matmul.weight: [8192, 8192] (float32)
#   matmul.bias: [8192] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    matmul_bias = _weights['matmul.bias']  # [8192]
    matmul_weight = _weights['matmul.weight']  # [8192, 8192]

    x = _torch.nn.functional.linear(x, matmul_weight, matmul_bias)
    x = x * 2.0
    x = x + x
    x = _torch.clamp(x, -10.0, 10.0)
    x = _torch.logsumexp(x, dim=1, keepdim=True)
    x = _torch.nn.functional.mish(x)
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
