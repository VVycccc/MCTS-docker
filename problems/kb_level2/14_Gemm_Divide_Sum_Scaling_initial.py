import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = torch.matmul(x, self.weight.T)  # Gemm
        x = x / 2  # Divide
        x = torch.sum(x, dim=1, keepdim=True) # Sum
        x = x * self.scaling_factor  # Scaling
        return x


batch_size   = 1024  
input_size   = 8192  
hidden_size  = 8192 
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
# Frozen weights (loaded from .pt at module init):
#   weight: [8192, 8192] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/14_Gemm_Divide_Sum_Scaling_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    weight = _weights['weight']  # [8192, 8192]

    x = _torch.matmul(x, weight.T)
    x = x / 2  # Divide
    x = _torch.sum(x, dim=1, keepdim=True) # Sum
    x = x * 1.5  # Scaling
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
