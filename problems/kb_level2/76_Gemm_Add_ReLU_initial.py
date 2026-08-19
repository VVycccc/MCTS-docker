import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = x + self.bias
        x = torch.relu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bias_shape]
# Frozen weights (loaded from .pt at module init):
#   bias: [8192] (float32)
#   gemm.weight: [8192, 8192] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/76_Gemm_Add_ReLU_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    bias = _weights['bias']  # [8192]
    gemm_weight = _weights['gemm.weight']  # [8192, 8192]

    x = _torch.nn.functional.linear(x, gemm_weight)
    x = x + bias
    x = _torch.relu(x)
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
