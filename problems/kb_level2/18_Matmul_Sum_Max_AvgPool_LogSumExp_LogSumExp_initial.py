import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.linear(x)  # (batch_size, out_features)
        x = torch.sum(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.max(x, dim=1, keepdim=True)[0] # (batch_size, 1)
        x = torch.mean(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
        return x

batch_size = 1024
in_features  = 8192  
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
# Frozen weights (loaded from .pt at module init):
#   linear.weight: [8192, 8192] (float32)
#   linear.bias: [8192] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    linear_bias = _weights['linear.bias']  # [8192]
    linear_weight = _weights['linear.weight']  # [8192, 8192]

    x = _torch.nn.functional.linear(x, linear_weight, linear_bias)
    x = _torch.sum(x, dim=1, keepdim=True) # (batch_size, 1)
    x = _torch.max(x, dim=1, keepdim=True)[0] # (batch_size, 1)
    x = _torch.mean(x, dim=1, keepdim=True) # (batch_size, 1)
    x = _torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
    x = _torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
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
