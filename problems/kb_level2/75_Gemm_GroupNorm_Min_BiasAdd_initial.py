import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a GEMM, Group Normalization, Minimum operation, and Bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.gemm(x)
        x = self.group_norm(x)
        x = torch.min(x, dim=1, keepdim=True)[0] 
        x = x + self.bias
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1, out_features, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
# Frozen weights (loaded from .pt at module init):
#   bias: [1, 8192, 1, 1] (float32)
#   gemm.weight: [8192, 8192] (float32)
#   gemm.bias: [8192] (float32)
#   group_norm.weight: [8192] (float32)
#   group_norm.bias: [8192] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/75_Gemm_GroupNorm_Min_BiasAdd_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    bias = _weights['bias']  # [1, 8192, 1, 1]
    gemm_bias = _weights['gemm.bias']  # [8192]
    gemm_weight = _weights['gemm.weight']  # [8192, 8192]
    group_norm_bias = _weights['group_norm.bias']  # [8192]
    group_norm_weight = _weights['group_norm.weight']  # [8192]

    x = _torch.nn.functional.linear(x, gemm_weight, gemm_bias)
    x = _torch.nn.functional.group_norm(x, num_groups=1, weight=group_norm_weight, bias=group_norm_bias)
    x = _torch.min(x, dim=1, keepdim=True)[0]
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
