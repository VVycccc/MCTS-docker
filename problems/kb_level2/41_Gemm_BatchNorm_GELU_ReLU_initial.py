import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a GEMM, BatchNorm, GELU, and ReLU in sequence.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.batch_norm(x)
        x = torch.nn.functional.gelu(x)
        x = torch.relu(x)
        return x

batch_size = 16384
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
# Frozen weights (loaded from .pt at module init):
#   gemm.weight: [4096, 4096] (float32)
#   gemm.bias: [4096] (float32)
#   batch_norm.weight: [4096] (float32)
#   batch_norm.bias: [4096] (float32)
#   batch_norm.running_mean: [4096] (float32)
#   batch_norm.running_var: [4096] (float32)
#   batch_norm.num_batches_tracked: [] (int64)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/41_Gemm_BatchNorm_GELU_ReLU_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    batch_norm_bias = _weights['batch_norm.bias']  # [4096]
    batch_norm_num_batches_tracked = _weights['batch_norm.num_batches_tracked']  # []
    batch_norm_running_mean = _weights['batch_norm.running_mean']  # [4096]
    batch_norm_running_var = _weights['batch_norm.running_var']  # [4096]
    batch_norm_weight = _weights['batch_norm.weight']  # [4096]
    gemm_bias = _weights['gemm.bias']  # [4096]
    gemm_weight = _weights['gemm.weight']  # [4096, 4096]

    x = _torch.nn.functional.linear(x, gemm_weight, gemm_bias)
    x = _torch.nn.functional.batch_norm(x, running_mean=None, running_var=None, weight=batch_norm_weight, bias=batch_norm_bias, training=True)
    x = _torch.nn.functional.gelu(x)
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
