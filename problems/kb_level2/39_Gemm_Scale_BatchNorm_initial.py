import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.scale
        x = self.bn(x)
        return x

batch_size = 16384
in_features = 4096
out_features = 4096
scale_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scale_shape]
# Frozen weights (loaded from .pt at module init):
#   scale: [4096] (float32)
#   gemm.weight: [4096, 4096] (float32)
#   gemm.bias: [4096] (float32)
#   bn.weight: [4096] (float32)
#   bn.bias: [4096] (float32)
#   bn.running_mean: [4096] (float32)
#   bn.running_var: [4096] (float32)
#   bn.num_batches_tracked: [] (int64)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/39_Gemm_Scale_BatchNorm_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    bn_bias = _weights['bn.bias']  # [4096]
    bn_num_batches_tracked = _weights['bn.num_batches_tracked']  # []
    bn_running_mean = _weights['bn.running_mean']  # [4096]
    bn_running_var = _weights['bn.running_var']  # [4096]
    bn_weight = _weights['bn.weight']  # [4096]
    gemm_bias = _weights['gemm.bias']  # [4096]
    gemm_weight = _weights['gemm.weight']  # [4096, 4096]
    scale = _weights['scale']  # [4096]

    x = _torch.nn.functional.linear(x, gemm_weight, gemm_bias)
    x = x * scale
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
