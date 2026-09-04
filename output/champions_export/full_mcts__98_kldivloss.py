import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')

batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [(torch.rand(batch_size, *input_shape)*scale).softmax(dim=-1), torch.rand(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []


# --- EXPANDED REFERENCE ---
# run() 调用原始 Model.forward()（PyTorch 算，正确）。
# 写 Triton kernel 时：不要照抄 nn.Module，用 torch 实现（有权重则从 _weights_path 加载并缓存）。
import torch as _torch
_weights_path = "problems/kb_level1/98_kldivloss_weights.pt"
_model_cache = None
_model_device = None

def run(predictions, targets):
    global _model_cache, _model_device
    if _model_cache is None or _model_device != str(predictions.device):
        _model_cache = Model(*get_init_inputs())
        _model_cache = _model_cache.to(predictions.device).eval()
        _model_device = str(predictions.device)
    return _model_cache(predictions, targets)
