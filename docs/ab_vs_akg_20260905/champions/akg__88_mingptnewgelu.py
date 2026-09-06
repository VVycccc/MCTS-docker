import torch
import triton
import triton.language as tl
import math


@triton.jit
def new_gelu_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    NewGELU激活函数内核
    公式: 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
    每个程序处理 BLOCK_SIZE 个元素
    """
    # 获取当前程序的 PID
    pid = tl.program_id(axis=0)
    
    # 计算当前程序处理的数据偏移
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建掩码以防止越界访问
    mask = offsets < n_elements
    
    # 从内存加载数据
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # 计算 NewGELU
    # sqrt(2.0 / pi) ≈ 0.7978845608028654
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    
    # 数值稳定的tanh实现: tanh(x) = 2 / (1 + exp(-2x)) - 1
    # 对于大正x: exp(-2x) -> 0, tanh -> 1
    # 对于大负x: exp(-2x) -> inf, tanh -> -1
    tanh_inner = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
    
    # 最终结果: 0.5 * x * (1.0 + tanh(...))
    result = 0.5 * x * (1.0 + tanh_inner)
    
    # 将结果写回内存
    tl.store(output_ptr + offsets, result, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Triton NewGELU 激活函数
        实现: 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
        """
        # 获取输入元素总数
        n_elements = x.numel()
        
        # 预分配输出张量
        output = torch.empty_like(x)
        
        # 设置块大小
        BLOCK_SIZE = 1024
        
        # 计算网格大小
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        # 启动内核
        new_gelu_kernel[grid](
            output,
            x,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output

# --- DirecTune shim (.pt weights aligned, cached) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class Model(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim)]

def get_init_inputs():
    return []

_weights_path = "/home/wangyichen/DirecTune-MCTS/problems/kb_level1/88_mingptnewgelu_weights.pt"
_MODEL = None
def run(x, *args):
    global _MODEL
    if _MODEL is None:
        import torch
        torch.manual_seed(0)
        _MODEL = ModelNew(*get_init_inputs())
        try:
            _ref = Model(*get_init_inputs())
            _ref.load_state_dict(torch.load(_weights_path, map_location='cpu', weights_only=True))
            _rp = list(_ref.parameters()); _np = list(_MODEL.parameters())
            for _pn, _pr in zip(_np, _rp):
                if _pn.shape == _pr.shape:
                    _pn.data.copy_(_pr.data)
        except FileNotFoundError:
            pass  # 权重文件缺失：保留 ModelNew 的 seed 初始化（与 AKG 内部验证条件一致）
        _MODEL = _MODEL.to(x.device).eval()
    return _MODEL(x, *args)
