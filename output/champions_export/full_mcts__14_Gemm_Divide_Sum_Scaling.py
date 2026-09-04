import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 32, 'BLOCK_I': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_B': 16, 'BLOCK_I': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_B': 32, 'BLOCK_I': 1024}, num_stages=3, num_warps=8),
    ],
    key=['batch_size', 'input_size'],
)
@triton.jit
def fused_gemm_divide_sum_scaling_kernel(
    x_ptr, w_sum_ptr, output_ptr,
    batch_size, input_size,
    stride_xb, stride_xi,
    BLOCK_B: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid = tl.program_id(0)
    
    b_offsets = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offsets < batch_size
    
    acc = tl.zeros((BLOCK_B,), dtype=tl.float32)
    
    for i_start in range(0, input_size, BLOCK_I):
        i_offsets = i_start + tl.arange(0, BLOCK_I)
        i_mask = i_offsets < input_size
        
        w_vals = tl.load(w_sum_ptr + i_offsets, mask=i_mask, other=0.0)
        
        x_ptrs = x_ptr + b_offsets[:, None] * stride_xb + i_offsets[None, :] * stride_xi
        x_mask = b_mask[:, None] & i_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        acc += tl.sum(x_vals * w_vals[None, :], axis=1)
    
    tl.store(output_ptr + b_offsets, acc, mask=b_mask)


batch_size   = 1024  
input_size   = 8192  
hidden_size  = 8192 
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/14_Gemm_Divide_Sum_Scaling_weights.pt"
_w_sum = None
def run(x, *args):
    global _w_sum
    if _w_sum is None or _w_sum.device != x.device:
        weight = torch.load(_weights_path, map_location='cpu', weights_only=True)['weight']
        _w_sum = (weight.sum(dim=0) * (scaling_factor / 2.0)).to(x.device).to(x.dtype)
    
    bs = x.shape[0]
    isz = x.shape[1]
    
    output = torch.zeros(bs, dtype=x.dtype, device=x.device)
    
    grid = (triton.cdiv(bs, 32),)
    
    fused_gemm_divide_sum_scaling_kernel[grid](
        x, _w_sum, output,
        bs, isz,
        x.stride(0), x.stride(1),
    )
    
    return output.unsqueeze(1)
