import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp_weights.pt"
_weight_cache = None
_weight_device = None


@triton.jit
def naive_fused_kernel(
    x_ptr, w_sum_ptr, b_sum_ptr, out_ptr,
    IN_FEATURES,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr
):
    pid = tl.program_id(0)
    x_row = x_ptr + pid * IN_FEATURES
    acc = tl.load(b_sum_ptr)
    for j in range(0, IN_FEATURES, BLOCK_K):
        offs = j + tl.arange(0, BLOCK_K)
        if EVEN_K:
            x_vec = tl.load(x_row + offs)
            w_vec = tl.load(w_sum_ptr + offs)
        else:
            mask = offs < IN_FEATURES
            x_vec = tl.load(x_row + offs, mask=mask, other=0.0)
            w_vec = tl.load(w_sum_ptr + offs, mask=mask, other=0.0)
        acc += tl.sum(x_vec * w_vec)
    tl.store(out_ptr + pid, acc)


def run(x):
    global _weight_cache, _weight_device

    if _weight_cache is None or _weight_device != str(x.device):
        _weight_cache = torch.load(_weights_path, map_location='cpu', weights_only=True)
        _weight_cache = {k: v.to(x.device) for k, v in _weight_cache.items()}
        _weight_cache['w_sum'] = _weight_cache['linear.weight'].sum(dim=0).contiguous()
        _weight_cache['b_sum'] = _weight_cache['linear.bias'].sum().reshape(1)
        _weight_device = str(x.device)

    x = x.contiguous()
    B, IN_FEATURES = x.shape

    out = torch.empty((B, 1), device=x.device, dtype=x.dtype)

    BLOCK_K = 8192
    EVEN_K = (IN_FEATURES % BLOCK_K == 0)
    grid = (B,)
    naive_fused_kernel[grid](
        x, _weight_cache['w_sum'], _weight_cache['b_sum'], out,
        IN_FEATURES,
        BLOCK_K=BLOCK_K,
        EVEN_K=EVEN_K,
        num_warps=16
    )

    return out
