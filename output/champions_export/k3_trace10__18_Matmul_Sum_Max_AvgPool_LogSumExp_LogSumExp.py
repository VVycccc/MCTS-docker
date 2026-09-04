import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp_weights.pt"
_W = None
_W_device = None
_w_sum = None
_b_sum = 0.0


@triton.jit
def _matmul_sum_kernel(
    x_ptr, wsum_ptr, out_ptr, b_sum,
    K,
    BLOCK_K: tl.constexpr,
):
    # sum over out_features commutes with the linear layer:
    #   sum_j( x[row] . W[j] + b[j] ) == x[row] . (sum_j W[j]) + sum_j b[j].
    # The frozen [N, K] weight is pre-folded into a single [K] vector (wsum),
    # turning the O(M*N*K) GEMM into an O(M*K) GEMV. One program per batch
    # row: blocked multiply-add over K, then a final tl.sum tree reduction.
    # max / mean / logsumexp on the singleton dim are identities (logsumexp of
    # one element is the element itself), so nothing extra is needed.
    row = tl.program_id(0)
    x_row = x_ptr + row * K

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x_vec = tl.load(x_row + offs_k)
        w_vec = tl.load(wsum_ptr + offs_k)
        acc += x_vec * w_vec

    total = tl.sum(acc, axis=0) + b_sum
    tl.store(out_ptr + row, total)


def run(x):
    global _W, _W_device, _w_sum, _b_sum
    if _W is None or _W_device != str(x.device):
        _W = {k: v.to(x.device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
        _W_device = str(x.device)
        # Fold the out_features summation into the frozen weights once:
        #   wsum[k] = sum_j W[j, k],  b_sum = sum_j b[j]
        _w_sum = _W['linear.weight'].sum(dim=0).contiguous()  # [in_features]
        _b_sum = float(_W['linear.bias'].sum())

    x = x.contiguous()

    batch_size, in_features = x.shape

    out = torch.empty((batch_size, 1), device=x.device, dtype=torch.float32)

    BLOCK_K = 1024
    grid = (batch_size,)
    _matmul_sum_kernel[grid](
        x, _w_sum, out, _b_sum,
        in_features,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=4,
    )
    return out
