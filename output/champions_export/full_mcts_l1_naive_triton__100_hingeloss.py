import torch
import triton
import triton.language as tl

@triton.jit
def hinge_loss_kernel(pred_ptr, target_ptr, out_ptr, N, D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    pred = tl.load(pred_ptr + offs, mask=mask, other=0.0)
    target_idx = offs % D
    target = tl.load(target_ptr + target_idx, mask=mask, other=0.0)

    val = 1.0 - pred * target
    val = tl.maximum(val, 0.0)
    val = tl.where(mask, val, 0.0)

    partial_sum = tl.sum(val, axis=0)
    tl.atomic_add(out_ptr, partial_sum)

def run(predictions, targets):
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    N = predictions.numel()
    D = targets.numel()
    BLOCK_SIZE = 1024

    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    output = torch.zeros(1, dtype=torch.float32, device=predictions.device)

    grid = (num_blocks,)
    hinge_loss_kernel[grid](predictions, targets, output, N, D, BLOCK_SIZE=BLOCK_SIZE)

    return (output[0] / N).float()
