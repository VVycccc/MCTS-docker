import torch
import triton
import triton.language as tl

_weights_path = "problems/kb_level1/50_conv_standard_2d_square_input_square_kernel_weights.pt"
_W = None
_W_device = None

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H_in, W_in, C_out, H_out, W_out,
    kH: tl.constexpr, kW: tl.constexpr, stride: tl.constexpr, padding: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_c_out_tiles = tl.cdiv(C_out, BLOCK_M)
    n = pid_nc // num_c_out_tiles
    c_out_start = (pid_nc % num_c_out_tiles) * BLOCK_M

    spatial_size = H_out * W_out
    spatial_start = pid_hw * BLOCK_N

    offs_m = c_out_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < C_out

    offs_n = spatial_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < spatial_size

    h_n = offs_n // W_out
    w_n = offs_n % W_out

    bias_vals = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc += bias_vals[:, None]

    K_total = C_in * kH * kW

    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_total

        c_in_k = offs_k // (kH * kW)
        kh_k = (offs_k % (kH * kW)) // kW
        kw_k = offs_k % kW

        w_offs = offs_m[:, None] * K_total + offs_k[None, :]
        a = tl.load(w_ptr + w_offs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        ih = h_n[None, :] * stride + kh_k[:, None] - padding
        iw = w_n[None, :] * stride + kw_k[:, None] - padding

        in_mask = (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in) & mask_k[:, None] & mask_n[None, :]

        x_offs = n * C_in * H_in * W_in + c_in_k[:, None] * H_in * W_in + ih * W_in + iw
        b = tl.load(x_ptr + x_offs, mask=in_mask, other=0.0)

        acc = tl.dot(a, b, acc=acc, allow_tf32=True)

    out_offs = n * C_out * spatial_size + offs_m[:, None] * spatial_size + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def run(x):
    global _W, _W_device
    if _W is None or _W_device != str(x.device):
        state_dict = torch.load(_weights_path, map_location='cpu', weights_only=True)
        _W = {
            'weight': state_dict['conv1.weight'].to(x.device).half().contiguous(),
            'bias': state_dict['conv1.bias'].to(x.device).half().contiguous(),
        }
        _W_device = str(x.device)

    x = x.contiguous().half()
    N, C_in, H_in, W_in = x.shape
    weight = _W['weight']
    bias = _W['bias']
    C_out, _, kH, kW = weight.shape
    stride = 4
    padding = 2
    H_out = (H_in + 2 * padding - kH) // stride + 1
    W_out = (W_in + 2 * padding - kW) // stride + 1

    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    spatial_size = H_out * W_out
    grid = (N * triton.cdiv(C_out, BLOCK_M), triton.cdiv(spatial_size, BLOCK_N))

    conv2d_kernel[grid](
        x, weight, bias, out,
        N, C_in, H_in, W_in, C_out, H_out, W_out,
        kH, kW, stride, padding,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_stages=3, num_warps=8,
    )

    return out
