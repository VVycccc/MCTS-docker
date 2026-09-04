import torch
import triton
import triton.language as tl

_weights_path = "problems/kb_level1/50_conv_standard_2d_square_input_square_kernel_weights.pt"
_params = None
_params_device = None


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K, stride, padding,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C_out * H_out * W_out
    mask = offs < total

    wo = offs % W_out
    tmp = offs // W_out
    ho = tmp % H_out
    tmp2 = tmp // H_out
    co = tmp2 % C_out
    n = tmp2 // C_out

    n_safe = tl.where(mask, n, 0)
    co_safe = tl.where(mask, co, 0)
    ho_safe = tl.where(mask, ho, 0)
    wo_safe = tl.where(mask, wo, 0)

    acc = tl.load(b_ptr + co_safe, mask=mask, other=0.0).to(tl.float32)

    x_base_n = n_safe * (C_in * H_in * W_in)
    w_base_co = co_safe * (C_in * K * K)

    for ci in range(0, C_in):
        for kh in range(0, K):
            h_in = ho_safe * stride - padding + kh
            valid_h = (h_in >= 0) & (h_in < H_in)
            h_in_clamped = tl.maximum(tl.minimum(h_in, H_in - 1), 0)
            x_base_ci = x_base_n + ci * (H_in * W_in) + h_in_clamped * W_in
            w_base_ci = w_base_co + ci * (K * K) + kh * K
            for kw in range(0, K):
                w_in = wo_safe * stride - padding + kw
                valid_w = (w_in >= 0) & (w_in < W_in)
                w_in_clamped = tl.maximum(tl.minimum(w_in, W_in - 1), 0)
                valid = mask & valid_h & valid_w

                x_offs = x_base_ci + w_in_clamped
                w_offs = w_base_ci + kw

                x_val = tl.load(x_ptr + x_offs, mask=valid, other=0.0).to(tl.float32)
                w_val = tl.load(w_ptr + w_offs, mask=mask, other=0.0).to(tl.float32)
                acc += x_val * w_val

    tl.store(y_ptr + offs, acc, mask=mask)


def run(x):
    global _params, _params_device
    if _params is None or _params_device != str(x.device):
        state = torch.load(_weights_path, map_location="cpu", weights_only=True)
        w = state["conv1.weight"].to(x.device)
        b = state["conv1.bias"].to(x.device)
        _params = (w, b)
        _params_device = str(x.device)

    weight, bias = _params
    x = x.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_out = weight.shape[0]
    K = weight.shape[2]  # square kernel
    stride = 4
    padding = 2

    H_out = (H_in + 2 * padding - K) // stride + 1
    W_out = (W_in + 2 * padding - K) // stride + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    total = y.numel()
    BLOCK = 64
    grid = (triton.cdiv(total, BLOCK),)
    conv2d_kernel[grid](
        x, weight, bias, y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding,
        BLOCK=BLOCK,
    )
    return y
