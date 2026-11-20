import torch


def min_max_per_token_quant_kv(kv: torch.Tensor, qbit: int) -> torch.Tensor:
    max_val = kv.amax(dim=-1, keepdim=True)
    min_val = kv.amin(dim=-1, keepdim=True)

    max_int = 2**qbit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-9) / max_int
    zeros = -min_val
    # print(zeros)
    return torch.clamp(torch.round((kv + zeros) / scales), min_int, max_int) * scales - zeros


def max_per_token_quant_kv(kv: torch.Tensor, qbit: int, smooth: bool) -> torch.Tensor:
    if smooth:
        # print(kv.shape)
        kv = kv - torch.min(kv, dim=-2, keepdim=True)[0]
    if qbit == 1:
        return torch.sign(kv)
    channel_max = torch.max(torch.abs(kv), dim=-1, keepdim=True)[0]
    scale = channel_max / (2**qbit - 1)
    scale = torch.max(scale, torch.tensor(1e-6))
    return torch.round(kv / scale) * scale  # quantize then dequantize back

def py_int4_min_max_per_token_quant_kv(kv: torch.Tensor, qbit: int):
    """Return integer quant codes and fp16 min/max quant params for dense estimator tests."""
    max_val = kv.amax(dim=-1, keepdim=True)
    min_val = kv.amin(dim=-1, keepdim=True)
    max_int = 2**qbit - 1
    scales = ((max_val - min_val).clamp(min=1e-6) / max_int).to(kv.dtype)
    zeros = min_val.to(kv.dtype)
    q = torch.clamp(torch.round((kv - zeros) / scales), 0, max_int).to(torch.uint8)
    return q, scales, zeros


def pack_min_max_int4_k_for_gemv(kv: torch.Tensor):
    """Pack per-token min/max int4 K cache for ftka int4 GEMV.

    Input layout:  [bs, num_heads, seq_len, head_dim]
    Output layout: [bs * seq_len, 1, num_heads, head_dim // 2]

    The ftka dequant path reconstructs values as q * scale + zero.
    """
    if kv.dim() != 4:
        raise ValueError(f"Expected kv with 4 dims, got shape {tuple(kv.shape)}")
    bs, num_heads, seq_len, head_dim = kv.shape
    if head_dim % 2 != 0:
        raise ValueError(f"INT4 packing requires even head_dim, got {head_dim}")

    max_val = kv.amax(dim=-1, keepdim=True)
    min_val = kv.amin(dim=-1, keepdim=True)
    scales = ((max_val - min_val).clamp(min=1e-6) / 15).to(kv.dtype)
    zeros = min_val.to(kv.dtype)
    q = torch.clamp(torch.round((kv - zeros) / scales), 0, 15).to(torch.uint8)

    low = q[..., 0::2]
    high = q[..., 1::2] << 4
    packed = (low | high).permute(0, 2, 1, 3).reshape(
        bs * seq_len, 1, num_heads, head_dim // 2
    ).contiguous()
    scales = scales.squeeze(-1).permute(0, 2, 1).reshape(
        bs * seq_len, 1, num_heads
    ).contiguous()
    zeros = zeros.squeeze(-1).permute(0, 2, 1).reshape(
        bs * seq_len, 1, num_heads
    ).contiguous()
    return packed, scales, zeros


def dequantize_packed_int4_k_for_gemv(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Reference unpacker for tests. Returns [bs, num_heads, seq_len, head_dim]."""
    packed = packed.reshape(batch_size, seq_len, 1, num_heads, head_dim // 2)
    packed = packed[:, :, 0].permute(0, 2, 1, 3).contiguous()
    values = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=packed.device,
        dtype=torch.uint8,
    )
    values[..., 0::2] = packed & 0x0F
    values[..., 1::2] = (packed >> 4) & 0x0F
    scales = scales.reshape(batch_size, seq_len, 1, num_heads)
    scales = scales[:, :, 0].permute(0, 2, 1).unsqueeze(-1)
    zeros = zeros.reshape(batch_size, seq_len, 1, num_heads)
    zeros = zeros[:, :, 0].permute(0, 2, 1).unsqueeze(-1)
    return values.to(scales.dtype) * scales + zeros

if __name__ == "__main__":
    # kv = torch.Tensor([-4, 5, 6, -7, 8, 9])
    kv = torch.Tensor(
        [
            0.31,
            0.42,
            -0.57,
            0.23,
            -1.79,
            -2.18,
            1.90,
            4.32,
            -3.56,
            0.78,
            0.69,
            1.45,
            2.84,
            -2.09,
            -1.26,
        ]
    )
    print(min_max_per_token_quant_kv(kv, 4))
    print(py_int4_min_max_per_token_quant_kv(kv, 4)[0])
