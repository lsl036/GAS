import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "flash-topk-attention"))
sys.path.insert(0, str(ROOT))

from ftka.cuda_ops.gemv import batched_sparse_gemv_int4_k


def main():
    device = "cuda"
    batch_size = 2
    num_heads = 4
    seq_len = 64
    head_dim = 128
    value = 3

    q = torch.ones(batch_size, num_heads, head_dim, device=device, dtype=torch.float16)
    packed_byte = value | (value << 4)
    k_cache = torch.full(
        (seq_len, 1, num_heads, head_dim // 2),
        packed_byte,
        dtype=torch.uint8,
        device=device,
    )
    scales = torch.ones(seq_len, 1, num_heads, dtype=torch.float16, device=device)
    zeros = torch.zeros(seq_len, 1, num_heads, dtype=torch.float16, device=device)
    out = torch.empty(batch_size, num_heads, seq_len, dtype=torch.float16, device=device)
    kv_page_indices = torch.arange(seq_len, dtype=torch.int32, device=device).repeat(batch_size)
    kv_page_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    batched_sparse_gemv_int4_k(
        q,
        out,
        k_cache,
        scales,
        zeros,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
    )
    torch.cuda.synchronize()

    expected = torch.full_like(out, value * head_dim)
    print(f"finite_ratio={torch.isfinite(out).float().mean().item():.4f}")
    print(f"out_min={out.nan_to_num().min().item():.6f}, out_max={out.nan_to_num().max().item():.6f}")
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
    print("constant int4 kernel test passed")


if __name__ == "__main__":
    main()
