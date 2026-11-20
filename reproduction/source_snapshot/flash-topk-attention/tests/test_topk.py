import pytest
from itertools import product

import torch
import torch.nn as nn
import math

from ftka.cuda_ops.topk import raft_topk
from ftka.utils.benchmark import benchmark_forward_with_warmup

# This file is used for testing topk kernel from libRAFT
# We do not seriously compare the topk indices since the random value leads to similar tensor.
# Detailed evaluation is in quest/tests/topk/*
# Therefore instead, we pay attention to the qk result (collected attention scores).

def bench_topk(batch_size, seq_len, topk):
    dtype = torch.float16
    device = torch.device("cuda:0")

    qo_len = 1
    kv_len = seq_len
    head_dim = 128
    
    # HND: [batch_size, seq_len, head_dim]
    q = torch.randn(batch_size, qo_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, kv_len, head_dim, dtype=dtype, device=device)

    # [batch_size, qo_len, kv_len]
    attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
    torch_topk_value, torch_topk_indices = torch.topk(attn_weights, k=topk, dim=-1)

    # sum on dim=-1
    torch_topk_value = torch_topk_value.sum(dim=-1).squeeze()

    cuda_input_data = attn_weights.squeeze()
    cuda_input_indices = torch.arange(0, kv_len, dtype=torch.int32, device=device).repeat(batch_size, 1)
    cuda_output_data = torch.randn(batch_size, topk, dtype=dtype, device=device)
    cuda_output_indices = torch.arange(0, topk, dtype=torch.int32, device=device).repeat(batch_size, 1)
    topk_buf = torch.zeros((batch_size, 8192 * 2 * (2+4) // 2 // 48), dtype=dtype, device=device)

    _, avg_time = benchmark_forward_with_warmup(
        torch.topk,
        attn_weights,
        topk,
        -1,
        warmups=10,
        repeats=100,
    )

    print(f"[torch] batch_size: {batch_size}, seq_len: {seq_len}, topk: {topk}: {avg_time.mean*1e3:.3f} ms")

    _, avg_time = benchmark_forward_with_warmup(
        raft_topk,
        cuda_input_data,
        cuda_input_indices,
        cuda_output_data,
        cuda_output_indices,
        topk_buf,
        topk,
        warmups=10,
        repeats=100,
    )

    cuda_output_data = cuda_output_data.sum(dim=-1).squeeze()
    # print(torch_topk_value)
    # print(cuda_output_data)
    torch.testing.assert_close(cuda_output_data, torch_topk_value, rtol=1e-2, atol=1e-2)

    print(f"[raft] batch_size: {batch_size}, seq_len: {seq_len}, topk: {topk}: {avg_time.mean*1e3:.3f} ms")



if __name__ == "__main__":
    # bench_topk(32, 16384, 2048)
    # bench_topk(64, 16384, 2048)
    # bench_topk(64, 32768, 2048)
    # bench_topk(512, 32768, 2048)
    bench_topk(512, 2048, 512)
    bench_topk(1024, 2048, 512)
    bench_topk(2048, 2048, 512)