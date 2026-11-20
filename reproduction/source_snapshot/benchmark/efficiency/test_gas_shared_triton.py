"""Experimental Triton attention against exact per-head references."""
import pytest
import torch

from bench_e6_qasper_trace import reference
from gas_shared_triton import decode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('parts,block', [(16, 64), (16, 128), (32, 64)])
def test_batch_masks_empty_exceptions_and_graph(parts, block):
    torch.manual_seed(81)
    batch, groups, length = 2, 2, 37
    q = torch.randn(batch, groups * 4, 128, device='cuda', dtype=torch.float16)
    k = torch.randn(length, groups, 128, device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    shared = torch.full((batch, groups, 12), -1, dtype=torch.int32, device='cuda')
    ex = torch.full((batch, groups * 4, 16), -1, dtype=torch.int32, device='cuda')
    sc = torch.zeros(batch, groups, dtype=torch.int32, device='cuda')
    ec = torch.zeros(batch, groups * 4, dtype=torch.int32, device='cuda')
    mask = torch.zeros(batch, groups * 4, length, dtype=torch.bool, device='cuda')
    for b in range(batch):
        for g in range(groups):
            common = [] if (b, g) == (1, 0) else list(range(1, 5 + b + g))
            shared[b, g, :len(common)] = torch.tensor(common, device='cuda', dtype=torch.int32)
            sc[b, g] = len(common)
            for h in range(g * 4, g * 4 + 4):
                extra = [] if common and h % 2 == 0 else list(range(25, 26 + h % 3))
                ex[b, h, :len(extra)] = torch.tensor(extra, device='cuda', dtype=torch.int32)
                ec[b, h] = len(extra)
                mask[b, h, common + extra] = True
    dense_k = k.transpose(0, 1).unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    dense_v = v.transpose(0, 1).unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    expected = reference(q, dense_k, dense_v, mask)
    out = decode(q, k, v, shared, ex, sc, ec, parts, block)
    torch.testing.assert_close(out.float(), expected, atol=.005, rtol=.005)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = decode(q, k, v, shared, ex, sc, ec, parts, block)
    output.fill_(float('nan'))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, atol=.005, rtol=.005)
