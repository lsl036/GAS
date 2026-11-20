"""Correctness for grouped attention on the shared token-major pool."""
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

from bench_e6_qasper_trace import reference

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(scope='module', params=['gas_group_warp_shared_pool', 'gas_cooperative_shared_pool'])
def module(request):
    name = request.param
    return name, load(name=name, sources=[str(ROOT / f'csrc/src/{name}.cu')],
                      extra_cuda_cflags=['-O3'], verbose=False)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('segments,parts,chunk,tiled', [(4, 1, 64, False), (8, 1, 64, True), (4, 8, 2, False), (4, 16, 2, False), (8, 8, 2, True)])
def test_varied_supports_and_graph(module, dtype, segments, parts, chunk, tiled):
    name, module = module
    if name == 'gas_cooperative_shared_pool' and (parts == 1 or tiled):
        pytest.skip('cooperative kernel does not implement this variant')
    torch.manual_seed(7)
    batch, groups, length = 2, 2, 37
    q = torch.randn(batch, groups * 4, 128, device='cuda', dtype=dtype)
    k = torch.randn(length, groups, 128, device='cuda', dtype=dtype)
    v = torch.randn_like(k)
    shared = torch.full((batch, groups, 12), -1, device='cuda', dtype=torch.int32)
    ex = torch.full((batch, groups * 4, 4), -1, device='cuda', dtype=torch.int32)
    sc = torch.zeros(batch, groups, device='cuda', dtype=torch.int32)
    ec = torch.zeros(batch, groups * 4, device='cuda', dtype=torch.int32)
    mask = torch.zeros(batch, groups * 4, length, device='cuda', dtype=torch.bool)
    for b in range(batch):
        for g in range(groups):
            # Include an empty group support, differing counts across batch,
            # and heads with no exceptions.
            common = [] if (b, g) == (1, 0) else list(range(1, 3 + b + g))
            shared[b, g, :len(common)] = torch.tensor(common, dtype=torch.int32, device='cuda')
            sc[b, g] = len(common)
            for h in range(4 * g, 4 * g + 4):
                extra = [] if common and h % 2 == 0 else list(range(25, 26 + h % 3))
                ex[b, h, :len(extra)] = torch.tensor(extra, dtype=torch.int32, device='cuda')
                ec[b, h] = len(extra)
                mask[b, h, common + extra] = True
    dense_k = k.transpose(0, 1).unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    dense_v = v.transpose(0, 1).unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    expected = reference(q, dense_k, dense_v, mask)
    call = module.decode_tiled if tiled else module.decode
    actual = call(q, k, v, shared, ex, sc, ec, segments, parts, chunk)
    torch.testing.assert_close(actual.float(), expected, atol=.005, rtol=.005)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call(q, k, v, shared, ex, sc, ec, segments, parts, chunk)
    out.fill_(float('nan'))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, atol=.005, rtol=.005)
