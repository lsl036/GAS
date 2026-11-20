"""Verify grouped FI custom-mask replay with different masks per request."""
import pytest
import torch

from bench_twilight_group_fi import grouped_mask_fi
from bench_twilight_upstream_six import fi_sparse
from gas_grouped_static_fi import StaticGroupedAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_distinct_batch_masks_and_graph():
    torch.manual_seed(91)
    batch, length = 3, 37
    q = torch.randn(batch, 32, 128, device='cuda', dtype=torch.float16)
    k = torch.randn(length, 8, 128, device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    mask = torch.rand(batch, 32, length) > .7
    mask[:, :, 0] = True
    flat, flat_wrapper, flat_pages = fi_sparse(q, k, v, mask)
    grouped, grouped_wrapper, grouped_pages = grouped_mask_fi(q, k, v, mask)
    grouped_gpu, grouped_gpu_wrapper, grouped_gpu_pages = grouped_mask_fi(q, k, v, mask.to('cuda'), True)
    expected = flat()
    torch.testing.assert_close(grouped().float(), expected.float(), atol=.003, rtol=.003)
    torch.testing.assert_close(grouped_gpu().float(), expected.float(), atol=.003, rtol=.003)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = grouped()
    output.fill_(float('nan'))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected.float(), atol=.003, rtol=.003)


def test_static_plan_updates_and_overflow():
    torch.manual_seed(27)
    batch, length = 2, 37
    q = torch.randn(batch, 32, 128, device='cuda', dtype=torch.float16)
    k = torch.randn(length, 8, 128, device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    mask = torch.rand(batch, 32, length, device='cuda') > .8
    mask[:, :, 0] = True
    static = StaticGroupedAttention(q, k, v, mask)
    for current in (mask, mask.roll(3, dims=-1), mask):
        flat, wrapper, pages = fi_sparse(q, k, v, current)
        static.update(current)
        torch.testing.assert_close(static.run(q).float(), flat().float(), atol=.003, rtol=.003)
        packed = static.packed.clone()
        static.update_triton(current.contiguous())
        assert torch.equal(static.packed, packed)
        torch.testing.assert_close(static.run(q).float(), flat().float(), atol=.003, rtol=.003)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = static.run(q)
    output.fill_(float('nan'))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), flat().float(), atol=.003, rtol=.003)
    small = torch.zeros_like(mask)
    small[:, :, 0] = True
    constrained = StaticGroupedAttention(q, k, v, small, quantum=2)
    with pytest.raises(ValueError, match='exceeds fixed plan capacity'):
        constrained.update(mask)
    with pytest.raises(ValueError, match='exceeds fixed plan capacity'):
        constrained.update_triton(mask)


def test_triton_compaction_crosses_tile_and_tail():
    torch.manual_seed(111)
    batch, length = 2, 1027
    q = torch.randn(batch, 32, 128, device='cuda', dtype=torch.float16)
    k = torch.randn(length, 8, 128, device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    mask = torch.rand(batch, 32, length, device='cuda') > .65
    mask[:, :, 0] = True
    plan = StaticGroupedAttention(q, k, v, mask)
    plan.update(mask)
    expected_bits = plan.packed.clone()
    expected = plan.run(q).clone()
    plan.update_triton(mask)
    assert torch.equal(plan.packed, expected_bits)
    torch.testing.assert_close(plan.run(q).float(), expected.float(), atol=.003, rtol=.003)
