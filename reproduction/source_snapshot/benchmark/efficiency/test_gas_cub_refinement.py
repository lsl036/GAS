"""CUDA compact refinement and mutable epoch regression tests."""
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

from gas_compact_refinement import allocate, exception_stage, exception_stage_unified
from test_gas_compact_refinement import materialize, pack
from twilight.pyimpl.gqa_refine import build_group_refined_mask

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(scope='module')
def cub():
    return load(name='gas_compact_refine', sources=[str(ROOT / 'csrc/src/gas_compact_refine.cu')],
                extra_cuda_cflags=['-O3'], verbose=False)


@pytest.mark.parametrize('method', ['tiled', 'tagged_tiled', 'tagged_blocksort', 'tagged_shared', 'tagged_hist', 'tagged_triton', 'tagged_dense', 'tagged_fused', 'tagged_warp', 'fused_exception', 'fast_scores', 'fused_mass', 'normalized', 'normalized_shared', 'normalized_fast_scores', 'normalized_prepared'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_graph_updated_inputs(cub, method, dtype):
    if method == 'tagged_hist' and dtype != torch.bfloat16:
        pytest.skip('radix histogram only accepts BF16 scores')
    torch.manual_seed(123)
    batch, length, capacity = 2, 256, 256
    masks = [(torch.rand(batch, 32, length, device='cuda') < 0.28) for _ in range(2)]
    scores_list = [torch.randn(batch, 32, length, device='cuda').to(dtype) for _ in range(2)]
    for m in masks:
        m[:, :, 0] = True
    scores = scores_list[0].clone()
    mask = masks[0].clone()
    ids = torch.full((batch, 32, capacity), -1, dtype=torch.int32, device='cuda')
    counts = torch.empty((batch, 32), dtype=torch.int32, device='cuda')
    out = allocate(batch, capacity, length, mask.device)
    epochs = None
    if method != 'tiled':
        out = (out[0], out[1], torch.zeros((batch, 8, length), dtype=torch.int32, device='cuda'), out[3], out[4])
        epochs = torch.zeros((batch, 8), dtype=torch.int32, device='cuda')
    scratch = torch.empty((batch, 8, 4, capacity), dtype=torch.float32, device='cuda')
    tiles = (capacity + 1023) // 1024
    tmp_score = torch.empty((batch, 32, tiles, 16), dtype=torch.float32, device='cuda')
    tmp_ids = torch.empty((batch, 32, tiles, 16), dtype=torch.int32, device='cuda')
    def update_input(index):
        mask.copy_(masks[index])
        scores.copy_(scores_list[index])
        compact = pack(mask)
        ids.fill_(-1)
        ids[:, :, :compact.shape[-1]].copy_(compact)
        counts.copy_(mask.sum(-1, dtype=torch.int32))
    def run():
        if method == 'fused_exception':
            cub.group_tagged_fused_exception(ids, counts, mask, scores, out[0], out[1], out[2], epochs,
                                             scratch, out[3], out[4], 0.8)
            return
        if method != 'tiled':
            group_fn = (cub.group_tagged_normalized_prepared if method == 'normalized_prepared' else
                        cub.group_tagged_normalized_fast_scores if method == 'normalized_fast_scores' else
                        cub.group_tagged_normalized_shared if method == 'normalized_shared' else
                        cub.group_tagged_normalized if method == 'normalized' else
                        cub.group_tagged_fused_mass if method == 'fused_mass' else
                        cub.group_tagged_fast_scores if method == 'fast_scores' else
                        cub.group_tagged_warp_reduce if method == 'tagged_warp' else
                        cub.group_tagged_fused_reduce if method == 'tagged_fused' else
                        cub.group_tagged_shared if method == 'tagged_shared' else
                        cub.group_tagged_dense if method == 'tagged_dense' else cub.group_tagged)
            group_fn(ids, counts, mask, scores, out[0], out[1], out[2], epochs, scratch, 0.8)
            if method in ('tagged_triton', 'tagged_dense', 'tagged_fused', 'tagged_warp', 'fast_scores', 'fused_mass', 'normalized', 'normalized_shared', 'normalized_fast_scores', 'normalized_prepared'):
                exception_stage(ids, scores, out, counts=counts, epochs=epochs)
            elif method == 'tagged_hist':
                cub.exception_histogram_tagged(ids, counts, scores, out[2], epochs, out[3], out[4])
            elif method in ('tagged_blocksort', 'tagged_shared'):
                cub.exception_blocksort_tagged(ids, counts, scores, out[2], epochs, out[3], out[4])
            else:
                cub.exception_tiled_tagged(ids, counts, scores, out[2], epochs, out[3], out[4], tmp_score, tmp_ids)
        else:
            out[2].zero_()
            cub.group(ids, counts, mask, scores, out[0], out[1], out[2], scratch, 0.8)
            cub.exception_tiled(ids, counts, scores, out[2], out[3], out[4], tmp_score, tmp_ids)
    update_input(0)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for index in (0, 1, 0):
        update_input(index)
        graph.replay()
        actual = materialize((out[0], out[1], out[3], out[4]), length)
        expected = build_group_refined_mask(scores.unsqueeze(2), mask.unsqueeze(2), 8, 0.8, 'group_max', 16)
        assert torch.equal(actual, expected.squeeze(2).cpu())


@pytest.mark.parametrize('case', ['empty', 'one', 'tied_residuals'])
def test_fused_exception_boundaries(cub, case):
    batch, length = 1, 256
    scores = torch.zeros(batch, 32, length, device='cuda', dtype=torch.bfloat16)
    mask = torch.zeros(batch, 32, length, device='cuda', dtype=torch.bool)
    if case == 'one':
        mask[:, :, 7] = True
    elif case == 'tied_residuals':
        mask.fill_(True)
        scores[:, :, 0] = 20
    ids = pack(mask)
    counts = mask.sum(-1, dtype=torch.int32).contiguous()
    out = allocate(batch, ids.shape[-1], length, mask.device)
    flags = torch.zeros((batch, 8, length), dtype=torch.int32, device='cuda')
    epochs = torch.zeros((batch, 8), dtype=torch.int32, device='cuda')
    scratch = torch.empty((batch, 8, 4 * ids.shape[-1]), dtype=torch.float32, device='cuda')
    cub.group_tagged_fused_exception(ids, counts, mask, scores, out[0], out[1], flags, epochs,
                                     scratch, out[3], out[4], 0.8)
    actual = materialize((out[0], out[1], out[3], out[4]), length)
    expected = build_group_refined_mask(scores.unsqueeze(2), mask.unsqueeze(2), 8, 0.8, 'group_max', 16)
    assert torch.equal(actual, expected.squeeze(2).cpu())


@pytest.mark.parametrize('method', ['normalized', 'normalized_shared', 'normalized_fast_scores', 'normalized_prepared'])
@pytest.mark.parametrize('case', ['empty', 'one', 'tied_residuals'])
def test_normalized_boundaries(cub, method, case):
    batch, length = 1, 256
    scores = torch.zeros(batch, 32, length, device='cuda', dtype=torch.bfloat16)
    mask = torch.zeros(batch, 32, length, device='cuda', dtype=torch.bool)
    if case == 'one':
        mask[:, :, 7] = True
    elif case == 'tied_residuals':
        mask.fill_(True)
        scores[:, :, 0] = 20
    ids = pack(mask)
    counts = mask.sum(-1, dtype=torch.int32).contiguous()
    out = allocate(batch, ids.shape[-1], length, mask.device)
    out = (out[0], out[1], torch.zeros((batch, 8, length), dtype=torch.int32, device='cuda'), out[3], out[4])
    epochs = torch.zeros((batch, 8), dtype=torch.int32, device='cuda')
    scratch = torch.empty((batch, 8, 4 * ids.shape[-1]), dtype=torch.float32, device='cuda')
    fn = getattr(cub, f'group_tagged_{method}')
    fn(ids, counts, mask, scores, out[0], out[1], out[2], epochs, scratch, 0.8)
    exception_stage(ids, scores, out, counts=counts, epochs=epochs)
    actual = materialize((out[0], out[1], out[3], out[4]), length)
    expected = build_group_refined_mask(scores.unsqueeze(2), mask.unsqueeze(2), 8, 0.8, 'group_max', 16)
    assert torch.equal(actual, expected.squeeze(2).cpu())


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_unified_exception_dynamic(cub, dtype):
    torch.manual_seed(123)
    batch, length, capacity = 2, 256, 256
    masks = [(torch.rand(batch, 32, length, device='cuda') < rate) for rate in (0.05, 0.7)]
    scores_list = [torch.randn(batch, 32, length, device='cuda').to(dtype) for _ in masks]
    ids = torch.full((batch, 32, capacity), -1, dtype=torch.int32, device='cuda')
    counts = torch.empty((batch, 32), dtype=torch.int32, device='cuda')
    mask, scores = masks[0].clone(), scores_list[0].clone()
    out = allocate(batch, capacity, length, mask.device)
    out = (out[0], out[1], torch.zeros((batch, 8, length), dtype=torch.int32, device='cuda'), out[3], out[4])
    epochs = torch.zeros((batch, 8), dtype=torch.int32, device='cuda')
    scratch = torch.empty((batch, 8, 4 * capacity), dtype=torch.float32, device='cuda')
    def update(index):
        mask.copy_(masks[index])
        scores.copy_(scores_list[index])
        ids.fill_(-1)
        compact = pack(mask)
        ids[:, :, :compact.shape[-1]].copy_(compact)
        counts.copy_(mask.sum(-1, dtype=torch.int32))
    def run():
        cub.group_tagged_normalized_prepared(ids, counts, mask, scores, out[0], out[1], out[2], epochs, scratch, 0.8)
        exception_stage_unified(ids, scores, out, counts, epochs)
    update(0)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for index in (0, 1, 0):
        update(index)
        graph.replay()
        actual = materialize((out[0], out[1], out[3], out[4]), length)
        expected = build_group_refined_mask(scores.unsqueeze(2), mask.unsqueeze(2), 8, 0.8, 'group_max', 16)
        assert torch.equal(actual, expected.squeeze(2).cpu())


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_unified_exception_bucket_boundaries(dtype):
    torch.manual_seed(11)
    batch, length, capacity = 1, 9216, 8192
    edges = [0, 1, 16, 17, 127, 128, 129, 511, 512, 513,
             2047, 2048, 2049, 8191, 8192]
    indices = torch.arange(capacity, dtype=torch.int32, device='cuda').view(1, 1, -1).expand(batch, 32, -1).contiguous()
    counts = torch.tensor((edges * 3)[:32], dtype=torch.int32, device='cuda').view(1, 32)
    scores = torch.randn(batch, 32, length, device='cuda').to(dtype)
    flags = (torch.rand(batch, 8, length, device='cuda') > 0.6).to(torch.int32)
    epochs = torch.ones((batch, 8), dtype=torch.int32, device='cuda')
    a = allocate(batch, capacity, length, scores.device)
    b = allocate(batch, capacity, length, scores.device)
    a = (a[0], a[1], flags, a[3], a[4])
    b = (b[0], b[1], flags, b[3], b[4])
    def run_unified():
        exception_stage_unified(indices, scores, a, counts, epochs)
    def compare():
        exception_stage(indices, scores, b, counts=counts, epochs=epochs)
        torch.cuda.synchronize()
        assert torch.equal(a[4], b[4])
        for head, count in enumerate(a[4][0].tolist()):
            assert set(a[3][0, head, :count].tolist()) == set(b[3][0, head, :count].tolist())
    run_unified()
    compare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_unified()
    counts.copy_(counts.flip(-1))
    scores.copy_(torch.randn_like(scores))
    graph.replay()
    compare()
