"""Correctness checks for compact group-max plus per-head exceptions."""
import pytest
import torch
from pathlib import Path
import math

from gas_compact_refinement import allocate, buckets, exception_stage, exception_stage_tiled_sort, group_stage, refine
from gas_attention.pyimpl.gqa_refine import build_group_refined_mask
from gas_attention.pyimpl.quantize import min_max_per_token_quant_kv


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def pack(mask):
    batch, heads, length = mask.shape
    counts = mask.sum(-1)
    capacity = max(16, int(counts.max()))
    indices = torch.full((batch, heads, capacity), -1, device=mask.device, dtype=torch.int32)
    for b in range(batch):
        for h in range(heads):
            ids = mask[b, h].nonzero().flatten()
            indices[b, h, :ids.numel()] = ids.int()
    return indices


def materialize(outputs, length):
    ids, counts, exceptions, exc_counts = [x.cpu() for x in outputs]
    mask = torch.zeros(ids.shape[0], 32, length, dtype=torch.bool)
    for b in range(ids.shape[0]):
        for h in range(32):
            group = h // 4
            mask[b, h, ids[b, group, :counts[b, group]].long()] = True
            mask[b, h, exceptions[b, h, :exc_counts[b, h]].long()] = True
    return mask


@pytest.mark.parametrize('seed', [1, 3, 42])
@pytest.mark.parametrize('case', ['random', 'empty_head', 'one_token'])
@pytest.mark.parametrize('bucketed', [False, True])
def test_against_reference(seed, case, bucketed):
    torch.manual_seed(seed)
    batch, heads, length = 2, 32, 256
    scores = torch.randn(batch, heads, length, device='cuda')
    mask = torch.rand(batch, heads, length, device='cuda') < 0.25
    if case == 'empty_head':
        mask[:, :4] = False
    elif case == 'one_token':
        mask.zero_()
        mask[:, :, 7] = True
    indices = pack(mask)
    output = allocate(batch, indices.shape[-1], length, mask.device)
    counts = mask.sum(-1, dtype=torch.int32).contiguous() if bucketed else None
    actual = materialize(refine(indices, mask, scores, output, counts=counts), length)
    expected = build_group_refined_mask(scores.unsqueeze(2), mask.unsqueeze(2), 8, 0.8, 'group_max', 16)
    assert torch.equal(actual, expected.squeeze(2).cpu())


@pytest.mark.parametrize('scenario,layer', [('rate08', 10), ('fixed8192', 2), ('fixed8192', 15)])
@pytest.mark.parametrize('sample', [15, 54])
@pytest.mark.parametrize('bucketed', [False, True])
def test_captured_logits(scenario, layer, sample, bucketed):
    root = Path(__file__).resolve().parents[1] / 'results_efficiency/fig6_traces'
    path = root / f'qasper16k_{scenario}_s{sample}' / f'step001_layer{layer:02d}.pt'
    trace = torch.load(path, map_location='cpu', weights_only=True)
    q = trace['q'].cuda()
    k = trace['k'].cuda().repeat_interleave(4, dim=1)
    mask = trace['twilight'].cuda().contiguous()
    scores = (torch.matmul(q.unsqueeze(2), min_max_per_token_quant_kv(k, 4).transpose(-1, -2))
              / math.sqrt(q.shape[-1])).squeeze(2).contiguous()
    indices = pack(mask)
    output = allocate(1, indices.shape[-1], mask.shape[-1], mask.device)
    counts = mask.sum(-1, dtype=torch.int32).contiguous() if bucketed else None
    actual = materialize(refine(indices, mask, scores, output, counts=counts), mask.shape[-1])
    expected = build_group_refined_mask(scores.unsqueeze(2), mask.unsqueeze(2), 8, 0.8, 'group_max', 16)
    assert torch.equal(actual, expected.squeeze(2).cpu())


@pytest.mark.parametrize('bucketed', [False, True])
def test_unordered_candidates_with_equal_exception_scores(bucketed):
    batch, length = 1, 512
    scores = torch.zeros(batch, 32, length, dtype=torch.bfloat16, device='cuda')
    scores[:, :, 0] = 20
    mask = torch.ones((batch, 32, length), dtype=torch.bool, device='cuda')
    indices = pack(mask)
    permutation = torch.randperm(length, device='cuda')
    indices = indices[:, :, permutation].contiguous()
    output = allocate(batch, length, length, mask.device)
    counts = mask.sum(-1, dtype=torch.int32).contiguous() if bucketed else None
    actual = materialize(refine(indices, mask, scores, output, counts=counts), length)
    expected = build_group_refined_mask(scores.unsqueeze(2), mask.unsqueeze(2), 8, 0.8, 'group_max', 16)
    assert torch.equal(actual, expected.squeeze(2).cpu())


@pytest.mark.parametrize('case', ['random', 'equal', 'empty', 'one'])
def test_pair_reduce_exception_matches_original(case):
    torch.manual_seed(19)
    batch, length = 2, 512
    scores = torch.randn(batch, 32, length, device='cuda', dtype=torch.bfloat16)
    mask = torch.rand(batch, 32, length, device='cuda') < 0.7
    if case == 'equal':
        scores.zero_()
        scores[:, :, 0] = 20
        mask.fill_(True)
    elif case == 'empty':
        mask.zero_()
    elif case == 'one':
        mask.zero_()
        mask[:, :, 7] = True
    indices = pack(mask)
    indices = indices[:, :, torch.randperm(indices.shape[-1], device='cuda')].contiguous()
    counts = mask.sum(-1, dtype=torch.int32).contiguous()
    # Shuffle valid entries before padding so count-based kernels can scan them.
    for b in range(batch):
        for h in range(32):
            valid = indices[b, h][indices[b, h] >= 0]
            indices[b, h].fill_(-1)
            indices[b, h, :valid.numel()] = valid
    output = allocate(batch, indices.shape[-1], length, mask.device)
    output[2].zero_()
    group_stage(indices, mask, scores, output, counts=counts)
    exception_stage(indices, scores, output, counts=counts)
    original = materialize((output[0], output[1], output[3], output[4]), length)
    exception_stage(indices, scores, output, counts=counts, pair_reduce=True)
    candidate = materialize((output[0], output[1], output[3], output[4]), length)
    assert torch.equal(candidate, original)
    exception_stage(indices, scores, output, counts=counts, sort_reduce=True)
    candidate = materialize((output[0], output[1], output[3], output[4]), length)
    assert torch.equal(candidate, original)
    tiles = (indices.shape[-1] + 127) // 128
    scratch = torch.empty((batch, 32, tiles, 16), device='cuda', dtype=torch.uint64)
    exception_stage_tiled_sort(indices, scores, output, counts, scratch, tile=128)
    candidate = materialize((output[0], output[1], output[3], output[4]), length)
    assert torch.equal(candidate, original)
    counts_cpu = counts.cpu()
    active = [(low, high) for low, high in buckets(indices.shape[-1])
              if bool(((counts_cpu > low) & (counts_cpu <= high)).any())]
    exception_stage(indices, scores, output, counts=counts, active_buckets=active)
    candidate = materialize((output[0], output[1], output[3], output[4]), length)
    assert torch.equal(candidate, original)
