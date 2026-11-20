"""Reference checks for compact-candidate head top-p."""
import math
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

from gas_compact_refinement import allocate, exception_stage
from sparse_head_top_p import sparse_head_top_p
from test_gas_compact_refinement import materialize, pack
from gas_attention.pyimpl.gqa_refine import build_group_refined_mask
from gas_attention.pyimpl.quantize import min_max_per_token_quant_kv
from gas_attention.pyimpl.top_p import top_p_unnormalized

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
ROOT = Path(__file__).resolve().parents[2]


def check(scores, candidate, double_sum=False, adaptive_sum=False):
    ids = pack(candidate)
    counts = candidate.sum(-1, dtype=torch.int32).contiguous()
    output_mask = torch.empty_like(candidate)
    output_ids = torch.empty_like(ids)
    output_counts = torch.empty_like(counts)
    ambiguous = torch.empty_like(counts) if adaptive_sum else None
    sparse_head_top_p(ids, counts, scores, output_mask, output_ids, output_counts, double_sum=double_sum,
                      adaptive_sum=adaptive_sum, ambiguous_out=ambiguous)
    masked = scores.masked_fill(~candidate, torch.finfo(scores.dtype).min)
    expected = top_p_unnormalized(masked.unsqueeze(2), .85).squeeze(2) & candidate
    assert torch.equal(output_mask, expected)
    assert torch.equal(output_counts, output_mask.sum(-1, dtype=torch.int32))
    for batch in range(scores.shape[0]):
        for head in range(scores.shape[1]):
            n = int(output_counts[batch, head])
            actual_ids = output_ids[batch, head, :n].long().sort().values
            assert torch.equal(actual_ids, output_mask[batch, head].nonzero().flatten())
    return ambiguous


@pytest.mark.parametrize('case', ['random', 'sink', 'no_sink', 'empty', 'equal', 'very_negative'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_synthetic(case, dtype):
    torch.manual_seed(43)
    scores = torch.randn(2, 32, 512, device='cuda', dtype=dtype).contiguous()
    candidate = (torch.rand(2, 32, 512, device='cuda') < .3).contiguous()
    if case == 'sink':
        candidate[:, :, 0] = True
        scores[:, :, 0] = 5
    elif case == 'no_sink':
        candidate[:, :, 0] = False
    elif case == 'empty':
        candidate.zero_()
    elif case == 'equal':
        scores.zero_()
        candidate.fill_(True)
    elif case == 'very_negative':
        scores.fill_(-1.0e31)
        candidate[:, :, 0] = True
    check(scores, candidate)


@pytest.mark.parametrize('scenario,layer', [('fixed8192', 2), ('rate08', 8)])
@pytest.mark.parametrize('sample', [15, 54])
def test_captured(scenario, layer, sample):
    path = ROOT / 'benchmark/results_efficiency/fig6_traces' / f'qasper16k_{scenario}_s{sample}' / f'step001_layer{layer:02d}.pt'
    trace = torch.load(path, map_location='cpu', weights_only=True)
    q, k = trace['q'].cuda(), trace['k'].cuda().repeat_interleave(4, dim=1)
    scores = (torch.matmul(q.unsqueeze(2), min_max_per_token_quant_kv(k, 4).transpose(-1, -2))
              / math.sqrt(q.shape[-1])).squeeze(2).contiguous()
    check(scores, trace['candidate'].cuda().bool().contiguous())


@pytest.mark.xfail(strict=True, reason='known near-threshold FP32 reduction mismatch; not an exact replacement')
def test_captured_32k_threshold_boundary():
    path = ROOT / 'benchmark/results_efficiency/fig6_traces/qasper32k_repeated_rate08_s54/step001_layer24.pt'
    trace = torch.load(path, map_location='cpu', weights_only=True)
    q, k = trace['q'].cuda(), trace['k'].cuda().repeat_interleave(4, dim=1)
    scores = (torch.matmul(q.unsqueeze(2), min_max_per_token_quant_kv(k, 4).transpose(-1, -2))
              / math.sqrt(q.shape[-1])).squeeze(2).contiguous()
    check(scores, trace['candidate'].cuda().bool().contiguous())


def test_double_sum_captured_32k_threshold_boundary():
    path = ROOT / 'benchmark/results_efficiency/fig6_traces/qasper32k_repeated_rate08_s54/step001_layer24.pt'
    trace = torch.load(path, map_location='cpu', weights_only=True)
    q, k = trace['q'].cuda(), trace['k'].cuda().repeat_interleave(4, dim=1)
    scores = (torch.matmul(q.unsqueeze(2), min_max_per_token_quant_kv(k, 4).transpose(-1, -2))
              / math.sqrt(q.shape[-1])).squeeze(2).contiguous()
    check(scores, trace['candidate'].cuda().bool().contiguous(), double_sum=True)


def test_adaptive_sum_captured_32k_threshold_boundary():
    path = ROOT / 'benchmark/results_efficiency/fig6_traces/qasper32k_repeated_rate08_s54/step001_layer24.pt'
    trace = torch.load(path, map_location='cpu', weights_only=True)
    q, k = trace['q'].cuda(), trace['k'].cuda().repeat_interleave(4, dim=1)
    scores = (torch.matmul(q.unsqueeze(2), min_max_per_token_quant_kv(k, 4).transpose(-1, -2))
              / math.sqrt(q.shape[-1])).squeeze(2).contiguous()
    ambiguous = check(scores, trace['candidate'].cuda().bool().contiguous(), adaptive_sum=True)
    assert int(ambiguous[0, 14]) == 1


def test_graph_replay_changes_inputs():
    torch.manual_seed(77)
    batch, heads, length = 2, 32, 512
    masks = [(torch.rand(batch, heads, length, device='cuda') < p).contiguous() for p in (.8, .08, 0)]
    scores_list = [torch.randn(batch, heads, length, device='cuda', dtype=torch.bfloat16) for _ in masks]
    ids = torch.empty((batch, heads, length), dtype=torch.int32, device='cuda')
    counts = torch.empty((batch, heads), dtype=torch.int32, device='cuda')
    scores = torch.empty_like(scores_list[0])
    output_mask = torch.empty_like(masks[0])
    output_ids = torch.empty_like(ids)
    output_counts = torch.empty_like(counts)

    def update(index):
        compact = pack(masks[index])
        ids.fill_(37)  # Deliberately invalid padding: counts must bound all reads.
        ids[:, :, :compact.shape[-1]].copy_(compact)
        counts.copy_(masks[index].sum(-1, dtype=torch.int32))
        scores.copy_(scores_list[index])

    def run():
        return sparse_head_top_p(ids, counts, scores, output_mask, output_ids, output_counts)

    update(0)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for index in (0, 1, 2, 0):
        update(index)
        graph.replay()
        masked = scores.masked_fill(~masks[index], torch.finfo(scores.dtype).min)
        expected = top_p_unnormalized(masked.unsqueeze(2), .85).squeeze(2) & masks[index]
        assert torch.equal(output_mask, expected)
        assert torch.equal(output_counts, output_mask.sum(-1, dtype=torch.int32))
        for b in range(batch):
            for h in range(heads):
                n = int(output_counts[b, h])
                assert torch.equal(output_ids[b, h, :n].long().sort().values,
                                   output_mask[b, h].nonzero().flatten())


@pytest.mark.parametrize('adaptive_sum', [False, True])
@pytest.mark.parametrize('tagged', [False, True])
def test_graph_pipeline_changes_inputs(adaptive_sum, tagged):
    torch.manual_seed(79)
    batch, heads, length = 1, 32, 512
    masks = [(torch.rand(batch, heads, length, device='cuda') < p).contiguous() for p in (.75, .15, 0)]
    scores_list = [torch.randn(batch, heads, length, device='cuda', dtype=torch.bfloat16) for _ in masks]
    for mask, values in zip(masks[:2], scores_list[:2]):
        mask[:, :, 0] = True
        values[:, :, 0] = 5
    ids = torch.empty((batch, heads, length), dtype=torch.int32, device='cuda')
    counts = torch.empty((batch, heads), dtype=torch.int32, device='cuda')
    scores = torch.empty_like(scores_list[0])
    head_mask = (torch.zeros_like(masks[0], dtype=torch.int32) if tagged else torch.empty_like(masks[0]))
    head_epochs = torch.zeros_like(counts) if tagged else None
    head_ids = torch.empty_like(ids)
    head_counts = torch.empty_like(counts)
    ambiguous = torch.empty_like(counts) if adaptive_sum else None
    out = allocate(batch, length, length, scores.device)
    out = (out[0], out[1], torch.zeros((batch, 8, length), dtype=torch.int32, device='cuda'), out[3], out[4])
    epochs = torch.zeros((batch, 8), dtype=torch.int32, device='cuda')
    scratch = torch.empty((batch, 8, 4 * length), dtype=torch.float32, device='cuda')
    cub = load(name='gas_compact_refine', sources=[str(ROOT / 'csrc/src/gas_compact_refine.cu')],
               extra_cuda_cflags=['-O3'], verbose=False)

    def update(index):
        compact = pack(masks[index])
        ids.fill_(37)
        ids[:, :, :compact.shape[-1]].copy_(compact)
        counts.copy_(masks[index].sum(-1, dtype=torch.int32))
        scores.copy_(scores_list[index])

    def run():
        sparse_head_top_p(ids, counts, scores, head_mask, head_ids, head_counts,
                          adaptive_sum=adaptive_sum, ambiguous_out=ambiguous, head_epochs=head_epochs)
        if tagged:
            cub.group_head_tagged(head_ids, head_counts, head_mask, head_epochs, scores, out[0], out[1],
                                  out[2], epochs, scratch, .8)
        else:
            cub.group_tagged_fused_reduce(head_ids, head_counts, head_mask, scores, out[0], out[1],
                                          out[2], epochs, scratch, .8)
        exception_stage(head_ids, scores, out, counts=head_counts, epochs=epochs)

    update(0)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for index in (0, 1, 2, 0):
        update(index)
        graph.replay()
        masked = scores.masked_fill(~masks[index], torch.finfo(scores.dtype).min)
        expected_head = top_p_unnormalized(masked.unsqueeze(2), .85) & masks[index].unsqueeze(2)
        expected = build_group_refined_mask(scores.unsqueeze(2), expected_head, 8, .8, 'group_max', 16)
        actual = materialize((out[0], out[1], out[3], out[4]), length)
        actual_head = head_mask == head_epochs.unsqueeze(-1) if tagged else head_mask
        assert torch.equal(actual_head, expected_head.squeeze(2))
        assert torch.equal(actual, expected.squeeze(2).cpu())
        if adaptive_sum and not bool(masks[index].any()):
            assert int(ambiguous.sum()) == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('mode', ['fp32', 'fp64', 'adaptive'])
def test_randomized_candidates_against_dense(dtype, mode):
    batch, heads, length = 2, 32, 2048
    for seed in range(40):
        torch.manual_seed(seed)
        scores = torch.randn(batch, heads, length, device='cuda', dtype=dtype).contiguous()
        candidate = (torch.rand(batch, heads, length, device='cuda') < (.03 + .07 * (seed % 12))).contiguous()
        if seed % 2 == 0:
            candidate[:, :, 0] = True
            scores[:, :, 0] = 1 + seed / 2
        ids = pack(candidate)
        counts = candidate.sum(-1, dtype=torch.int32).contiguous()
        output_mask = torch.empty_like(candidate)
        output_ids = torch.empty_like(ids)
        output_counts = torch.empty_like(counts)
        ambiguous = torch.empty_like(counts) if mode == 'adaptive' else None
        sparse_head_top_p(ids, counts, scores, output_mask, output_ids, output_counts,
                          double_sum=mode == 'fp64', adaptive_sum=mode == 'adaptive', ambiguous_out=ambiguous)
        masked = scores.masked_fill(~candidate, torch.finfo(dtype).min)
        expected = top_p_unnormalized(masked.unsqueeze(2), .85).squeeze(2) & candidate
        mismatch = (output_mask != expected).nonzero()
        if mismatch.numel():
            b, h, token = mismatch[0].tolist()
            raise AssertionError(
                f'mismatch seed={seed} dtype={dtype} mode={mode} count={mismatch.shape[0]} '
                f'first={(b, h, token)} score={float(scores[b, h, token])} '
                f'compact={bool(output_mask[b, h, token])} dense={bool(expected[b, h, token])} '
                f'compact_count={int(output_mask[b, h].sum())} dense_count={int(expected[b, h].sum())}'
            )
