#!/usr/bin/env python3
"""Replay Qasper traces using E6 kernels, with an explicitly labelled union proxy."""
import argparse
import csv
import json
import statistics
from pathlib import Path

import torch

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics, padded_tensor
from bench_e6_kernel_baselines import get_gas_module
from bench_e6_qasper_official_b1 import less_mask
from bench_gas_mixed_trace import chunked_reference, make_batch


def lists(mask):
    return [torch.nonzero(row, as_tuple=True)[0].tolist() for row in mask]


def indices(source_lists, batch, heads, device):
    return padded_tensor([source_lists[b % len(source_lists)][h]
                          for b in range(batch) for h in range(heads)],
                         (batch, heads), device)


def compare(output, expected):
    return error_metrics(output.float(), expected.float(), 0.02, 0.02)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace-dirs', type=Path, nargs=2, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--layers', type=int, nargs='+', default=list(range(2, 32)))
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--less-k', type=int, default=1080)
    p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--graph-iterations', type=int, default=8)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'config.json').write_text(json.dumps({**vars(args),
        'trace_dirs': [str(x.resolve()) for x in args.trace_dirs],
        'output_dir': str(args.output_dir.resolve()),
        'semantics': 'Twilight uses group union for every head, not original per-head mask; selection excluded',
        'less': 'E6 shared kernel, mask reconstructed from layers 2 and 13; dense FI for search layers',
    }, indent=2) + '\n')
    module = get_gas_module()
    device = torch.device('cuda')
    # Match the historical 16K/32-head/8-KV-head setup.
    less_lists = {}
    for search_layer in (2, 13):
        less_lists[search_layer] = []
        for directory in args.trace_dirs:
            trace = torch.load(directory / f'step001_layer{search_layer:02d}.pt', map_location='cpu', weights_only=True)
            trace = {key: value.to(device=device, dtype=torch.float16) if key in ('q', 'k', 'v') else value
                     for key, value in trace.items()}
            mask = less_mask(trace, trace['k'].shape[2], args.less_k)
            less_lists[search_layer].append(torch.nonzero(mask[0, 0], as_tuple=True)[0].cpu().tolist())
            del trace, mask
    rows = []
    for layer in args.layers:
        traces = [torch.load(d / f'step001_layer{layer:02d}.pt', map_location='cpu', weights_only=True)
                  for d in args.trace_dirs]
        for trace in traces:
            for key in ('q', 'k', 'v'):
                trace[key] = trace[key].to(torch.float16)
        q, k, v, masks, shared, ex, lengths = make_batch(traces, args.batch_size, device)
        candidates = [lists(t['candidate'][0].bool()) for t in traces]
        unions = [lists(t['twilight'][0].bool().view(8, 4, -1).any(1)) for t in traces]
        support = {'quest': masks['candidate'], 'gas': masks['gas']}
        support['twilight_union'] = masks['twilight'].view(args.batch_size, 8, 4, -1).any(2).repeat_interleave(4, 1)
        candidate_indices = indices(candidates, args.batch_size, 32, device)
        union_indices = indices(unions, args.batch_size, 8, device)
        less_idx = indices([[values for _ in range(8)] for values in less_lists[2 if layer < 13 else 13]],
                           args.batch_size, 8, device)
        less_support = torch.zeros_like(masks['gas'])
        for b in range(args.batch_size):
            less_support[b, :, less_lists[2 if layer < 13 else 13][b % 2]] = True
        support['less'] = less_support
        per_head_union = None
        # E6's group kernel is capped at 96 KiB; preserve all tokens on fallback.
        if 4 * union_indices.shape[-1] * 4 + 1024 > 96 * 1024:
            per_head_union = indices([[group for group in source for _ in range(4)] for source in unions],
                                     args.batch_size, 32, device)
        per_head_gas = None
        if 4 * (shared.shape[-1] + ex.shape[-1]) * 4 + 1024 > 96 * 1024:
            per_head_gas = indices([lists(t['gas'][0].bool()) for t in traces], args.batch_size, 32, device)
        calls = {
            'quest': (lambda: module.batched_per_head_sparse_decode(q, k, v, candidate_indices), 'per_head'),
            'twilight_union': ((lambda: module.batched_per_head_sparse_decode(q, k, v, per_head_union)) if per_head_union is not None
                               else (lambda: module.batched_gqa_group_shared_decode(q, k, v, union_indices)),
                               'per_head_fallback' if per_head_union is not None else 'group_shared'),
            'gas': ((lambda: module.batched_per_head_sparse_decode(q, k, v, per_head_gas)) if per_head_gas is not None
                    else (lambda: module.batched_gqa_group_exception_decode(q, k, v, shared, ex)),
                    'per_head_fallback' if per_head_gas is not None else 'group_exception'),
            'less': (lambda: module.batched_gqa_group_shared_decode(q, k, v, less_idx), 'group_shared'),
        }
        for method, (call, backend) in calls.items():
            ref = chunked_reference(q[:2], k[:2], v[:2], support[method][:2], 1)
            output = call()
            check = compare(output[:2], ref)
            if not check['pass']:
                raise ValueError(f'{layer} {method} failed reference: {check}')
            original_error = {}
            if method == 'twilight_union':
                original = chunked_reference(q[:2], k[:2], v[:2], masks['twilight'][:2], 1)
                original_error = compare(output[:2], original)
                del original
            for _ in range(args.warmups):
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.graph_iterations):
                    graph_output = call()
            graph_output.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            graph_check = compare(graph_output[:2], ref)
            if not graph_check['pass']:
                raise ValueError(f'{layer} {method} graph failed: {graph_check}')
            samples = [t / args.graph_iterations for _ in range(args.rounds)
                       for t in cuda_event_samples(graph.replay, device, args.warmups, args.repeats)]
            rows.append(dict(layer=layer, method=method, backend=backend,
                             median_us=statistics.median(samples), samples_us=json.dumps(samples),
                             mean_selected=float(support[method].sum(-1).float().mean()),
                             union_mean=float(support[method].view(args.batch_size, 8, 4, -1).any(2).sum(-1).float().mean()),
                             check=json.dumps(check), original_twilight_error=json.dumps(original_error)))
            del graph, graph_output, output, ref
        with (args.output_dir / 'layers.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'layer {layer}: ' + ', '.join(f"{r['method']}={r['median_us']:.2f} us ({r['backend']})" for r in rows[-4:]), flush=True)
        del q, k, v, masks, support, shared, ex, calls
        torch.cuda.empty_cache()
    print(f'COMPLETE {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
