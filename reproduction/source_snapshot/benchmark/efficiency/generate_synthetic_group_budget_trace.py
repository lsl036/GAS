#!/usr/bin/env python3
"""Generate reproducible nested sparse masks from synthetic estimator logits."""
import argparse
import json
from pathlib import Path

import torch

from twilight.pyimpl.gqa_refine import build_group_refined_mask
from twilight.pyimpl.top_p import top_p_unnormalized


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--length', type=int, default=32768)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--head-offset-step', type=float, default=8.0)
    p.add_argument('--sink-gap', type=float, default=6.0)
    p.add_argument('--first-head-sink-gap', type=float,
                   help='Override the sink logit gap for the first head in each group.')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.length < 32765:
        p.error('disjoint 8192-token supports require length >=32765')
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    length, budget = args.length, 8192
    candidate = torch.zeros((1, 32, length), dtype=torch.bool, device=device)
    scores = torch.full((1, 32, length), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16, device=device)
    for group in range(8):
        permutation = torch.randperm(length - 1, device=device) + 1
        for head in range(4):
            row = group * 4 + head
            tokens = permutation[head * (budget - 1):(head + 1) * (budget - 1)]
            candidate[0, row, 0] = True
            candidate[0, row, tokens] = True
            logits = torch.randn(tokens.numel(), device=device) * 0.5 - head * args.head_offset_step
            scores[0, row, tokens] = logits.to(scores.dtype)
            sink_gap = args.first_head_sink_gap if head == 0 and args.first_head_sink_gap is not None else args.sink_gap
            scores[0, row, 0] = sink_gap - head * args.head_offset_step
    twilight = top_p_unnormalized(scores.unsqueeze(2), .85).squeeze(2) & candidate
    gas = build_group_refined_mask(scores.unsqueeze(2), twilight.unsqueeze(2), 8, .8, 'group_max', 16).squeeze(2)
    if not torch.all(twilight <= candidate):
        raise ValueError('Twilight mask exceeds Quest candidates')
    for group in range(8):
        union = twilight[0, group * 4:group * 4 + 4].any(0)
        if (gas[0, group * 4:group * 4 + 4] & ~union).any():
            raise ValueError('GAS mask exceeds the Twilight group union')
    q = torch.randn((1, 32, 128), dtype=torch.bfloat16, device='cpu')
    k = torch.randn((1, 8, length, 128), dtype=torch.bfloat16, device='cpu')
    v = torch.randn_like(k)
    trace = {'q': q, 'k': k, 'v': v, 'candidate': candidate.cpu(),
             'twilight': twilight.cpu(), 'gas': gas.cpu(), 'scores': scores.cpu()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, args.output)
    for name in ('candidate', 'twilight', 'gas'):
        mask = trace[name].reshape(8, 4, length)
        print(name, 'head_mean=', float(mask.sum(-1).float().mean()),
              'union_mean=', float(mask.any(1).sum(-1).float().mean()), flush=True)
    print(json.dumps({'path': str(args.output), 'seed': args.seed,
                      'head_offset_step': args.head_offset_step, 'sink_gap': args.sink_gap,
                      'first_head_sink_gap': args.first_head_sink_gap,
                      'warning': 'Synthetic independent logits and Q/K, not a model trace'}), flush=True)


if __name__ == '__main__':
    main()
