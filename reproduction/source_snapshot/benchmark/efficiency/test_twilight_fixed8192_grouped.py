from pathlib import Path

import torch

from bench_twilight_fixed8192_grouped import make_masks
from bench_twilight_upstream_six import author_budgets


def test_synthetic_support_budget_and_group_matching():
    upstream = Path(__file__).resolve().parents[3] / 'Twilight_Upstream_20261113'
    budgets = author_budgets(upstream / 'benchmark/efficiency/bench_breakdown.py', 256)
    masks, shared = make_masks(16384, budgets, 16426)
    quest, twi, gas, less = (masks[key] for key in ('quest', 'twilight', 'gas', 'less_style'))
    union = twi.view(8, 4, -1).any(1)
    assert torch.all(quest.sum(-1) == 8192)
    assert twi.sum(-1).tolist() == budgets
    assert not (twi & ~quest).any()
    assert not (gas & ~union.repeat_interleave(4, 0)).any()
    assert not (shared & ~union).any()
    assert (gas & ~shared.repeat_interleave(4, 0)).sum(-1).max() <= 16
    assert torch.equal(less.view(8, 4, -1).any(1).sum(-1), union.sum(-1))
    assert torch.equal(less.view(8, 4, -1), less.view(8, 4, -1)[:, :1].expand(-1, 4, -1))
