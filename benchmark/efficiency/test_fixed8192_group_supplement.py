from pathlib import Path

import pytest
import torch

from bench_fixed8192_group_supplement import make_supplement_masks
from bench_twilight_fixed8192_grouped import make_masks
from bench_twilight_upstream_six import author_budgets


@pytest.mark.parametrize('length', [16384, 32768])
def test_supplement_budget_nesting_and_original_masks(length):
    upstream = Path(__file__).resolve().parents[3] / 'Twilight_Upstream_20261113'
    budgets = author_budgets(upstream / 'benchmark/efficiency/bench_breakdown.py', 256)
    masks, original = make_supplement_masks(length, budgets, 42 + length)
    previous, _ = make_masks(length, budgets, 42 + length)
    quest, less = masks['quest_group12514'], masks['less_group8192']
    assert torch.all(quest.sum(-1) == 8192)
    assert torch.all(quest.view(8, 4, length).any(1).sum(-1) == 12514)
    assert torch.all(less.sum(-1) == 8192)
    assert torch.all(less.view(8, 4, length).any(1).sum(-1) == 8192)
    assert torch.equal(less.view(8, 4, length), less.view(8, 4, length)[:, :1].expand(-1, 4, -1))
    assert not (original['twilight'] & ~quest).any()
    assert torch.equal(previous['twilight'], original['twilight'])
    assert torch.equal(previous['gas'], original['gas'])
