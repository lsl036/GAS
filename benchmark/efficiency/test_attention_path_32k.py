from types import SimpleNamespace

import torch

from bench_attention_path_32k import grouped_quest, less_select, padded_bounds
from bench_e6_qasper_official_b1 import less_mask
from gas_attention.pyimpl.quest import quest_selector


def test_quest_matches_reference_with_partial_last_page():
    torch.manual_seed(42)
    q = torch.randn(2, 32, 16)
    k = torch.randn(2, 8, 37, 16)
    lo, hi = padded_bounds(k)
    actual = grouped_quest(q, lo, hi, length=37, budget=19)
    expected = quest_selector(SimpleNamespace(num_heads=32), q[:, :, None], k.repeat_interleave(4, 1),
                              chunk_size=16, token_budget=19).squeeze(2)
    assert torch.equal(actual, expected)


def test_less_search_matches_reference_for_repeated_requests():
    torch.manual_seed(123)
    q = torch.randn(1, 32, 16)
    k = torch.randn(1, 8, 73, 16)
    expected = less_mask({"q": q, "k": k}, 73, 12)
    actual = less_select(q.repeat(3, 1, 1), k.repeat(3, 1, 1, 1), 12)
    assert torch.equal(actual, expected.expand_as(actual))
