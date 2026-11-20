import csv
import json
import sys
import pytest
import torch
from types import SimpleNamespace

from bench_e6_qasper_trace import gas_indices, reference
from bench_e6_qasper_official_b1 import less_mask, quest_setup, quest_setup_split
from summarize_e6_qasper_trace import main as summarize_main
from bench_e6_qasper_union import build_union_membership


def test_reference_respects_each_heads_mask():
    q = torch.ones(1, 4, 2)
    k = torch.zeros(1, 1, 3, 2)
    v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])
    mask = torch.tensor([[[True, False, False], [False, True, False],
                          [False, False, True], [True, True, False]]])
    expected = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [2.0, 3.0]]])
    torch.testing.assert_close(reference(q, k, v, mask), expected)


def test_gas_indices_preserve_shared_and_head_specific_exceptions():
    shared = torch.zeros(32, 12, dtype=torch.bool)
    final = torch.zeros_like(shared)
    for head in range(32):
        shared[head, head // 4] = True
        final[head] = shared[head]
        final[head, 8 + head % 4] = True
    groups, exceptions = gas_indices(shared, final, 2, torch.device("cpu"))
    assert groups.shape == (2, 8, 1)
    assert exceptions.shape == (2, 32, 1)
    assert groups[0, :, 0].tolist() == list(range(8))
    assert exceptions[0, :, 0].tolist() == [8 + h % 4 for h in range(32)]
    assert torch.equal(groups[0], groups[1])
    assert torch.equal(exceptions[0], exceptions[1])


def test_less_support_is_globally_shared_with_recent_window():
    torch.manual_seed(7)
    search = {"q": torch.randn(1, 32, 8), "k": torch.randn(1, 8, 32, 8)}
    mask = less_mask(search, 32, 8)
    assert mask.shape == (1, 32, 32)
    assert torch.all(mask.sum(-1) == 8)
    assert mask[0, 0, -2:].all()
    assert torch.equal(mask, mask[:, :1, :].expand_as(mask))


def test_quest_page_relocation_keeps_each_heads_selected_tokens():
    class FakeWrapper:
        def __init__(self, layout):
            self.layout = layout
            self.args = None

        def begin_forward(self, *args):
            self.begin = args

        def forward(self, *args):
            self.args = args

    torch.manual_seed(8)
    q = torch.randn(1, 32, 128, dtype=torch.float16)
    k = torch.randn(1, 8, 64, 128, dtype=torch.float16)
    v = torch.randn_like(k)
    mask = torch.zeros(32, 64, dtype=torch.bool)
    for head in range(32):
        pages = sorted({head % 4, (head + 1) % 4})
        for page in pages:
            mask[head, page * 16:(page + 1) * 16] = True
    call, wrapper = quest_setup(SimpleNamespace(BatchDecodeWithPagedKVCachePyTorchWrapper=FakeWrapper), q, k, v, mask)
    call()
    _, _, data, indices, _, last_len, last_idx, _, _ = wrapper.args
    assert last_len == 16 and last_idx == 4
    for head in range(32):
        selected_pages = torch.nonzero(mask[head].view(4, 16).all(-1), as_tuple=True)[0]
        assert int(indices[head, 0]) == int(selected_pages[0])
        expected_k = k[0, head // 4, selected_pages[1] * 16:(selected_pages[1] + 1) * 16]
        expected_v = v[0, head // 4, selected_pages[1] * 16:(selected_pages[1] + 1) * 16]
        torch.testing.assert_close(data[last_idx, 0, :, head], expected_k)
        torch.testing.assert_close(data[last_idx, 1, :, head], expected_v)


def test_quest_split_reconstructs_head_order_and_partial_pages():
    wrappers = []

    class FakeWrapper:
        def __init__(self, layout):
            self.last_len = None
            wrappers.append(self)

        def begin_forward(self, *args):
            pass

        def forward(self, q, out, data, indices, indptr, last_len, last_idx, *args):
            self.last_len = last_len
            out.copy_(q)

        def end_forward(self):
            pass

    q = torch.randn(1, 32, 128, dtype=torch.float16)
    k = torch.randn(1, 8, 35, 128, dtype=torch.float16)
    v = torch.randn_like(k)
    mask = torch.zeros(32, 35, dtype=torch.bool)
    mask[:, :16] = True
    mask[::2, 16:32] = True
    mask[1::2, 32:] = True
    call, wrapper, count = quest_setup_split(SimpleNamespace(BatchDecodeWithPagedKVCachePyTorchWrapper=FakeWrapper), q, k, v, mask)
    assert count == 2
    torch.testing.assert_close(call(), q)
    assert {w.last_len for w in wrappers} == {3, 16}
    wrapper.end_forward()


def test_summary_uses_matched_trace_ratios(tmp_path, monkeypatch):
    replay = tmp_path / "replay"
    replay.mkdir()
    config = {"trace_files": ["trace.pt"], "batch_sizes": [32], "methods": ["gas_fused", "gas_per_head"]}
    (replay / "config.json").write_text(json.dumps(config))
    fields = ["trace", "trace_sha256", "batch", "method", "status", "median_us", "group_union_mean"]
    with (replay / "trace_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, latency in (("gas_fused", 10), ("gas_per_head", 20)):
            writer.writerow(dict(trace="trace.pt", trace_sha256="abc", batch=32, method=method, status="available", median_us=latency, group_union_mean=3))
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["summary", "--replay-dir", str(replay), "--output-dir", str(output)])
    summarize_main()
    with (output / "paired_speedups.csv").open(newline="") as f:
        pair = next(csv.DictReader(f))
    assert pair["baseline"] == "gas_per_head"
    assert float(pair["median_baseline_over_gas"]) == 2.0
    assert pair["same_mask"] == "True"


def test_union_membership_preserves_each_heads_exact_mask():
    mask = torch.zeros(32, 11, dtype=torch.bool)
    for head in range(32):
        mask[head, head % 11] = True
        mask[head, (head + 3) % 11] = True
    indices, membership = build_union_membership(mask, 2, torch.device("cpu"))
    for batch in range(2):
        for head in range(32):
            valid = indices[batch, head // 4] >= 0
            observed = torch.zeros(11, dtype=torch.bool)
            ids = indices[batch, head // 4, valid].long()
            observed[ids] = membership[batch, head, valid]
            assert torch.equal(observed, mask[head])


def test_union_membership_rejects_empty_head():
    mask = torch.ones(32, 4, dtype=torch.bool)
    mask[3] = False
    with pytest.raises(ValueError, match="empty"):
        build_union_membership(mask, 1, torch.device("cpu"))


def test_summary_rejects_incomplete_matrix(tmp_path, monkeypatch):
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "config.json").write_text(json.dumps({"trace_files": ["trace.pt"], "batch_sizes": [32], "methods": ["gas_fused"]}))
    (replay / "trace_results.csv").write_text("trace,trace_sha256,batch,method,status,median_us,group_union_mean\n")
    monkeypatch.setattr(sys, "argv", ["summary", "--replay-dir", str(replay), "--output-dir", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="incomplete"):
        summarize_main()
