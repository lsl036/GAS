import json

import pytest
import torch

from bench_e6_qasper_trace import digest, reference, write_rows
from bench_gas_mixed_trace import chunked_reference, choose_replay_policy, load_cached, make_batch, pack_dense_pages


def trace(length, value):
    q = torch.full((1, 32, 128), value, dtype=torch.bfloat16)
    k = torch.full((1, 8, length, 128), value, dtype=torch.bfloat16)
    shared = torch.zeros(1, 32, length, dtype=torch.bool)
    shared[:, :, 0] = True
    gas = shared.clone()
    gas[:, :, length - 1] = True
    return {"q": q, "k": k, "v": k.clone(), "shared": shared,
            "gas": gas, "twilight": gas.clone(), "candidate": gas.clone()}


def test_mixed_padding_and_repetition():
    q, k, v, masks, shared, exceptions, lengths = make_batch([trace(3, 1), trace(5, 2)], 3, "cpu")
    assert lengths == [3, 5]
    assert q[:, 0, 0].tolist() == [1, 2, 1]
    assert k.shape == v.shape == (3, 8, 5, 128)
    assert torch.all(k[[0, 2], :, 3:] == 0)
    assert not masks["gas"][[0, 2], :, 3:].any()
    assert shared[:, :, 0].eq(0).all()
    assert exceptions[:, :, 0].tolist() == [[2] * 32, [4] * 32, [2] * 32]


@pytest.mark.parametrize("lengths", [(5, 7), (8, 8), (1, 8)])
def test_pack_dense_pages_matches_valid_kv_and_zeroes_padding(lengths):
    source = torch.arange(2 * 3 * 8 * 4).view(2, 3, 8, 4)
    packed = pack_dense_pages(source, lengths, 4).view(2, 2, 3, 4, 4)
    restored = packed.permute(0, 2, 1, 3, 4).reshape(2, 3, 8, 4)
    for b, length in enumerate(lengths):
        torch.testing.assert_close(restored[b, :, :length], source[b, :, :length])
        assert not restored[b, :, length:].any()


def test_rejects_inconsistent_shared_mask():
    t = trace(5, 1)
    t["shared"][0, 1, 0] = False
    with pytest.raises(ValueError, match="group-shared"):
        make_batch([t], 1, "cpu")


def test_rejects_empty_support():
    t = trace(5, 1)
    t["twilight"][0, 0].zero_()
    with pytest.raises(ValueError, match="empty support"):
        make_batch([t], 1, "cpu")


@pytest.mark.parametrize("batch,width,expected", [
    (32, 255, "inline8"), (128, 255, "inline8"),
    (32, 256, "inline16"), (64, 512, "inline16"),
    (96, 256, "prefix_groupsplit128"), (128, 512, "prefix_groupsplit128"),
    (32, 1023, "inline16"), (32, 1024, "prefix_groupsplit128"),
    (64, 2048, "prefix_groupsplit128"),
])
def test_replay_policy_uses_shape_only(batch, width, expected):
    assert choose_replay_policy(batch, width) == expected


@pytest.mark.parametrize("chunk", [1, 2, 8])
@pytest.mark.parametrize("masked", [False, True])
def test_chunked_reference_matches_full_batch(chunk, masked):
    torch.manual_seed(7)
    q = torch.randn(5, 8, 16)
    k = torch.randn(5, 2, 11, 16)
    v = torch.randn_like(k)
    mask = torch.rand(5, 8, 11) > 0.5 if masked else None
    if mask is not None:
        mask[:, :, 0] = True
    torch.testing.assert_close(chunked_reference(q, k, v, mask, chunk), reference(q, k, v, mask))


def test_chunked_reference_rejects_zero_chunk():
    with pytest.raises(ValueError, match="positive"):
        chunked_reference(torch.empty(1), None, None, None, 0)


def test_resume_validates_trace_and_raw_samples(tmp_path):
    trace_path = tmp_path / "trace.pt"
    trace_path.write_bytes(b"trace")
    shard = tmp_path / "shard.csv"
    sample = tmp_path / "samples.csv"
    assert load_cached(shard, [sample], [trace_path], 2) is None
    write_rows(shard, [{"samples": 2, "trace_sha256": json.dumps([digest(trace_path)])}])
    write_rows(sample, [{"latency_us": 1}, {"latency_us": 2}])
    assert load_cached(shard, [sample], [trace_path], 2)["samples"] == "2"
    trace_path.write_bytes(b"different")
    with pytest.raises(ValueError, match="invalid existing shard"):
        load_cached(shard, [sample], [trace_path], 2)
    trace_path.write_bytes(b"trace")
    write_rows(sample, [{"latency_us": 1}])
    with pytest.raises(ValueError, match="invalid sample shard"):
        load_cached(shard, [sample], [trace_path], 2)
