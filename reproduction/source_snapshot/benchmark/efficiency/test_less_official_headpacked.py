import pytest
import torch

from bench_less_official_headpacked import official_grouped_setup, official_setup


class FakeWrapper:
    def __init__(self, layout):
        self.closed = False

    def begin_forward(self, indptr, num_heads, num_kv_heads, dim, page_size, dtype):
        self.num_heads = num_heads

    def forward(self, q, output, data, indices, indptr, last_len, last_idx, empty, scale, theta):
        assert q.shape[1] == self.num_heads
        assert indices.shape[0] == self.num_heads
        output.copy_(q)

    def end_forward(self):
        self.closed = True


class FakeModule:
    BatchDecodeWithPagedKVCachePyTorchWrapper = FakeWrapper


def inputs(batch):
    q = torch.arange(batch * 4 * 128, dtype=torch.float32).reshape(batch, 4, 128)
    k = torch.randn(batch, 1, 4, 128)
    ids = torch.tensor([[0, 3]] * batch)
    return q, k, k.clone(), ids, [4] * batch


def test_grouped_output_and_call_count():
    q, k, v, ids, lengths = inputs(5)
    call, wrappers = official_grouped_setup(FakeModule(), q, k, v, ids, lengths, 2)
    assert len(wrappers) == 3
    assert torch.equal(call(), q)
    for wrapper in wrappers:
        wrapper.end_forward()
    assert all(wrapper.closed for wrapper in wrappers)


def test_rejects_non_latest_token():
    q, k, v, ids, lengths = inputs(2)
    ids[0, -1] = 2
    with pytest.raises(ValueError, match="latest token"):
        official_setup(FakeModule(), q, k, v, ids, lengths)
