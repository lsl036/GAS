import pytest
import torch

from bench_quest_official_headpacked import page_groups, quest_setup


def test_groups_full_and_tail_pages():
    masks = torch.zeros(2, 32, 48, dtype=torch.bool)
    masks[0, :, :32] = True
    masks[1, :, :16] = True
    masks[1, :, 32:37] = True
    groups = page_groups(masks, [48, 37])
    assert sorted((key, len(values)) for key, values in groups.items()) == [((1, 5), 32), ((1, 16), 32)]
    assert groups[(1, 5)][0] == (32, 8, [0], 2)
    assert groups[(1, 16)][0] == (0, 0, [0], 1)


def test_rejects_non_tail_partial_page():
    mask = torch.zeros(1, 32, 48, dtype=torch.bool)
    mask[:, :, :16] = True
    mask[:, :, 17:21] = True
    with pytest.raises(ValueError, match="non-tail partial"):
        page_groups(mask, [48])


def test_rejects_masked_padding():
    mask = torch.zeros(1, 32, 48, dtype=torch.bool)
    mask[:, :, :16] = True
    mask[:, :, 40] = True
    with pytest.raises(ValueError, match="padding"):
        page_groups(mask, [37])


def test_scatter_covers_every_request_and_head():
    class Wrapper:
        def __init__(self, layout):
            self.closed = False

        def begin_forward(self, indptr, heads, kv_heads, dim, page_size, dtype):
            assert heads == kv_heads

        def forward(self, q, out, data, indices, indptr, last_len, last_idx, scale, theta):
            out.copy_(q)

        def end_forward(self):
            self.closed = True

    class Module:
        BatchDecodeWithPagedKVCachePyTorchWrapper = Wrapper

    q = torch.arange(2 * 32 * 128, dtype=torch.float32).view(2, 32, 128)
    k = torch.randn(2, 8, 48, 128)
    mask = torch.zeros(2, 32, 48, dtype=torch.bool)
    mask[0, :, :32] = True
    mask[1, :, :16] = True
    mask[1, :, 32:37] = True
    call, wrappers = quest_setup(Module(), q, k, k, mask, [48, 37], 10)
    assert len(wrappers) == 8
    assert torch.equal(call(), q)
    for wrapper in wrappers:
        wrapper.end_forward()
    assert all(wrapper.closed for wrapper in wrappers)
