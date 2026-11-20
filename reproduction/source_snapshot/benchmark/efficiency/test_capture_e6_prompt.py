import pytest
import torch

from capture_e6_qasper_trace import split_prompt_tokens


def test_prompt_split_preserves_exact_tokens():
    tokens = torch.arange(12).reshape(1, 12)
    prefill, forced = split_prompt_tokens(tokens, 3)
    assert prefill.shape == (1, 9)
    assert forced.shape == (1, 3)
    assert torch.equal(torch.cat((prefill, forced), dim=1), tokens)


@pytest.mark.parametrize("shape,count", [((1, 3), 3), ((2, 4), 1), ((4,), 1), ((1, 4), 0)])
def test_prompt_split_rejects_invalid_inputs(shape, count):
    with pytest.raises(ValueError):
        split_prompt_tokens(torch.zeros(shape, dtype=torch.long), count)
