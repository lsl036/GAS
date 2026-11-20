import torch

from bench_e7_v2_estimated_tpot import common_callable, output_callable


def test_common_component_shapes():
    torch.manual_seed(7)
    batch, hidden, heads, kv_heads, head_dim, mlp = 3, 8, 4, 1, 2, 12
    data = {"x": torch.randn(batch, hidden), "attn_input": torch.randn(batch, hidden),
            "cos": torch.ones(batch, head_dim), "sin": torch.zeros(batch, head_dim)}
    weights = {"input_layernorm": torch.ones(hidden), "post_attention_layernorm": torch.ones(hidden),
               "self_attn.q_proj": torch.randn(heads * head_dim, hidden),
               "self_attn.k_proj": torch.randn(kv_heads * head_dim, hidden),
               "self_attn.v_proj": torch.randn(kv_heads * head_dim, hidden),
               "self_attn.o_proj": torch.randn(hidden, hidden), "mlp.gate_proj": torch.randn(mlp, hidden),
               "mlp.up_proj": torch.randn(mlp, hidden), "mlp.down_proj": torch.randn(hidden, mlp)}
    x, q, k, v = common_callable(data, weights, 1e-5, heads, kv_heads)()
    assert x.shape == (batch, hidden)
    assert q.shape == (batch, heads, head_dim)
    assert k.shape == v.shape == (batch, kv_heads, head_dim)
    assert all(torch.isfinite(t).all() for t in (x, q, k, v))


def test_output_component_shapes():
    x = torch.randn(2, 8)
    data = {"x": x, "token": torch.tensor([1, 2])}
    embed = torch.randn(10, 8)
    ids, embedded, logits = output_callable(data, [embed, torch.ones(8), embed], 1e-5)()
    assert ids.shape == (2,) and embedded.shape == (2, 8) and logits.shape == (2, 10)
    assert torch.equal(ids, logits.argmax(-1))
