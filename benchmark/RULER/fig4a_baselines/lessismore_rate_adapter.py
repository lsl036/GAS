"""Dynamic-rate wrapper around the unmodified LessIsMore attention forward."""

from lessismore_adapter import AccessRecorder
from src.tidal_build.modify_llama_lim import llama_tidal_attention_forward


class RateRecorder(AccessRecorder):
    def __init__(self, num_kv_heads, rate):
        super().__init__(num_kv_heads, sparse_start=2, correction_layer=13)
        self.rate = rate
        self.selected_count = None
        self.last_summary = None

    def reset(self):
        super().reset()
        self.selected_count = None

    def record_layer(self, module, seq_len, budget):
        layer = module.layer_idx
        if layer < self.sparse_start:
            return
        selection = layer in (self.sparse_start, self.correction_layer)
        if selection:
            if budget >= seq_len:
                self.selected_count = seq_len
            else:
                self.selected_count = module.pos_index.shape[-1]
            loaded = self.num_kv_heads * seq_len
        else:
            if self.selected_count is None:
                raise RuntimeError("Missing LessIsMore selection for this decode step")
            loaded = self.num_kv_heads * self.selected_count
        self.records.append({"layer_id": layer, "seq_len": seq_len, "selection_layer": selection,
                             "loaded_kv": float(loaded), "budget": budget})

    def snapshot(self):
        if not self.records:
            return None
        result = self.summarize()
        budgets = [row["budget"] for row in self.records]
        result.update({"budget_min": min(budgets), "budget_max": max(budgets),
                       "budget_mean": sum(budgets) / len(budgets), "rate": self.rate})
        return result

    def get_total_avg_budget(self):
        self.last_summary = self.snapshot()
        return None if self.last_summary is None else self.last_summary["final_loaded_kv_tokens_mean"]


def install_rate(model, rate, correction_layer=13, recent_ratio=0.25):
    if not 0 < rate <= 1 or not 0 <= recent_ratio < 1:
        raise ValueError("invalid rate or recent-window ratio")
    if not 2 < correction_layer < len(model.model.layers):
        raise ValueError("correction layer must follow layer 2 and be in the model")
    tracker = RateRecorder(model.config.num_key_value_heads, rate)
    tracker.correction_layer = correction_layer
    for decoder in model.model.layers:
        module = decoder.self_attn
        module.original_forward = module.forward

        def forward(hidden_states, position_embeddings, attention_mask=None, past_key_value=None,
                    cache_position=None, _module=module, **kwargs):
            q_len = hidden_states.shape[1]
            previous = past_key_value.get_seq_length(_module.layer_idx) if past_key_value is not None else 0
            seq_len = previous + q_len
            budget = max(1, int(seq_len * rate))
            result = llama_tidal_attention_forward(
                _module, hidden_states, position_embeddings, attention_mask, past_key_value,
                cache_position, top_k=budget, sparse_layer_start=2, correction_layer=correction_layer,
                attention_sink=0, lim_ratio_factor=recent_ratio, **kwargs)
            if q_len == 1 and past_key_value is not None:
                tracker.record_layer(_module, seq_len, budget)
            return result

        module.forward = forward
    return tracker
