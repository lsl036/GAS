"""Compatibility wrapper around the checked-out LessIsMore LLaMA implementation."""

from __future__ import annotations

import sys
import typing
from pathlib import Path

import torch
import transformers.utils

VENDOR = Path(__file__).resolve().parent / "vendor" / "LessIsMore"
sys.path.insert(0, str(VENDOR))

# The upstream model uses LossKwargs only as a type annotation; Transformers
# 4.57 no longer exports it from this location.
if not hasattr(transformers.utils, "LossKwargs"):
    transformers.utils.LossKwargs = typing.TypedDict("LossKwargs", {})

from src.models.llama_tidaldecoding import LlamaForCausalLM  # noqa: E402
from src.tidal_build.modify_llama_lim import enable_llama_tidal_attention  # noqa: E402


class AccessRecorder:
    def __init__(self, num_kv_heads: int, sparse_start: int, correction_layer: int):
        self.num_kv_heads = num_kv_heads
        self.sparse_start = sparse_start
        self.correction_layer = correction_layer
        self.records = []

    def reset(self):
        self.records.clear()

    def record(self, layer, mask, seq_len):
        if layer < self.sparse_start:
            return
        selection_layer = layer in (self.sparse_start, self.correction_layer)
        # Upstream computes and returns full attention at the selection layers;
        # pos_mask is used only by subsequent sparse attention layers.
        if selection_layer:
            count = float(self.num_kv_heads * seq_len)
        else:
            if mask is None or mask.ndim != 4 or mask.shape[2] != 1:
                raise ValueError(f"Missing decode mask at layer {layer}")
            batch, heads, _, length = mask.shape
            if heads % self.num_kv_heads:
                raise ValueError("Invalid GQA head count")
            groups = mask.to(torch.bool).reshape(batch, self.num_kv_heads, heads // self.num_kv_heads, length)
            count = float(groups.any(dim=2).sum(dim=(-2, -1)).float().mean().item())
        self.records.append({"layer_id": layer, "seq_len": seq_len,
                             "selection_layer": selection_layer, "loaded_kv": count})

    def summarize(self):
        if not self.records:
            raise ValueError("No decode attention records; sparse patch may not have run")
        sparse = [r for r in self.records if not r["selection_layer"]]
        average = lambda rows: sum(r["loaded_kv"] for r in rows) / len(rows) if rows else None
        return {"final_loaded_kv_tokens_mean": average(self.records),
                "sparse_layers_only_loaded_kv_mean": average(sparse),
                "profile_record_count": len(self.records),
                "selection_layer_record_count": len(self.records) - len(sparse),
                "metric_start_layer": self.sparse_start}


def install(model, top_k: int, correction_layer: int = 13, recent_ratio: float = 0.25,
            track_metrics: bool = True):
    if top_k <= 0 or not 0 <= recent_ratio < 1:
        raise ValueError("invalid LessIsMore budget or recent-window ratio")
    layers = model.model.layers
    if not 2 < correction_layer < len(layers):
        raise ValueError("correction_layer must be after layer 2 and inside the model")
    recorder = AccessRecorder(model.config.num_key_value_heads, 2, correction_layer)
    # Upstream enable_tidal(..., 'lim') forwards 'lim' to a helper that only
    # installs its forward for 'tidal'. Calling the LessIsMore helper with that
    # sentinel enables its actual implementation without changing its math.
    enable_llama_tidal_attention(model, top_k, attn_type="tidal", sparse_layer_start=2,
                                correction_layer=correction_layer, attn_sink=0,
                                lim_ratio=recent_ratio)
    if not track_metrics:
        return None
    for decoder in layers:
        module = decoder.self_attn
        original = module.forward

        def wrapped(*args, _module=module, _original=original, **kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            result = _original(*args, **kwargs)
            if hidden is not None and hidden.shape[1] == 1:
                cache = kwargs.get("past_key_value")
                if cache is None:
                    raise RuntimeError("LessIsMore decode called without a KV cache")
                seq_len = cache.get_seq_length(_module.layer_idx)
                recorder.record(_module.layer_idx, getattr(_module, "pos_mask", None), seq_len)
            return result

        module.forward = wrapped
    return recorder
