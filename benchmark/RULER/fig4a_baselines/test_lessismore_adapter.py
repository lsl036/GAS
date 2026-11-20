import unittest
import copy

import torch
from transformers import LlamaConfig

from lessismore_adapter import LlamaForCausalLM, install
from lessismore_rate_adapter import install_rate


class LessIsMoreAdapterTest(unittest.TestCase):
    def test_rate_matches_fixed_budget_and_counts_actual_masks(self):
        config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=5, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=64)
        config._attn_implementation = "sdpa"
        torch.manual_seed(7)
        fixed = LlamaForCausalLM(config).eval()
        dynamic = copy.deepcopy(fixed)
        fixed_tracker = install(fixed, top_k=2, correction_layer=3, recent_ratio=0.25)
        rate_tracker = install_rate(dynamic, rate=0.2, correction_layer=3, recent_ratio=0.25)
        inputs = torch.arange(12).unsqueeze(0)
        with torch.inference_mode():
            a_cache = fixed(input_ids=inputs, use_cache=True).past_key_values
            b_cache = dynamic(input_ids=inputs, use_cache=True).past_key_values
            a = fixed(input_ids=torch.tensor([[4]]), past_key_values=a_cache, use_cache=True).logits
            b = dynamic(input_ids=torch.tensor([[4]]), past_key_values=b_cache, use_cache=True).logits
        torch.testing.assert_close(a, b)
        actual = fixed_tracker.summarize()
        measured = rate_tracker.snapshot()
        self.assertEqual(measured["budget_min"], 2)
        self.assertEqual(measured["budget_max"], 2)
        self.assertEqual(measured["final_loaded_kv_tokens_mean"], actual["final_loaded_kv_tokens_mean"])
        self.assertEqual(measured["sparse_layers_only_loaded_kv_mean"], actual["sparse_layers_only_loaded_kv_mean"])
        with torch.inference_mode():
            dynamic(input_ids=torch.tensor([[5]]), past_key_values=b_cache, use_cache=True)
            dynamic(input_ids=torch.tensor([[6]]), past_key_values=b_cache, use_cache=True)
        self.assertEqual(rate_tracker.snapshot()["budget_min"], 2)
        self.assertEqual(rate_tracker.snapshot()["budget_max"], 3)

    def test_disabling_metrics_does_not_change_output(self):
        config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=5, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=64)
        config._attn_implementation = "sdpa"
        torch.manual_seed(7)
        tracked = LlamaForCausalLM(config).eval()
        untracked = copy.deepcopy(tracked)
        install(tracked, top_k=4, correction_layer=3)
        self.assertIsNone(install(untracked, top_k=4, correction_layer=3, track_metrics=False))
        inputs = torch.arange(12).unsqueeze(0)
        with torch.inference_mode():
            tracked_cache = tracked(input_ids=inputs, use_cache=True).past_key_values
            untracked_cache = untracked(input_ids=inputs, use_cache=True).past_key_values
            a = tracked(input_ids=torch.tensor([[4]]), past_key_values=tracked_cache, use_cache=True).logits
            b = untracked(input_ids=torch.tensor([[4]]), past_key_values=untracked_cache, use_cache=True).logits
        torch.testing.assert_close(a, b)

    def test_prefill_decode_and_layer_cost(self):
        config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=5, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=64)
        config._attn_implementation = "sdpa"
        torch.manual_seed(7)
        model = LlamaForCausalLM(config).eval()
        recorder = install(model, top_k=4, correction_layer=3, recent_ratio=0.25)
        with torch.inference_mode():
            prefill = model(input_ids=torch.randint(0, 128, (1, 12)), use_cache=True)
            model(input_ids=torch.tensor([[4]]), past_key_values=prefill.past_key_values,
                  use_cache=True)
        info = recorder.summarize()
        self.assertEqual(info["profile_record_count"], 3)
        self.assertEqual(info["selection_layer_record_count"], 2)
        self.assertEqual(info["sparse_layers_only_loaded_kv_mean"], 8.0)
        self.assertEqual(info["final_loaded_kv_tokens_mean"], 20.0)


if __name__ == "__main__":
    unittest.main()
