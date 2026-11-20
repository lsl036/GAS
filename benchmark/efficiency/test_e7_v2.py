import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from e7_v2_pipeline import mask_to_pages, quest_candidates
from summarize_e7_v2 import summarize


class ConnectedE7Test(unittest.TestCase):
    def test_resume_requires_matching_protocol_and_complete_repeats(self):
        from bench_e7_v2_estimated_tpot import completed, unit_path
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output_dir=Path(temporary), repeats=2, retry_unavailable=False)
            meta = {"length": 16384, "batch_size": 32, "sample": 0, "step": 1, "layer_id": 2, "round": 1, "method": "gas"}
            file = unit_path(args, meta)
            file.parent.mkdir()
            payload = {"protocol_hash": "frozen", "status": "available", "timings": [{}, {}]}
            file.write_text(json.dumps(payload))
            self.assertTrue(completed(args, meta, {"protocol_hash": "frozen"}))
            with self.assertRaises(ValueError):
                completed(args, meta, {"protocol_hash": "changed"})
            payload["timings"].pop()
            file.write_text(json.dumps(payload))
            self.assertFalse(completed(args, meta, {"protocol_hash": "frozen"}))

    def test_quest_page_extrema_match_direct_signed_key_scoring(self):
        torch.manual_seed(123)
        batch, heads, kv_heads, length, dim = 2, 8, 2, 64, 8
        q = torch.randn(batch, heads, dim)
        k = torch.randn(batch, kv_heads, length, dim)
        blocks = k.view(batch, kv_heads, -1, 16, dim)
        actual = quest_candidates(q, blocks.amin(3), blocks.amax(3), length, 32)
        pages = quest_candidates(q, blocks.amin(3), blocks.amax(3), length, 32, pages_only=True)
        token_ids = (pages.long()[..., None] * 16 + torch.arange(16)).flatten(-2)
        expanded = torch.zeros_like(actual).scatter_(-1, token_ids, True)
        self.assertTrue(torch.equal(actual, expanded))
        for b in range(batch):
            for h in range(heads):
                sign = torch.where(q[b, h] > 0, 1, -1)
                signed = (k[b, h // (heads // kv_heads)] * sign).view(-1, 16, dim).amax(1)
                scores = signed @ q[b, h].abs()
                ids = scores.topk(2).indices[:, None] * 16 + torch.arange(16)
                expected = torch.zeros(length, dtype=torch.bool)
                expected[ids.flatten()] = True
                self.assertTrue(torch.equal(actual[b, h], expected))

    def test_varlen_metadata_retains_head_to_physical_kv_mapping(self):
        mask = torch.zeros(2, 4, 17, dtype=torch.bool)
        for b in range(2):
            for h in range(4):
                mask[b, h, :h + b + 1] = True
        indptr, pages, last = mask_to_pages(mask, 17, 2)
        for logical in range(8):
            selected = mask.reshape(8, 17)[logical].nonzero().flatten()
            expected = (logical // 2) * 17 + selected
            actual = pages[indptr[logical]:indptr[logical + 1]].long()
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(last.tolist(), [1] * 8)

    def test_dense_layers_use_dense_backend_and_missing_data_is_not_averaged(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = {"trace_manifest": {"sources": [{}], "steps": [1], "layers": [0, 1, 2]}, "protocol_hash": "test", "lengths": [512], "batch_sizes": [2], "rounds": 1, "formal_matrix": False, "input_type": "unit_test"}
            (directory / "config.json").write_text(json.dumps(config))
            (directory / "shards").mkdir()
            for layer in [0, 1, 2, "output"]:
                methods = ["output"] if layer == "output" else ["common", "fa2", "flashinfer"] + (["quest", "quest_twi", "gas"] if layer == 2 else [])
                for method in methods:
                    cost = {"common": 10, "output": 5, "fa2": 100, "flashinfer": 80, "quest": 30, "quest_twi": 20, "gas": 15}[method]
                    row = {"length": 512, "batch_size": 2, "sample": 0, "step": 1, "trace_id": "sample0_step1", "layer_id": layer, "round": 1, "method": method, "order": "test", "status": "available", "protocol_hash": "test", "peak_allocated_gib": 1, "peak_reserved_gib": 1, "timings": [{"repeat": 1, "component": "full_path", "latency_us": cost, "host_wall_us": cost, "timing_scope": "primary"}], "diagnostic_timings": [{"repeat": 1, "component": "attention", "latency_us": cost, "timing_scope": "diagnostic_stage"}]}
                    (directory / "shards" / f"{layer}_{method}.json").write_text(json.dumps(row))
            summarize(directory)
            import csv
            with (directory / "E7_summary.csv").open() as file:
                rows = {r["method"]: r for r in csv.DictReader(file)}
            self.assertAlmostEqual(float(rows["gas"]["estimated_tpot_ms"]), (30 + 5 + 200 + 15) / 1000)
            self.assertAlmostEqual(float(rows["gas"]["dense_layer_ms"]), 0.2)
            self.assertEqual(rows["gas"]["speedup_vs_quest"], "")
            (directory / "shards" / "2_quest.json").unlink()
            summarize(directory)
            with (directory / "E7_summary.csv").open() as file:
                rows = {r["method"]: r for r in csv.DictReader(file)}
            self.assertEqual(rows["quest"]["estimated_tpot_ms"], "")
            self.assertEqual(rows["quest"]["status"], "incomplete_or_unavailable")


if __name__ == "__main__":
    unittest.main()
