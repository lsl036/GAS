#!/usr/bin/env python3
"""Reconstruct captured Quest/Twilight/GAS and Less support for one trace."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from bench_e6_qasper_official_b1 import less_mask
from bench_e6_qasper_trace import digest
from twilight.pyimpl.gqa_refine import build_group_refined_mask
from twilight.pyimpl.quantize import min_max_per_token_quant_kv
from twilight.pyimpl.top_p import top_p_unnormalized


def describe(tensor):
    values = tensor.float()
    return {"min": float(values.min()), "mean": float(values.mean()), "max": float(values.max())}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required to reproduce the captured attention path")
    manifest = json.loads((args.trace_dir / "manifest.json").read_text())
    bench = json.loads(args.benchmark_config.read_text())
    if manifest["config"] != bench["config"] or digest(args.trace_dir / "step001_layer15.pt") not in bench["trace_sha256"]:
        raise ValueError("trace does not belong to the benchmark configuration")
    data = torch.load(args.trace_dir / "step001_layer15.pt", map_location="cpu", weights_only=True)
    search = torch.load(args.trace_dir / "step001_layer13.pt", map_location="cpu", weights_only=True)
    q, k = data["q"].to("cuda"), data["k"].to("cuda")
    candidate = data["candidate"].to("cuda").bool().unsqueeze(2)
    twilight = data["twilight"].to("cuda").bool().unsqueeze(2)
    shared = data["shared"].to("cuda").bool().unsqueeze(2)
    gas = data["gas"].to("cuda").bool().unsqueeze(2)
    keys = k.repeat_interleave(4, dim=1)
    quantized = min_max_per_token_quant_kv(keys, 4)
    scores = torch.matmul(q.unsqueeze(2), quantized.transpose(2, 3)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~candidate, torch.finfo(scores.dtype).min)
    rebuilt_twilight = candidate & top_p_unnormalized(scores, 0.85)
    rebuilt_shared = build_group_refined_mask(scores, twilight, 8, 0.8, "group_max", 0)
    rebuilt_gas = build_group_refined_mask(scores, twilight, 8, 0.8, "group_max", 16)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    retained_mass = (probabilities * twilight).sum(-1)
    group_union = twilight.view(1, 8, 4, 1, -1).any(dim=2, keepdim=True).expand(-1, -1, 4, -1, -1).reshape_as(twilight)
    exceptions = gas & ~shared
    less_search = {name: search[name].to("cuda") for name in ("q", "k")}
    less = less_mask(less_search, k.shape[2], bench["less_k"]).unsqueeze(2)
    less_indices = torch.nonzero(less[0, 0, 0], as_tuple=True)[0].cpu().contiguous()
    result = {
        "sample_index": manifest["sample_index"], "input_length": k.shape[2], "less_k": bench["less_k"],
        "trace_sha256": digest(args.trace_dir / "step001_layer15.pt"),
        "search_sha256": digest(args.trace_dir / "step001_layer13.pt"),
        "head_counts": {name: describe(value.sum(-1)) for name, value in
                        (("quest", candidate), ("twilight", twilight), ("shared", shared), ("gas", gas), ("less", less))},
        "group_union_counts": {name: describe(value.view(1, 8, 4, 1, -1).any(dim=2).sum(-1)) for name, value in
                               (("quest", candidate), ("twilight", twilight), ("gas", gas), ("less", less))},
        "twilight_estimated_probability_mass": describe(retained_mass),
        "violations": {"twilight_outside_quest": int((twilight & ~candidate).sum()),
                       "shared_outside_twilight_union": int((shared & ~group_union).sum()),
                       "exceptions_outside_own_twilight": int((exceptions & ~twilight).sum()),
                       "heads_over_exception_16": int((exceptions.sum(-1) > 16).sum()),
                       "gas_outside_twilight_group_union": int((gas & ~group_union).sum()),
                       "twilight_reconstruction_differences": int((rebuilt_twilight ^ twilight).sum()),
                       "shared_reconstruction_differences": int((rebuilt_shared ^ shared).sum()),
                       "gas_reconstruction_differences": int((rebuilt_gas ^ gas).sum())},
        "gas_outside_own_quest": int((gas & ~candidate).sum()),
        "less_outside_each_head_quest": int((less & ~candidate).sum()),
        "less_support_sha256": hashlib.sha256(less_indices.numpy().tobytes()).hexdigest(),
        "less_recent_tokens": bench["less_k"] - int(bench["less_k"] * 0.75),
        "less_note": "Reconstructed from search-layer GAS-state Q/K; the original benchmark did not store mask hashes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
