#!/usr/bin/env python3
"""Prototype group-union shared-KV decode with exact per-head masks."""

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load
from bench_e4_kernel_evidence import cuda_event_samples, error_metrics, padded_tensor
from bench_e6_qasper_trace import digest, reference, write_rows

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "csrc/src/e6_gqa_union_masked.cu"
METHODS = {"quest_union_masked_proto": "candidate", "twilight_union_masked_proto": "twilight"}


def build_union_membership(mask, batch, device):
    if mask.ndim != 2 or mask.shape[0] != 32:
        raise ValueError("expected a [32,length] mask")
    grouped = mask.view(8, 4, -1)
    ids = [torch.nonzero(grouped[g].any(0), as_tuple=True)[0].tolist() for g in range(8)]
    if any(not values for values in ids) or not mask.any(-1).all():
        raise ValueError("empty per-head or group support is unsupported")
    width = max(map(len, ids))
    indices = padded_tensor(ids * batch, (batch, 8), device, width=width)
    membership = torch.zeros(32, width, dtype=torch.bool)
    for group, values in enumerate(ids):
        membership[group * 4:(group + 1) * 4, :len(values)] = grouped[group, :, values]
    membership = membership.unsqueeze(0).repeat(batch, 1, 1).to(device).contiguous()
    return indices, membership


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 64, 96, 128])
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if min(args.batch_sizes + [args.repeats, args.rounds]) < 1 or args.warmups < 0:
        parser.error("batch sizes, repeats and rounds must be positive; warmups nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    manifest_path = args.trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = [args.trace_dir / name for name in manifest["files"]
             if (not args.steps or int(name[4:7]) in args.steps) and (not args.layers or int(name[13:15]) in args.layers)]
    if not files:
        raise ValueError("no trace files selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"input": "real Qasper GAS-state Q/K/V, exact Quest and Twilight masks", "trace_manifest_sha256": digest(manifest_path),
              "source_dtype": manifest["dtype"], "replay_dtype": "float16, cast outside timing",
              "batch": "independent physical copies of the same captured request", "batch_sizes": args.batch_sizes,
              "methods": list(METHODS), "warmups": args.warmups, "repeats": args.repeats, "rounds": args.rounds,
              "scope": "attention-only prototype; union/membership construction, selection and cache updates excluded",
              "implementation": "in-house union+per-head-membership kernel, not an official Quest/Twilight kernel",
              "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
              "script_sha256": digest(Path(__file__)), "cuda_source_sha256": digest(SOURCE), "trace_files": [p.name for p in files]}
    path = args.output_dir / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("output directory has a different protocol")
    path.write_text(json.dumps(config, indent=2) + "\n")
    module = load(name="e6_gqa_union_masked", sources=[str(SOURCE)], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    device = torch.device("cuda:0")
    for trace_file in files:
        trace = torch.load(trace_file, map_location="cpu", weights_only=True)
        trace_hash = digest(trace_file)
        q0, k0, v0 = [trace[key].to(device=device, dtype=torch.float16).contiguous() for key in ("q", "k", "v")]
        length = k0.shape[-2]
        for batch in args.batch_sizes:
            methods = list(METHODS)
            random.Random(f"{trace_file.name}:{batch}").shuffle(methods)
            for method in methods:
                stem = f"{trace_file.stem}_B{batch}_{method}"
                shard, raw_shard = args.output_dir / "shards" / f"{stem}.csv", args.output_dir / "samples" / f"{stem}.csv"
                if shard.exists():
                    with shard.open(newline="") as f:
                        previous = next(csv.DictReader(f))
                    if previous["status"] == "available" and not raw_shard.exists():
                        raise ValueError(f"missing raw samples for {stem}")
                    continue
                row = {"trace": trace_file.name, "trace_sha256": trace_hash, "batch": batch, "length": length,
                       "method": method, "status": "failed", "reason": "", "max_abs_error": "", "relative_l2_error": "",
                       "median_us": "", "p10_us": "", "p90_us": "", "samples": 0,
                       "per_head_mean": "", "group_union_mean": "", "group_shared_mean": "", "padded_union_width": ""}
                q = k = v = indices = membership = result = ref = mask0 = call = None
                try:
                    q = q0.repeat(batch, 1, 1).contiguous()
                    k = k0.repeat(batch, 1, 1, 1).contiguous()
                    v = v0.repeat(batch, 1, 1, 1).contiguous()
                    mask = trace[METHODS[method]].squeeze(0).bool()
                    mask0 = mask.unsqueeze(0).to(device)
                    indices, membership = build_union_membership(mask, batch, device)
                    row["padded_union_width"] = int(indices.shape[-1])
                    row["per_head_mean"] = float(mask.sum(-1).float().mean())
                    row["group_union_mean"] = float(mask.view(8, 4, length).any(1).sum(-1).float().mean())
                    ref = reference(q0, k0, v0, mask0)
                    call = lambda: module.decode(q, k, v, indices, membership)
                    result = call()
                    torch.cuda.synchronize(device)
                    check = error_metrics(result, ref.expand_as(result), 0.02, 0.02)
                    row["max_abs_error"], row["relative_l2_error"] = check["max_abs_error"], check["relative_l2_error"]
                    if not check["pass"] or not check["finite"]:
                        raise ValueError(f"exact per-head mask correctness failed: {check}")
                    samples, raw = [], []
                    for r in range(args.rounds):
                        measured = cuda_event_samples(call, device, args.warmups, args.repeats)
                        samples.extend(measured)
                        raw.extend({"trace": trace_file.name, "batch": batch, "method": method, "round": r + 1,
                                    "repeat": i + 1, "latency_us": value} for i, value in enumerate(measured))
                    ordered = sorted(samples)
                    write_rows(raw_shard, raw)
                    row.update(status="available", median_us=statistics.median(samples), p10_us=ordered[int((len(ordered)-1)*.1)],
                               p90_us=ordered[int((len(ordered)-1)*.9)], samples=len(samples))
                except Exception as exc:
                    row["reason"] = f"{type(exc).__name__}: {exc}"
                write_rows(shard, [row])
                print(f"{stem}: {row['status']} {row['reason']}", flush=True)
                q = k = v = indices = membership = result = ref = mask0 = call = None
                torch.cuda.empty_cache()
    rows = []
    for shard in sorted((args.output_dir / "shards").glob("*.csv")):
        with shard.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    write_rows(args.output_dir / "trace_results.csv", rows)
    print(f"wrote {len(rows)} union-prototype rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
