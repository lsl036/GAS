#!/usr/bin/env python3
"""Correctness-gated attention replay on real Qasper Q/K/V and exact masks.

FlashInfer sparse rows are explicitly compatible controls, not official Quest or
Twilight kernels. A batch replicates a real trace into independent KV storage.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import torch
from bench_e4_kernel_evidence import cuda_event_samples, error_metrics, padded_tensor
from bench_e6_kernel_baselines import backend_fa2, get_gas_module

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("fa2_dense", "flashinfer_dense", "quest_fi_exact", "twilight_fi_exact",
           "twilight_author_dense_ops", "gas_fi_exact", "gas_per_head", "gas_fused")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(q, k, v, mask):
    # Per-head support is retained even when KV is shared by a GQA group.
    batch, heads, dim = q.shape
    kv_heads, length = k.shape[1:3]
    group = heads // kv_heads
    query = q.float().view(batch, kv_heads, group, dim)
    scores = torch.matmul(query, k.float().transpose(-1, -2)) / math.sqrt(dim)
    if mask is not None:
        scores = scores.masked_fill(~mask.view(batch, kv_heads, group, length), -torch.inf)
    probs = torch.softmax(scores, -1)
    return torch.matmul(probs, v.float()).view(batch, heads, dim)


def fi_dense(q, k, v):
    import flashinfer
    batch, heads, dim = q.shape
    kv_heads, length = k.shape[1:3]
    page = 16
    n_pages = (length + page - 1) // page
    pad = n_pages * page - length
    if pad:
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))
    def paged(x):
        return x.view(batch, kv_heads, n_pages, page, dim).permute(0, 2, 1, 3, 4).reshape(-1, kv_heads, page, dim).contiguous()
    pages = (paged(k), paged(v))
    workspace = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="HND")
    indptr = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * n_pages
    indices = torch.arange(batch * n_pages, dtype=torch.int32, device=q.device)
    last = torch.full((batch,), length - (n_pages - 1) * page, dtype=torch.int32, device=q.device)
    wrapper.plan(indptr, indices, last, heads, kv_heads, dim, page, pos_encoding_mode="NONE",
                 q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
    return lambda: wrapper.run(q, pages)


def fi_exact(q, k, v, mask):
    """Exact per-head support via page-1 FI; metadata and plan are precomputed."""
    import flashinfer
    batch, heads, dim = q.shape
    kv_heads, length = k.shape[1:3]
    group = heads // kv_heads
    mask = mask.reshape(batch * heads, length)
    rows, tokens = mask.nonzero(as_tuple=True)
    counts = mask.sum(-1, dtype=torch.int32)
    if (counts == 0).any():
        raise ValueError("empty support cannot be evaluated")
    indptr = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    ids = ((rows // group) * length + tokens).to(torch.int32)
    last = torch.ones(batch * heads, dtype=torch.int32, device=q.device)
    pages = (k.view(-1, 1, 1, dim), v.view(-1, 1, 1, dim))
    workspace = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(indptr, ids, last, 1, 1, dim, 1, pos_encoding_mode="NONE",
                 q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
    return lambda: wrapper.run(q.view(-1, 1, dim), pages).view_as(q)


def twilight_author_dense_ops(q, k, v, mask):
    """Replay the upstream accuracy path's dense QK, masking, softmax and PV."""
    group = q.shape[1] // k.shape[1]
    expanded_k = k.repeat_interleave(group, 1)
    expanded_v = v.repeat_interleave(group, 1)
    def call():
        scores = torch.matmul(q.unsqueeze(2), expanded_k.transpose(2, 3)) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~mask.unsqueeze(2), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(probs, expanded_v).squeeze(2)
    return call


def gas_indices(shared, final, batch, device):
    heads, length = shared.shape
    if heads != 32:
        raise ValueError("Expected 32 Q heads")
    grouped = shared.view(8, 4, length)
    if not torch.equal(grouped, grouped[:, :1, :].expand_as(grouped)):
        raise ValueError("GAS shared support differs within a group")
    exceptions = final & ~shared
    groups = [torch.nonzero(grouped[h, 0], as_tuple=True)[0].tolist() for h in range(8)]
    ex = [torch.nonzero(exceptions[h], as_tuple=True)[0].tolist() for h in range(heads)]
    if max(map(len, ex)) > 16:
        raise ValueError("exception budget exceeded")
    # Avoid zero-width arrays in the CUDA interface.
    gw, ew = max(1, max(map(len, groups))), max(1, max(map(len, ex)))
    return (padded_tensor(groups * batch, (batch, 8), device, width=gw),
            padded_tensor(ex * batch, (batch, heads), device, width=ew))


def write_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if min(args.batch_sizes + [args.repeats, args.rounds]) < 1 or args.warmups < 0:
        parser.error("batch sizes, repeats and rounds must be positive; warmups nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    manifest_file = args.trace_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    files = [args.trace_dir / f for f in manifest["files"]
             if (not args.steps or int(f[4:7]) in args.steps) and (not args.layers or int(f[13:15]) in args.layers)]
    if not files:
        raise ValueError("No trace files selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"input": "real single-sample GAS Q/K/V and masks, question decode", "trace_manifest_sha256": digest(manifest_file),
              "source_dtype": manifest["dtype"], "replay_dtype": "float16; cast before timing",
              "batch": "independent physical copies of same captured request", "batch_sizes": args.batch_sizes,
              "methods": args.methods, "warmups": args.warmups, "repeats": args.repeats, "rounds": args.rounds,
              "attention_only": True, "excluded": "selection, compaction, FI plan, layout/cast, prefill, projections",
              "official_quest": "not implemented by FI control; official wrapper batch=1 and MHA-only",
              "official_twilight": "dense QK/mask/softmax/PV accuracy-path operations; pre-expanded KV, not an optimized sparse kernel",
              "official_lessismore": "not included in this script; separate B1 evaluation",
              "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
              "torch_cuda": torch.version.cuda, "script_sha256": digest(Path(__file__)),
              "gas_source_sha256": digest(ROOT / "csrc/src/gqa_group_decode.cu"),
              "order": "deterministic shuffled methods per trace and batch",
              "trace_files": [f.name for f in files]}
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing output has different protocol; select a new output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    device = torch.device("cuda:0")
    module = get_gas_module() if {"gas_fused", "gas_per_head"} & set(args.methods) else None
    for trace_file in files:
        trace = torch.load(trace_file, map_location="cpu", weights_only=True)
        trace_hash = digest(trace_file)
        q0, k0, v0 = [trace[key].to(device=device, dtype=torch.float16).contiguous() for key in ("q", "k", "v")]
        length = k0.shape[-2]
        masks = {key: trace[key].squeeze(0).bool() for key in ("candidate", "twilight", "shared", "gas")}
        for batch in args.batch_sizes:
            method_order = list(args.methods)
            random.Random(f"{trace_file.name}:{batch}").shuffle(method_order)
            for method in method_order:
                stem = f"{trace_file.stem}_B{batch}_{method}"
                shard = args.output_dir / "shards" / f"{stem}.csv"
                raw_shard = args.output_dir / "samples" / f"{stem}.csv"
                if shard.exists():
                    with shard.open(newline="") as stream:
                        previous = next(csv.DictReader(stream))
                    if previous["status"] == "available" and not raw_shard.exists():
                        raise ValueError(f"Missing raw timing samples for completed shard {stem}")
                    continue
                row = {"trace": trace_file.name, "trace_sha256": trace_hash, "batch": batch,
                       "length": length, "method": method, "status": "failed", "reason": "", "max_abs_error": "",
                       "relative_l2_error": "", "median_us": "", "p10_us": "", "p90_us": "", "samples": 0,
                       "per_head_mean": "", "group_union_mean": "", "group_shared_mean": ""}
                q = k = v = result = ref = mask0 = call = group = exceptions = None
                try:
                    q, k, v = [x.repeat(batch, 1, 1).contiguous() if x.ndim == 3 else x.repeat(batch, 1, 1, 1).contiguous() for x in (q0, k0, v0)]
                    key = {"quest_fi_exact": "candidate", "twilight_fi_exact": "twilight",
                           "twilight_author_dense_ops": "twilight", "gas_fi_exact": "gas",
                           "gas_per_head": "gas", "gas_fused": "gas"}.get(method)
                    mask0 = masks[key].unsqueeze(0).to(device) if key else None
                    ref = reference(q0, k0, v0, mask0)
                    if mask0 is not None:
                        row["per_head_mean"] = float(mask0.sum(-1).float().mean())
                        row["group_union_mean"] = float(mask0.view(1, 8, 4, length).any(2).sum(-1).float().mean())
                    if method == "fa2_dense":
                        call = backend_fa2(q, k, v)
                    elif method == "flashinfer_dense":
                        call = fi_dense(q, k, v)
                    elif method in ("quest_fi_exact", "twilight_fi_exact", "gas_fi_exact"):
                        call = fi_exact(q, k, v, mask0.repeat(batch, 1, 1))
                    elif method == "twilight_author_dense_ops":
                        call = twilight_author_dense_ops(q, k, v, mask0.repeat(batch, 1, 1))
                    elif method == "gas_per_head":
                        head_lists = [torch.nonzero(masks["gas"][h], as_tuple=True)[0].tolist() for h in range(32)]
                        head_ids = padded_tensor(head_lists * batch, (batch, 32), device)
                        call = lambda: module.batched_per_head_sparse_decode(q, k, v, head_ids)
                    else:
                        group, exceptions = gas_indices(masks["shared"], masks["gas"], batch, device)
                        row["group_shared_mean"] = float(masks["shared"].view(8, 4, length)[:, 0].sum(-1).float().mean())
                        call = lambda: module.batched_gqa_group_exception_decode(q, k, v, group, exceptions)
                    result = call()
                    torch.cuda.synchronize()
                    metrics = error_metrics(result, ref.expand_as(result), 0.02, 0.02)
                    row.update({"max_abs_error": metrics["max_abs_error"], "relative_l2_error": metrics["relative_l2_error"]})
                    if not metrics["pass"] or not torch.isfinite(result).all():
                        raise ValueError(f"FP32 exact-mask correctness failed: {metrics}")
                    samples, raw_rows = [], []
                    for round_id in range(args.rounds):
                        measured = cuda_event_samples(call, device, args.warmups, args.repeats)
                        samples.extend(measured)
                        raw_rows.extend({"trace": trace_file.name, "batch": batch, "method": method,
                                         "round": round_id + 1, "repeat": idx + 1, "latency_us": value}
                                        for idx, value in enumerate(measured))
                    ordered = sorted(samples)
                    write_rows(raw_shard, raw_rows)
                    row.update(status="available", median_us=statistics.median(samples),
                               p10_us=ordered[int((len(ordered)-1)*0.1)], p90_us=ordered[int((len(ordered)-1)*0.9)], samples=len(samples))
                except Exception as exc:
                    row["reason"] = f"{type(exc).__name__}: {exc}"
                write_rows(shard, [row])
                print(f"{stem}: {row['status']} {row['reason']}", flush=True)
                # Closures own the FI wrapper and its workspace; drop them before next method.
                call = q = k = v = result = ref = mask0 = group = exceptions = None
                torch.cuda.empty_cache()
    rows = []
    for path in sorted((args.output_dir / "shards").glob("*.csv")):
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    write_rows(args.output_dir / "trace_results.csv", rows)
    print(f"wrote {len(rows)} trace-method rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
