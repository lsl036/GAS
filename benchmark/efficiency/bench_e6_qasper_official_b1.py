#!/usr/bin/env python3
"""B1-only replay of the upstream Quest or LessIsMore sparse CUDA extension.

Quest uses physically repeated KV heads because upstream requires MHA. For
LessIsMore the shared mask follows the authors' head-ranking and recent-window
selection on a captured search layer. Index preparation is outside timing.
"""

import argparse
import csv
import importlib.util
import json
import math
import statistics
import subprocess
from pathlib import Path

import torch
from bench_e4_kernel_evidence import cuda_event_samples, error_metrics
from bench_e6_qasper_trace import digest, reference, write_rows


class UnsupportedSupport(ValueError):
    pass


def load_extension(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quest_setup(module, q, k, v, mask):
    length = k.shape[-2]
    pages = (length + 15) // 16
    pad = pages * 16 - length
    # The official wrapper has one mandatory, globally addressed last page.
    # Store each query head's own selected last full page in that page's
    # replicated head slot, preserving the exact per-head support.
    mask = torch.nn.functional.pad(mask, (0, pad)).view(32, pages, 16)
    counts = mask.sum(-1)
    if ((counts != 0) & (counts != 16)).any():
        raise UnsupportedSupport("official fixed-page Quest cannot exactly represent a selected partial page")
    selected = counts.bool()
    if selected.sum(-1).unique().numel() != 1:
        raise UnsupportedSupport("official fixed-page Quest requires equal selected page counts per head")
    n = int(selected.sum(-1)[0])
    if n < 2:
        raise UnsupportedSupport("official Quest requires at least two selected pages")
    ids = torch.stack([torch.nonzero(selected[h], as_tuple=True)[0] for h in range(32)]).to(q.device)
    def convert(x):
        x = torch.nn.functional.pad(x, (0, 0, 0, pad)).repeat_interleave(4, 1)
        return x[0].view(32, pages, 16, 128).permute(1, 2, 0, 3).contiguous()
    kp, vp = convert(k), convert(v)
    data = torch.empty(pages + 1, 2, 16, 32, 128, device=q.device, dtype=q.dtype)
    data[:pages, 0], data[:pages, 1] = kp, vp
    for head in range(32):
        data[pages, :, :, head, :] = data[int(ids[head, -1]), :, :, head, :]
    indices = ids[:, :-1].to(torch.int32).contiguous()
    indptr = torch.tensor([0, n - 1], dtype=torch.int32, device=q.device)
    output = torch.empty_like(q)
    wrapper = module.BatchDecodeWithPagedKVCachePyTorchWrapper(0)
    wrapper.begin_forward(indptr, 32, 32, 128, 16, torch.empty(0, dtype=q.dtype))
    def call():
        wrapper.forward(q, output, data, indices, indptr, 16, pages, 1.0, 1e4)
        return output
    return call, wrapper


def quest_setup_split(module, q, k, v, mask):
    """Represent a partial last page exactly via separate official head groups."""
    length = k.shape[-2]
    pages = (length + 15) // 16
    tail = length % 16
    padded = torch.nn.functional.pad(mask, (0, pages * 16 - length)).view(32, pages, 16)
    counts = padded.sum(-1).cpu()
    groups = {}
    for head in range(32):
        full = torch.nonzero(counts[head] == 16, as_tuple=True)[0].tolist()
        partial = torch.nonzero((counts[head] > 0) & (counts[head] < 16), as_tuple=True)[0].tolist()
        if len(partial) > 1 or (partial and (not tail or partial[0] != pages - 1 or int(counts[head, partial[0]]) != tail)):
            raise UnsupportedSupport("mask contains a non-page-aligned non-tail selection")
        if partial:
            remaining, last_page, last_len = full, partial[0], tail
        elif full:
            remaining, last_page, last_len = full[:-1], full[-1], 16
        else:
            raise UnsupportedSupport("empty Quest support")
        if not remaining:
            raise UnsupportedSupport("official Quest requires at least two selected pages")
        groups.setdefault((len(remaining), last_len), []).append((head, remaining, last_page))
    output = torch.empty_like(q)
    entries = []
    try:
        for (n_remaining, last_len), values in sorted(groups.items()):
            head_ids = torch.tensor([v_[0] for v_ in values], dtype=torch.int64, device=q.device)
            kv_ids = head_ids // 4
            q_part = q.index_select(1, head_ids).contiguous()
            def convert(x):
                x = torch.nn.functional.pad(x.index_select(1, kv_ids), (0, 0, 0, pages * 16 - length))
                return x[0].view(len(values), pages, 16, 128).permute(1, 2, 0, 3).contiguous()
            kp, vp = convert(k), convert(v)
            data = torch.empty(pages + 1, 2, 16, len(values), 128, device=q.device, dtype=q.dtype)
            data[:pages, 0], data[:pages, 1] = kp, vp
            for local, (_, _, last_page) in enumerate(values):
                data[pages, :, :, local, :] = data[last_page, :, :, local, :]
            indices = torch.tensor([v_[1] for v_ in values], dtype=torch.int32, device=q.device).contiguous()
            indptr = torch.tensor([0, n_remaining], dtype=torch.int32, device=q.device)
            part_out = torch.empty_like(q_part)
            wrapper = module.BatchDecodeWithPagedKVCachePyTorchWrapper(0)
            wrapper.begin_forward(indptr, len(values), len(values), 128, 16, torch.empty(0, dtype=q.dtype))
            entries.append((wrapper, q_part, part_out, data, indices, indptr, last_len, head_ids))
    except Exception:
        for wrapper, *_ in entries:
            wrapper.end_forward()
        raise

    class Composite:
        def end_forward(self):
            for wrapper, *_ in entries:
                wrapper.end_forward()

    def call():
        for wrapper, query, part, data, indices, indptr, last_len, head_ids in entries:
            wrapper.forward(query, part, data, indices, indptr, last_len, pages, 1.0, 1e4)
            output.index_copy_(1, head_ids, part)
        return output

    return call, Composite(), len(entries)


def less_mask(search, length, k_budget):
    # Match modify_llama_lim.py: rank-interleaved dedup of top-k per head,
    # then reserve the most recent 25%. Selection is outside timing.
    q = search["q"]
    k = search["k"].repeat_interleave(4, 1)
    scores = torch.matmul(q.unsqueeze(2), k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    middle_budget = int(k_budget * 0.75)
    recent = k_budget - middle_budget
    middle = scores[..., : length - recent].topk(middle_budget, dim=-1).indices
    ordered = middle.transpose(1, 3).contiguous().view(-1).tolist()
    unique = list(dict.fromkeys(ordered))[:middle_budget]
    if len(unique) != middle_budget:
        raise ValueError("insufficient unique LessIsMore selection candidates")
    indices = unique + list(range(length - recent, length))
    mask = torch.zeros(1, 32, length, dtype=torch.bool, device=q.device)
    mask[..., indices] = True
    return mask


def less_setup(module, q, k, v, mask):
    length = k.shape[-2]
    ids = torch.nonzero(mask[0, 0], as_tuple=True)[0]
    if int(ids[-1]) != length - 1:
        raise ValueError("official LessIsMore sparse path requires newest token in support")
    # NHD page-size 1, layout [pages, 2, 1, kv_heads, head_dim].
    data = torch.stack((k[0].transpose(0, 1), v[0].transpose(0, 1)), 1).unsqueeze(2).contiguous()
    indices = ids[:-1].to(q.device, dtype=torch.int32).view(1, -1).repeat(32, 1).contiguous()
    indptr = torch.tensor([0, len(ids) - 1], dtype=torch.int32, device=q.device)
    output = torch.empty_like(q)
    qk_product = torch.empty(0, dtype=q.dtype, device=q.device)
    wrapper = module.BatchDecodeWithPagedKVCachePyTorchWrapper(0)
    wrapper.begin_forward(indptr, 32, 8, 128, 1, torch.empty(0, dtype=q.dtype))
    def call():
        wrapper.forward(q, output, data, indices, indptr, 1, length - 1, qk_product, 1.0, 1e4)
        return output
    return call, wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("quest", "less"), required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--module-name", default="_kernels", help="Use the explicit bridge module name when applicable")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--official-object", type=Path, help="CMake-built upstream decode object used by bridge")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--less-k", type=int, default=91)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if min(args.less_k, args.repeats, args.rounds) < 1 or args.warmups < 0:
        parser.error("K, repeats, rounds must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    manifest = json.loads((args.trace_dir / "manifest.json").read_text())
    files = [args.trace_dir / f for f in manifest["files"]
             if (not args.steps or int(f[4:7]) in args.steps) and (not args.layers or int(f[13:15]) in args.layers)]
    if not files:
        raise ValueError("no trace files selected")
    module = load_extension(args.extension, args.module_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(args.source_root), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    config = {"method": args.method, "extension": str(args.extension.resolve()), "extension_sha256": digest(args.extension),
              "source_root": str(args.source_root.resolve()), "source_commit": commit,
              "official_object": str(args.official_object.resolve()) if args.official_object else None,
              "official_object_sha256": digest(args.official_object) if args.official_object else None,
              "extension_module_name": args.module_name,
              "trace_manifest_sha256": digest(args.trace_dir / "manifest.json"), "replay_dtype": "float16",
              "source_dtype": manifest["dtype"], "batch": 1, "less_k": args.less_k,
              "search_layers": [2, 13], "recent_ratio": 0.25,
              "scope": "attention only; selection, data relocation, index preparation and plan excluded",
              "quest_layout": "replicated MHA KV; partial pages use head-grouped upstream calls plus timed output scatter; no native GQA",
              "less_selection": "author quality-path rank interleaving, search-layer trace, full-head shared support",
              "state": "Q/K/V taken from GAS trajectory, not standalone method trajectory",
              "warmups": args.warmups, "repeats": args.repeats, "rounds": args.rounds,
              "script_sha256": digest(Path(__file__)), "trace_files": [p.name for p in files]}
    config_path = args.output_dir / f"{args.method}.config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("existing output has a different protocol")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    for path in files:
        layer, step = int(path.name[13:15]), int(path.name[4:7])
        stem = path.stem + "_" + args.method
        shard = args.output_dir / "shards" / f"{stem}.csv"
        raw_shard = args.output_dir / "samples" / f"{stem}.csv"
        if shard.exists():
            with shard.open(newline="") as f:
                previous = next(csv.DictReader(f))
            if previous["status"] == "available" and not raw_shard.exists():
                raise ValueError(f"missing raw samples for {stem}")
            continue
        row = {"trace": path.name, "trace_sha256": digest(path), "method": args.method, "batch": 1,
               "length": "", "status": "failed", "reason": "", "per_head_mean": "", "group_union_mean": "", "quest_call_groups": "",
               "max_abs_error": "", "relative_l2_error": "", "median_us": "", "samples": 0}
        wrapper = call = None
        try:
            trace = torch.load(path, map_location="cpu", weights_only=True)
            q, k, v = [trace[key].to(device="cuda", dtype=torch.float16).contiguous() for key in ("q", "k", "v")]
            length = k.shape[-2]
            row["length"] = length
            if args.method == "quest":
                mask = trace["candidate"].to("cuda")
                try:
                    call, wrapper = quest_setup(module, q, k, v, mask[0])
                    row["quest_call_groups"] = 1
                except UnsupportedSupport:
                    call, wrapper, row["quest_call_groups"] = quest_setup_split(module, q, k, v, mask[0])
            else:
                # Search layers are dense in the author implementation; report
                # sparse decode only for subsequent layers.
                if layer in (2, 13):
                    raise UnsupportedSupport("author search layer is dense, not a sparse replay")
                search_layer = 2 if layer < 13 else 13
                source = args.trace_dir / f"step{step:03d}_layer{search_layer:02d}.pt"
                search = torch.load(source, map_location="cpu", weights_only=True)
                search = {key: value.to("cuda") for key, value in search.items() if key in ("q", "k")}
                mask = less_mask(search, length, args.less_k)
                call, wrapper = less_setup(module, q, k, v, mask)
            row["per_head_mean"] = float(mask.sum(-1).float().mean())
            row["group_union_mean"] = float(mask.view(1, 8, 4, length).any(2).sum(-1).float().mean())
            ref = reference(q, k, v, mask)
            result = call()
            torch.cuda.synchronize()
            check = error_metrics(result, ref, 0.02, 0.02)
            row["max_abs_error"], row["relative_l2_error"] = check["max_abs_error"], check["relative_l2_error"]
            if not check["pass"] or not check["finite"]:
                raise ValueError(f"exact-mask correctness failed: {check}")
            samples, raw = [], []
            for r in range(args.rounds):
                measured = cuda_event_samples(call, q.device, args.warmups, args.repeats)
                samples.extend(measured)
                raw.extend({"trace": path.name, "method": args.method, "round": r + 1,
                            "repeat": i + 1, "latency_us": value} for i, value in enumerate(measured))
            write_rows(raw_shard, raw)
            row.update(status="available", median_us=statistics.median(samples), samples=len(samples))
        except UnsupportedSupport as exc:
            row["status"] = "unavailable"
            row["reason"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            row["reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            if wrapper is not None:
                wrapper.end_forward()
        write_rows(shard, [row])
        print(f"{stem}: {row['status']} {row['reason']}", flush=True)
        call = wrapper = None
        torch.cuda.empty_cache()
    rows = []
    for shard in sorted((args.output_dir / "shards").glob(f"*_{args.method}.csv")):
        with shard.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    write_rows(args.output_dir / f"{args.method}.results.csv", rows)
    print(f"wrote {len(rows)} official {args.method} rows", flush=True)


if __name__ == "__main__":
    main()
