#!/usr/bin/env python3
"""One-shape component timing for the actual Python sparse-attention prototype.

The selectors/refinement and FI plan are measured, but checkpoint projections,
prefix metadata/quantization, KV append and full-model inference are excluded.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from bench_e4_kernel_evidence import error_metrics
from bench_e6_qasper_official_b1 import less_mask
from bench_e6_qasper_trace import digest, reference as attention_reference
from bench_gas_mixed_trace import dense_backends
from e7_v2_pipeline import mask_to_pages
from twilight.pyimpl.gqa_refine import build_group_refined_mask
from twilight.pyimpl.quantize import min_max_per_token_quant_kv
from twilight.pyimpl.quest import quest_selector
from twilight.pyimpl.top_p import top_p_unnormalized


class Timer:
    def __init__(self, active):
        self.active = active
        self.rows = []

    def run(self, name, fn):
        if not self.active:
            return fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        self.rows.append({"stage": name, "wall_us": (time.perf_counter() - wall) * 1e6,
                          "event_us": start.elapsed_time(end) * 1000})
        return out


def padded_bounds(k, chunk=16):
    batch, groups, length, dim = k.shape
    full, tail = divmod(length, chunk)
    blocks = k[:, :, :full * chunk].reshape(batch, groups, full, chunk, dim)
    low, high = blocks.amin(3), blocks.amax(3)
    if tail:
        last = k[:, :, full * chunk:]
        low = torch.cat((low, last.amin(2, keepdim=True)), dim=2)
        high = torch.cat((high, last.amax(2, keepdim=True)), dim=2)
    return low, high


def grouped_quest(q, low, high, length, budget, chunk=16):
    batch, heads, dim = q.shape
    groups = low.shape[1]
    query = q.reshape(batch, groups, heads // groups, 1, dim)
    # Match the reference's sign-adjusted chunk maxima and one dot product.
    bounds = torch.where(query > 0, high.unsqueeze(2), -low.unsqueeze(2))
    positive_query = query.abs()
    if length % chunk:
        # The reference concatenates a float32 padding tensor before amax.
        bounds, positive_query = bounds.float(), positive_query.float()
    scores = torch.matmul(positive_query, bounds.transpose(-1, -2)).squeeze(-2)
    selected = scores.topk(min(math.ceil(budget / chunk), scores.shape[-1]), dim=-1).indices
    tokens = (selected[..., None] * chunk + torch.arange(chunk, device=q.device)).reshape(batch, heads, -1)
    padded = low.shape[2] * chunk
    mask = torch.zeros(batch, heads, padded, dtype=torch.bool, device=q.device)
    mask.scatter_(-1, tokens, True)
    return mask[..., :length].contiguous()


def less_select(q, k, budget):
    """Author's rank-interleaved first-request union, valid here for repeated inputs only."""
    batch, heads, dim = q.shape
    groups, length = k.shape[1:3]
    recent = budget - int(budget * 0.75)
    middle = budget - recent
    logits = torch.matmul(q.view(batch, groups, heads // groups, dim), k.transpose(-1, -2)) / math.sqrt(dim)
    ranks = logits.reshape(batch, heads, length)[..., :length - recent].topk(middle, dim=-1).indices
    ordered = ranks.transpose(1, 2).contiguous().view(batch, -1)[0].tolist()
    chosen = list(dict.fromkeys(ordered))[:middle]
    if len(chosen) != middle:
        raise ValueError("insufficient Less unique candidates")
    ids = torch.tensor(chosen + list(range(length - recent, length)), device=q.device)
    mask = torch.zeros(batch, heads, length, dtype=torch.bool, device=q.device)
    mask.scatter_(-1, ids.view(1, 1, -1).expand(batch, heads, -1), True)
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=("sdpa_flash", "fi_dense", "quest", "quest_twi", "gas", "less"), required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--less-k", type=int, default=468)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch < 1 or args.repeats < 1 or args.warmups < 0:
        p.error("positive batch/repeats and nonnegative warmups required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    trace_path = args.trace_dir / "step001_layer15.pt"
    manifest_path = args.trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    trace = torch.load(trace_path, map_location="cpu", weights_only=True)
    q, k, v = [trace[x].to(device="cuda", dtype=torch.float16).repeat(args.batch, 1, 1, *([1] if x != "q" else [])) for x in ("q", "k", "v")]
    batch, heads, dim = q.shape
    groups, length = k.shape[1:3]
    if heads != 32 or groups != 8 or dim != 128:
        raise ValueError("expected Llama-3.1 GQA shape")
    budget = int(length * 0.08)
    if args.method == "less" and args.less_k >= length:
        raise ValueError("invalid Less budget")
    low = high = khat = qsearch = ksearch = None
    prefix_quant_wall_us = None
    reference_diff = None
    if args.method in ("quest", "quest_twi", "gas"):
        low, high = padded_bounds(k)
        candidate = grouped_quest(q[:1], low[:1], high[:1], length, budget)
        repeated = k[:1].repeat_interleave(4, 1)
        reference = quest_selector(SimpleNamespace(num_heads=heads), q[:1, :, None], repeated,
                                   chunk_size=16, sparsity_rate=0.08).squeeze(2)
        reference_diff = int((candidate != reference).sum().item())
        del repeated, reference, candidate
    if args.method in ("quest_twi", "gas"):
        khat = torch.empty_like(k)
        torch.cuda.synchronize()
        quant_start = time.perf_counter()
        for start in range(0, length, 256):
            khat[:, :, start:start + 256] = min_max_per_token_quant_kv(k[:, :, start:start + 256], 4)
        torch.cuda.synchronize()
        prefix_quant_wall_us = (time.perf_counter() - quant_start) * 1e6
    search_hash = None
    if args.method == "less":
        search_path = args.trace_dir / "step001_layer13.pt"
        search = torch.load(search_path, map_location="cpu", weights_only=True)
        qsearch = search["q"].to(device="cuda", dtype=torch.float16).repeat(batch, 1, 1)
        ksearch = search["k"].to(device="cuda", dtype=torch.float16).repeat(batch, 1, 1, 1)
        if ksearch.shape[2] != length:
            raise ValueError("search and replay length mismatch")
        search_hash = digest(search_path)
    wrapper = None
    if args.method in ("quest", "quest_twi", "gas", "less"):
        import flashinfer
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
        pages = (k.view(-1, 1, 1, dim), v.view(-1, 1, 1, dim))
    if args.method == "fi_dense":
        _, dense_fi = dense_backends(q, k, v, [length], include_fa2=False, include_fi=True)

    def run(active):
        t = Timer(active)
        if args.method == "sdpa_flash":
            def dense():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(q.unsqueeze(2), k, v, dropout_p=0.0, enable_gqa=True).squeeze(2)
            output = t.run("attention", dense)
            return output, None, t.rows
        if args.method == "fi_dense":
            return t.run("attention", dense_fi), None, t.rows
        if args.method == "less":
            mask = t.run("less_search_qk_cross_head_selection", lambda: less_select(qsearch, ksearch, args.less_k))
        else:
            mask = t.run("selector", lambda: grouped_quest(q, low, high, length, budget))
            if args.method in ("quest_twi", "gas"):
                def estimate():
                    query = q.view(batch, groups, heads // groups, dim)
                    scores = torch.empty(batch, groups, heads // groups, length, dtype=q.dtype, device=q.device)
                    for head in range(heads // groups):
                        scores[:, :, head:head + 1] = torch.matmul(query[:, :, head:head + 1], khat.transpose(-1, -2)) / math.sqrt(dim)
                    return scores.reshape(batch, heads, 1, length).masked_fill(~mask.unsqueeze(2), torch.finfo(q.dtype).min)
                scores = t.run("estimator_cached_fake_quant_qk", estimate)
                selected = t.run("top_p", lambda: top_p_unnormalized(scores, 0.85) & mask.unsqueeze(2))
                if args.method == "gas":
                    mask = t.run("gas_refinement", lambda: build_group_refined_mask(scores, selected, groups, 0.8, "group_max", 16)).squeeze(2)
                else:
                    mask = selected.squeeze(2)
        metadata = t.run("mask_to_fi_metadata", lambda: mask_to_pages(mask, length, heads // groups))
        def plan():
            indptr, ids, last = metadata
            wrapper.plan(indptr, ids, last, num_qo_heads=1, num_kv_heads=1, head_dim=dim,
                         page_size=1, pos_encoding_mode="NONE", q_data_type=q.dtype,
                         kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
        t.run("fi_plan", plan)
        output = t.run("attention", lambda: wrapper.run(q.reshape(-1, 1, dim), pages).reshape(batch, heads, dim))
        return output, mask, t.rows

    with torch.inference_mode():
        for _ in range(args.warmups):
            out, _, _ = run(False)
            if not torch.isfinite(out).all():
                raise ValueError("nonfinite output")
        torch.cuda.synchronize()
        runs = []
        for _ in range(args.repeats):
            out, mask, rows = run(True)
            if not torch.isfinite(out).all():
                raise ValueError("nonfinite output")
            runs.append(rows)
        connected = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            wall = time.perf_counter()
            out, mask, _ = run(False)
            torch.cuda.synchronize()
            connected.append((time.perf_counter() - wall) * 1e6)
        ref_mask = mask[:1] if mask is not None else torch.ones(1, heads, length, dtype=torch.bool, device=q.device)
        expected = attention_reference(q[:1], k[:1], v[:1], ref_mask)
        check = error_metrics(out[:1], expected, 0.02, 0.02)
        if not check["pass"]:
            raise ValueError(f"attention output does not match FP32 reference: {check}")
        quality_mask_diff = None
        if args.method in ("quest_twi", "gas"):
            repeated = k[:1].repeat_interleave(4, 1)
            quality_candidate = quest_selector(SimpleNamespace(num_heads=heads), q[:1, :, None], repeated,
                                               chunk_size=16, sparsity_rate=0.08)
            quality_weights = torch.matmul(q[:1, :, None], min_max_per_token_quant_kv(repeated, 4).transpose(-1, -2)) / math.sqrt(dim)
            quality_weights.masked_fill_(~quality_candidate, torch.finfo(q.dtype).min)
            quality_selected = top_p_unnormalized(quality_weights, 0.85) & quality_candidate
            quality_mask = (build_group_refined_mask(quality_weights, quality_selected, groups, 0.8, "group_max", 16)
                            if args.method == "gas" else quality_selected).squeeze(2)
            quality_mask_diff = int((mask[:1] != quality_mask).sum().item())
        elif args.method == "less":
            quality_mask = less_mask({"q": qsearch[:1], "k": ksearch[:1]}, length, args.less_k)
            quality_mask_diff = int((mask[:1] != quality_mask).sum().item())
        result = {"method": args.method, "input": str(trace_path.resolve()), "trace_sha256": digest(trace_path),
                  "search_trace_sha256": search_hash, "sample_index": manifest["sample_index"],
                  "length": length, "batch": batch, "dtype": str(q.dtype), "quest_budget": budget,
                  "less_k": args.less_k if args.method == "less" else None,
                  "selector_vs_quality_token_differences_first_request": reference_diff,
                  "final_mask_vs_quality_token_differences_first_request": quality_mask_diff,
                  "final_per_head_tokens_mean": float(mask.sum(-1).float().mean()) if mask is not None else None,
                  "final_group_union_mean": float(mask.reshape(batch, groups, heads // groups, length).any(2).sum(-1).float().mean()) if mask is not None else None,
                  "stages": {r["stage"]: {key: statistics.median(item[i][key] for item in runs) for key in ("wall_us", "event_us")}
                             for i, r in enumerate(runs[0])},
                  "raw_runs": runs, "connected_wall_us": connected,
                  "connected_wall_median_us": statistics.median(connected),
                  "torch": torch.__version__, "device": torch.cuda.get_device_name(),
                  "fp32_attention_reference_check_first_request": check,
                  "prefix_quantization_setup_wall_us": prefix_quant_wall_us,
                  "peak_process_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "caveats": ["single Qasper trace repeated across batch; synthetic 32K repeated context",
                              "stage synchronization alters timings; connected wall measurement is separately recorded",
                              "grouped Torch selector uses persistent page bounds, which may differ numerically from quality selector",
                              "fake-quant FP16 prefix and page min/max prepared outside timing; no cache append or checkpoint projection",
                              "sparse attention uses FI page-size-1 per-head adapter, not native Quest or GAS fused kernel",
                              "FI metadata and plan are charged to sparse paths, dense FI plan is cached",
                              "Less selection is recomputed at search layer 13, using identical inputs; original code shares first batch mask and reuses it across layers"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.method, {name: round(x["wall_us"], 1) for name, x in result["stages"].items()}, flush=True)


if __name__ == "__main__":
    main()
