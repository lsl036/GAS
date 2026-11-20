"""E7 v2: real-weight, real-trace, connected component-based decode estimate."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from bench_e4_kernel_evidence import error_metrics, fp32_reference, padded_tensor
from bench_e6_kernel_baselines import get_gas_module
from e7_v2_pipeline import AttentionPath, reference_support, rms_norm, rotate, support_statistics
from prepare_e7_v2_traces import trace_path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("fa2", "flashinfer", "quest", "quest_twi", "gas")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class CheckpointWeights:
    def __init__(self, directory):
        self.directory = directory
        self.mapping = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name):
        with safe_open(str(self.directory / self.mapping[name]), framework="pt", device="cpu") as file:
            return file.get_tensor(name).to(device="cuda", dtype=torch.float16)

    def layer(self, index):
        names = ("input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
        return {name: self.get(f"model.layers.{index}.{name}.weight") for name in names}


def common_callable(data, weights, eps, heads, kv_heads):
    x, attn = data["x"], data["attn_input"]
    cos, sin = data["cos"][:, None], data["sin"][:, None]

    def run():
        norm = rms_norm(x, weights["input_layernorm"], eps)
        query = F.linear(norm, weights["self_attn.q_proj"]).view(x.shape[0], heads, -1)
        key = F.linear(norm, weights["self_attn.k_proj"]).view(x.shape[0], kv_heads, -1)
        value = F.linear(norm, weights["self_attn.v_proj"]).view(x.shape[0], kv_heads, -1)
        query, key = rotate(query, cos, sin), rotate(key, cos, sin)
        residual = x + F.linear(attn, weights["self_attn.o_proj"])
        norm = rms_norm(residual, weights["post_attention_layernorm"], eps)
        mlp = F.linear(F.silu(F.linear(norm, weights["mlp.gate_proj"])) * F.linear(norm, weights["mlp.up_proj"]), weights["mlp.down_proj"])
        return residual + mlp, query, key, value

    return run


def output_callable(data, weights, eps):
    embedding, norm, head = weights

    def run():
        embedded = F.embedding(data["token"].flatten(), embedding)
        logits = F.linear(rms_norm(data["x"], norm, eps), head)
        return logits.argmax(-1), embedded, logits

    return run


def correctness(path):
    output = path.run(keep=True)
    if path.method in ("fa2", "flashinfer"):
        final = torch.ones(1, path.heads, path.length, dtype=torch.bool, device=path.q.device)
        checks = {"selection_matches_quality_reference": True, "set_invariants": True}
    else:
        reference = reference_support(path)
        differences = {key: int((path.last[key][:1] != mask).sum()) for key, mask in reference.items()}
        checks = {"selection_matches_quality_reference": not any(differences.values()), "selection_differences_json": json.dumps(differences, sort_keys=True)}
        selected = path.last.get("selected", path.last["candidate"])
        invariants = not bool((selected & ~path.last["candidate"]).any())
        if "shared" in path.last:
            union = selected.view(path.batch, path.kv_heads, path.group, path.length).any(2).repeat_interleave(path.group, 1)
            exceptions = path.last["final"] & ~path.last["shared"]
            invariants = invariants and not bool((path.last["shared"] & ~union).any())
            invariants = invariants and not bool((exceptions & ~selected).any()) and bool((exceptions.sum(-1) <= 16).all())
        checks["set_invariants"] = invariants
        final = path.last["final"][:1]
    lists = [row.nonzero().flatten().tolist() for row in final[0]]
    ids = padded_tensor(lists, (1, path.heads), path.q.device)
    expected = fp32_reference(path.q[:1], path.k[:1], path.v[:1], ids, [len(row) for row in lists], path.group)
    metrics = error_metrics(output[:1], expected, 0.02, 0.02)
    metrics["finite"] = bool(torch.isfinite(output).all())
    stats = support_statistics(path)
    path.last.clear()
    return {**checks, **metrics, "pass": checks["selection_matches_quality_reference"] and checks["set_invariants"] and metrics["pass"] and metrics["finite"]}, stats


def measure(function, warmups, repeats, profile=False):
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    rows = []
    for repeat in range(repeats):
        stages = [] if profile else None
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        start = time.perf_counter()
        result = function(stages) if profile else function()
        end.record()
        end.synchronize()
        host_us = (time.perf_counter() - start) * 1e6
        rows.append({"repeat": repeat + 1, "component": "full_path", "latency_us": begin.elapsed_time(end) * 1000, "host_wall_us": host_us, "timing_scope": "primary_cuda_interval_including_host_gaps" if not profile else "diagnostic_total"})
        if profile:
            rows.extend({"repeat": repeat + 1, "component": name, "latency_us": a.elapsed_time(b) * 1000, "host_wall_us": None, "timing_scope": "diagnostic_stage"} for name, a, b in stages)
        del result
    return rows


def expand_trace(path, batch, include_kv=True):
    data = load_file(str(path), device="cpu")
    result = {}
    for key, value in data.items():
        if key in ("k", "v") and not include_kv:
            continue
        value = value.cuda()
        result[key] = value.repeat(batch, *([1] * (value.ndim - 1))).contiguous()
    return result


def protocol(args, manifest):
    sources = [Path(__file__), Path(__file__).with_name("e7_v2_pipeline.py"), Path(__file__).with_name("prepare_e7_v2_traces.py"), Path(__file__).with_name("summarize_e7_v2.py"), Path(__file__).with_name("run_e7_v2.sh"), Path(__file__).with_name("bench_e4_kernel_evidence.py"), Path(__file__).with_name("bench_e6_kernel_baselines.py"), ROOT / "csrc/src/gqa_group_decode.cu"]
    sources += [ROOT / "gas_attention/pyimpl" / name for name in ("quest.py", "quantize.py", "gqa_refine.py", "top_p.py")]
    hashes = {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest() for file in sources}
    value = {"version": "e7_v2_real_trace_connected", "source_hashes": hashes, "trace_manifest": manifest, "warmups": args.warmups, "repeats": args.repeats, "rounds": args.rounds, "profile_repeats": args.profile_repeats, "seed": args.seed, "candidate_budget": 8192, "chunk_size": 16, "p_t": 0.85, "p_g": 0.80, "exception_budget": 16, "dense_layers": [0, 1], "dense_backend_for_sparse": "fa2", "graph": False, "dispatch": "GAS fused unless its fixed 96KiB shared-memory capacity is exceeded; then same-support FI", "primary_timing": "CUDA event interval across serialized connected replay, including host dispatch/plan gaps; synchronized host wall time also provided", "common_scope": "actual weights: two norms, QKV, RoPE, O, residuals, SwiGLU; cache append belongs to path", "output_scope": "actual embedding, final norm, LM head, argmax", "estimator_storage": "persistent FP16 dequantized 4-bit min-max K; prefix setup excluded, append quantization included", "input_type": "real_dense_checkpoint_trace_homogeneous_batch_replay", "residency_assumption": "per-layer measurement; sum excludes offload/communication; full model+all KV need not fit", "all_layers_measured": len(manifest["layers"]) == manifest["model_config"]["num_hidden_layers"]}
    quality_config = json.loads((ROOT / "benchmark/configs/config_quest_twi_gqa_group_max_p080_exception16.json").read_text())
    frozen = (quality_config["selector"]["type"], quality_config["selector"]["token_budget"], quality_config["selector"]["chunk_size"], quality_config["weight_estimator"]["type"], quality_config["weight_estimator"]["quant_bit"], quality_config["weight_pruner"]["threshold"], quality_config["gqa_refinement"]["mode"], quality_config["gqa_refinement"]["threshold"], quality_config["gqa_refinement"]["exception_budget"], quality_config["skip_first_two_layers"])
    if frozen != ("quest", 8192, 16, "min_max_quant", 4, 0.85, "group_max", 0.80, 16, True):
        raise ValueError(f"Quality config differs from the frozen E7 experiment: {frozen}")
    value["quality_config"] = quality_config
    model = Path(manifest["model"])
    model_index = model / "model.safetensors.index.json"
    value["checkpoint_index_sha256"] = hashlib.sha256(model_index.read_bytes()).hexdigest()
    value["checkpoint_shards"] = [{"name": f.name, "bytes": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns} for f in sorted(model.glob("model-*.safetensors"))]
    value["gpu"] = torch.cuda.get_device_name()
    value["torch"], value["cuda"] = torch.__version__, torch.version.cuda
    value["software"] = {}
    for name in ("transformers", "flash-attn", "flashinfer-python", "safetensors"):
        try:
            value["software"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            value["software"][name] = "not_found"
    value["protocol_hash"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    try:
        value["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True)
    except (OSError, subprocess.CalledProcessError):
        value["nvidia_smi"] = "unavailable"
    return value


def unit_path(args, meta):
    stem = f"L{meta['length']}_B{meta['batch_size']}_s{meta['sample']}_d{meta['step']}_l{meta['layer_id']}_r{meta['round']}_{meta['method']}"
    return args.output_dir / "shards" / (stem + ".json")


def completed(args, meta, config):
    file = unit_path(args, meta)
    if not file.exists():
        return False
    previous = json.loads(file.read_text())
    if previous["protocol_hash"] != config["protocol_hash"]:
        raise ValueError(f"Protocol mismatch: {file}; use a new output directory")
    return (previous["status"] == "available" and len(previous.get("timings", [])) == args.repeats) or (previous["status"] == "unavailable" and not args.retry_unavailable)


def run_unit(args, config, meta, build):
    if completed(args, meta, config):
        return
    payload = {**meta, "trace_id": f"sample{meta['sample']}_step{meta['step']}", "protocol_hash": config["protocol_hash"], "status": "available"}
    function = resource = data = weights = None
    try:
        function, resource, check, stats = build()
        if not check["pass"]:
            raise ValueError(f"Correctness failed: {check}")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        payload.update(correctness=check, support=stats)
        payload["timings"] = measure(function, args.warmups, args.repeats)
        if isinstance(resource, AttentionPath):
            payload["diagnostic_timings"] = measure(resource.run, 1, args.profile_repeats, profile=True)
            payload["dispatch"] = resource.dispatch
            if resource.method in ("quest_twi", "gas") and meta["batch_size"] in (32, 128) and meta["layer_id"] == 2 and meta["sample"] == 0 and meta["step"] == 1 and meta["round"] == 1:
                def refresh_all_then_run():
                    for start in range(0, resource.length, 256):
                        resource.khat[:, :, start:start + 256] = resource.quant(resource.k[:, :, start:start + 256], 4)
                    return resource.run()
                before_result = refresh_all_then_run().clone()
                after_result = resource.run()
                equivalent = bool(torch.allclose(before_result, after_result, atol=0.02, rtol=0.02))
                before = measure(refresh_all_then_run, 1, args.profile_repeats)
                after = measure(resource.run, 1, args.profile_repeats)
                payload["optimization_timings"] = [dict(row, component="full_path_" + label, timing_scope="diagnostic_optimization") for label, records in (("refresh_all_prefix", before), ("persistent_prefix", after)) for row in records]
                payload["optimization"] = {"variant": "full_prefix_refresh_vs_persistent_key_grid", "same_semantics_pass": equivalent, "before_full_path_us": statistics.fmean(r["latency_us"] for r in before), "after_full_path_us": statistics.fmean(r["latency_us"] for r in after), "repeats": args.profile_repeats, "scope": "diagnostic_validation_after_fixed_design; not tuning or primary timing"}
                del before_result, after_result
                if not equivalent:
                    raise ValueError("Persistent estimator cache changed attention output")
        payload["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        payload["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    except (torch.OutOfMemoryError, ImportError) as exc:
        payload.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        payload.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        dump(unit_path(args, meta), payload)
        raise
    finally:
        function = resource = data = weights = None
        gc.collect()
        torch.cuda.empty_cache()
    dump(unit_path(args, meta), payload)
    elapsed = statistics.fmean(row["latency_us"] for row in payload.get("timings", [])) if payload.get("timings") else None
    print(f"saved {unit_path(args, meta).name}: {payload['status']}, mean_us={elapsed}", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[16384, 32768])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 96, 128])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--allow-subset", action="store_true", help="Smoke only; incomplete matrices are never paper-ready")
    parser.add_argument("--retry-unavailable", action="store_true")
    args = parser.parse_args()
    if min(args.batch_sizes + args.lengths + [args.repeats, args.rounds, args.profile_repeats]) < 1 or args.warmups < 0:
        parser.error("dimensions and counts must be positive")
    manifest = json.loads((args.trace_dir / "trace_manifest.json").read_text())
    cfg = manifest["model_config"]
    if not manifest["complete"]:
        parser.error("Trace preparation is incomplete; resume it first")
    formal = (args.lengths == [16384, 32768] and args.batch_sizes == [32, 64, 96, 128] and manifest["layers"] == list(range(32)) and len(manifest["sources"]) >= 3 and manifest["steps"] == [1, 16] and args.rounds == 5 and args.repeats == 100 and args.warmups == 20)
    if not formal and not args.allow_subset:
        parser.error("Nonformal matrix requires --allow-subset")
    for length in args.lengths:
        if length not in manifest["lengths"]:
            parser.error(f"Missing trace length {length}")
    config = protocol(args, manifest)
    config.update(formal_matrix=formal, lengths=args.lengths, batch_sizes=args.batch_sizes)
    config_file = args.output_dir / "config.json"
    if config_file.exists() and json.loads(config_file.read_text())["protocol_hash"] != config["protocol_hash"]:
        parser.error("Existing directory uses a different protocol; choose a new output directory")
    dump(config_file, config)
    for relative in config["source_hashes"]:
        destination = args.output_dir / "code_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    dump(args.output_dir / "E7_trace_manifest.json", {**manifest, "trace_directory": str(args.trace_dir.resolve()), "batch_construction": "each real source replayed independently; each batch row has separate physical KV, no aliasing", "full_layer_weights": {str(i): 1 for i in manifest["layers"]}})
    dump(args.output_dir / "methods.json", {"fa2": "flash_attn_with_kvcache, persistent native layout", "flashinfer": "native 32/8 GQA page-16 decode, dynamic plan included", "quest": "cached per-page extrema + actual Quest scores/topk + FI per-head page-16 decode", "quest_twi": "Quest + cached 4-bit-grid K estimation + repository head top-p + group union + FI GQA page-1 decode", "gas": "same Quest/estimator/head top-p + repository group-max p=.80 + exception16 + fused decode (fixed capacity fallback to exact-support FI page-1)", "sparse_dense_layers": "FA2 at actual layer IDs 0 and 1", "graphs": False})
    module = get_gas_module()
    checkpoint = CheckpointWeights(Path(manifest["model"]))
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    if cfg["num_attention_heads"] != 32 or cfg["num_key_value_heads"] != 8 or head_dim != 128:
        raise ValueError("E7 protocol expects LLaMA-3.1-8B GQA 32/8/128")
    try:
        for length in args.lengths:
            for batch in args.batch_sizes:
                for sample in range(len(manifest["sources"])):
                    for step in manifest["steps"]:
                        for layer in manifest["layers"] + ["output"]:
                            metas = []
                            for round_id in range(1, args.rounds + 1):
                                methods = ["common"] + list(args.methods) if layer != "output" else ["output"]
                                if isinstance(layer, int) and layer < 2:
                                    methods = ["common"] + [method for method in ("fa2", "flashinfer") if method in args.methods or method == "fa2"]
                                random.Random(args.seed + length * 31 + batch * 17 + sample * 13 + step * 7 + (layer if isinstance(layer, int) else 32) * 11 + round_id).shuffle(methods)
                                for method in methods:
                                    metas.append({"length": length, "batch_size": batch, "sample": sample, "step": step, "layer_id": layer, "round": round_id, "method": method, "order": ">".join(methods)})
                            pending = [meta for meta in metas if not completed(args, meta, config)]
                            if not pending:
                                continue
                            file = trace_path(args.trace_dir, length, sample, step, layer)
                            data = expand_trace(file, batch)
                            for meta in pending:
                                def build(meta=meta):
                                    method = meta["method"]
                                    if method == "output":
                                        weights = [checkpoint.get(name) for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")]
                                        function = output_callable(data, weights, cfg["rms_norm_eps"])
                                        result, embedded, logits = function()
                                        agreement = float((result == data["next_token"].flatten()).float().mean())
                                        reference = F.linear(rms_norm(data["x"][:1].float(), weights[1].float(), cfg["rms_norm_eps"]), weights[2].float())
                                        check = error_metrics(logits[:1], reference, 0.02, 0.02)
                                        check["greedy_trace_agreement"] = agreement
                                        check["embedding_finite"] = bool(torch.isfinite(embedded).all())
                                        check["pass"] = check["pass"] and check["embedding_finite"]
                                        return function, weights, check, {}
                                    if method == "common":
                                        weights = checkpoint.layer(layer)
                                        function = common_callable(data, weights, cfg["rms_norm_eps"], cfg["num_attention_heads"], cfg["num_key_value_heads"])
                                        y, query, key, val = function()
                                        check = error_metrics(y[:1], data["y"][:1].float(), 0.02, 0.02)
                                        check["qkv_match"] = all(torch.allclose(a[:1], b[:1], atol=0.02, rtol=0.02) for a, b in ((query, data["q"]), (key, data["k"][:, :, -1]), (val, data["v"][:, :, -1])))
                                        check["pass"] = check["pass"] and check["qkv_match"]
                                        return function, weights, check, {}
                                    path = AttentionPath(method, data["q"], data["k"], data["v"], module)
                                    check, stats = correctness(path)
                                    return path.run, path, check, stats
                                run_unit(args, config, meta, build)
                            del data, build
                            gc.collect()
                            torch.cuda.empty_cache()
    finally:
        from summarize_e7_v2 import summarize
        summarize(args.output_dir)


if __name__ == "__main__":
    main()
