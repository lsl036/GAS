#!/usr/bin/env python3
"""Capture Qasper or RULER GAS Q/K/V and masks for kernel replay."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark/LongBench"))


def split_prompt_tokens(input_ids, num_decode_tokens):
    """Keep the last prompt tokens as teacher-forced decode inputs."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or num_decode_tokens < 1 or input_ids.shape[1] <= num_decode_tokens:
        raise ValueError("prompt must contain a nonempty prefill and requested decode tokens")
    return input_ids[:, :-num_decode_tokens], input_ids[:, -num_decode_tokens:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=14)
    parser.add_argument("--steps", type=int, nargs="+", default=[1])
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(2, 32)))
    parser.add_argument("--model", type=Path, default=Path("/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct"))
    parser.add_argument("--config", type=Path, default=ROOT / "benchmark/configs/config_quest_twi_gas_ratio8_llama31.json")
    parser.add_argument("--dataset", type=Path, default=ROOT.parent / "data/LongBench/data/qasper.jsonl")
    parser.add_argument("--max-prompt-length", type=int, help="Qasper middle-truncation limit before chat formatting")
    parser.add_argument("--ruler-jsonl", type=Path, help="RULER JSONL with preformatted 'input'; bypass Qasper prompting")
    args = parser.parse_args()
    if args.sample_index < 0 or any(x < 1 or x > 14 for x in args.steps):
        parser.error("sample index must be nonnegative; trace steps must be between 1 and 14 question tokens")
    if args.max_prompt_length is not None and (args.max_prompt_length < 2 or args.ruler_jsonl):
        parser.error("max-prompt-length must be >= 2 and is only supported for Qasper")
    if any(x < 2 or x >= 32 for x in args.layers) or len(set(args.layers)) != len(args.layers):
        parser.error("layers must be distinct sparse layer IDs in [2,31]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Trace output directory must be empty")
    import torch
    import pred
    import gas_attention.pyimpl.attention as attention

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    source = args.ruler_jsonl or args.dataset
    rows = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
    row = rows[args.sample_index]
    if args.ruler_jsonl and not isinstance(row.get("input"), str):
        raise ValueError("RULER sample must contain a string input")
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    config = args.config.resolve()
    model_name = "Meta-Llama-3.1-8B-Instruct"
    pred.seed_everything(42)
    model, tokenizer, _, budget_info, score_info = pred.load_model_and_tokenizer(
        str(args.model), model_name, torch.device("cuda"), SimpleNamespace(algo_config_path=str(config)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = {"step": 0, "active": False, "data": {}}
    orig_selector, orig_refine = attention.quest_selector, attention.build_group_refined_mask
    original_forwards = []

    def selector(*pos, **kw):
        result = orig_selector(*pos, **kw)
        if context["active"]:
            context["data"].update(q=pos[1].detach().squeeze(2).cpu(),
                                   k=pos[2][:, ::4].detach().cpu(),
                                   candidate=result.detach().squeeze(2).cpu())
        return result

    def refine(*pos, **kw):
        result = orig_refine(*pos, **kw)
        if context["active"]:
            shared = orig_refine(pos[0], pos[1], pos[2], pos[3], pos[4], 0)
            context["data"].update(twilight=pos[1].detach().squeeze(2).cpu(),
                                   shared=shared.detach().squeeze(2).cpu(),
                                   gas=result.detach().squeeze(2).cpu())
        return result

    attention.quest_selector, attention.build_group_refined_mask = selector, refine
    for layer_id, layer in enumerate(model.model.layers):
        module = layer.self_attn
        original = module.forward
        original_forwards.append((module, original))

        def wrapped(self, *pos, _original=original, _layer=layer_id, **kw):
            hidden = kw.get("hidden_states", pos[0] if pos else None)
            is_decode = hidden is not None and hidden.shape[1] == 1
            if is_decode and _layer == 0:
                context["step"] += 1
            context["active"] = is_decode and _layer in args.layers and context["step"] in args.steps
            if context["active"]:
                context["data"] = {}
            result = _original(*pos, **kw)
            if context["active"]:
                cache = kw.get("past_key_values", kw.get("past_key_value"))
                if cache is None:
                    raise RuntimeError("Expected a mutable KV cache in decode")
                cache_k, cache_v = cache[_layer]
                data = context["data"]
                data["v"] = cache_v.detach().cpu()
                if not torch.equal(data["k"], cache_k.detach().cpu()):
                    raise RuntimeError("Captured selector K does not match KV cache")
                expected = {"q", "k", "v", "candidate", "twilight", "shared", "gas"}
                if set(data) != expected:
                    raise RuntimeError(f"Incomplete trace fields: {sorted(data)}")
                if (data["twilight"] & ~data["candidate"]).any():
                    raise RuntimeError("Twilight mask exceeds Quest candidates")
                if (data["shared"] & ~data["gas"]).any():
                    raise RuntimeError("Shared mask is not contained in GAS mask")
                exceptions = data["gas"] & ~data["shared"]
                if (exceptions & ~data["twilight"]).any():
                    raise RuntimeError("Exception mask exceeds per-head Twilight support")
                twi_union = data["twilight"].view(1, 8, 4, -1).any(2, keepdim=True)
                if (data["shared"].view(1, 8, 4, -1) & ~twi_union).any():
                    raise RuntimeError("Shared mask exceeds Twilight group union")
                if (data["gas"] & ~data["shared"]).sum(-1).max() > 16:
                    raise RuntimeError("Per-head exception budget exceeded")
                target = args.output_dir / f"step{context['step']:03d}_layer{_layer:02d}.pt"
                torch.save(data, target)
            context["active"] = False
            return result

        module.forward = MethodType(wrapped, module)

    try:
        if args.ruler_jsonl:
            input_ids = tokenizer(row["input"], add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
            prefill, forced = split_prompt_tokens(input_ids, max(args.steps))
            with torch.no_grad():
                output = model(input_ids=prefill, use_cache=True)
                cache = output.past_key_values
                for token in forced[0]:
                    output = model(input_ids=token.view(1, 1), past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
                prediction = tokenizer.decode(output.logits[:, -1, :].argmax(dim=-1).tolist())
            input_token_count = input_ids.shape[1]
        else:
            prompt = json.loads((ROOT / "benchmark/LongBench/config/dataset2prompt.json").read_text())["qasper"]
            max_length = json.loads((ROOT / "benchmark/LongBench/config/model2maxlen.json").read_text())[model_name]
            if args.max_prompt_length is not None:
                max_length = args.max_prompt_length
            prediction = pred._predict_one(model, tokenizer, row, max_length, 1, prompt, "qasper",
                                           torch.device("cuda"), model_name, budget_info, score_info)
    finally:
        attention.quest_selector, attention.build_group_refined_mask = orig_selector, orig_refine
        for module, original in original_forwards:
            module.forward = original
    expected = [args.output_dir / f"step{s:03d}_layer{layer:02d}.pt" for s in args.steps for layer in args.layers]
    if not all(path.is_file() for path in expected):
        raise RuntimeError(f"Missing trace files: {[str(p) for p in expected if not p.is_file()]}")
    manifest = {"task": "ruler_forced_prompt_decode" if args.ruler_jsonl else "qasper", "sample_index": args.sample_index,
                "sample_sha256": hashlib.sha256(payload).hexdigest(), "model": str(args.model.resolve()),
                "config": json.loads(config.read_text()), "dtype": str(model.dtype),
                "steps": args.steps, "layers": args.layers, "files": [p.name for p in expected],
                "prediction": prediction, "source_state": "GAS model inference, question decode",
                "note": "All comparison masks derive from the same GAS Q/K/V; not standalone Quest model states."}
    if not args.ruler_jsonl:
        manifest["max_prompt_length_before_chat"] = max_length
    if args.ruler_jsonl:
        manifest.update(source_jsonl=str(source.resolve()), input_tokens=input_token_count,
                        decode_protocol="teacher-forced final prompt tokens; not generated answers or RULER scoring")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"captured {len(expected)} trace files at {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
