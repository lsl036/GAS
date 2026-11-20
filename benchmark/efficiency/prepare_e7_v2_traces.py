"""Capture real dense LLaMA decode traces without enabling Twilight patches."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path("/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct")
DATA = ROOT / "benchmark/RULER/scripts/dataset/llama3-8b-chat-128k/synthetic/65536/data"


def hidden(args, kwargs):
    return kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]


def trace_path(output, length, sample, step, layer):
    return output / f"L{length}/sample{sample}/step{step}/layer{layer}.safetensors"


def save_tensors(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    save_file({name: value.detach().cpu().contiguous() for name, value in values.items()}, str(temporary))
    temporary.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[16384, 32768])
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--input-files", type=Path, nargs="+", default=[DATA / task / "validation.jsonl" for task in ("niah_single_1", "niah_multikey_3", "vt")])
    parser.add_argument("--layers", type=int, nargs="+", help="Smoke tests only; default captures every layer")
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = json.loads((args.model / "config.json").read_text())
    if cfg["model_type"] != "llama":
        raise ValueError("This protocol is frozen for the local LLaMA checkpoint")
    layers = args.layers if args.layers is not None else list(range(cfg["num_hidden_layers"]))
    if min(args.lengths + args.steps) < 1 or any(n % 16 for n in args.lengths):
        parser.error("lengths must be positive multiples of 16; steps must be positive")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    inputs, sources = [], []
    for file in args.input_files:
        with file.open() as stream:
            row = json.loads(next(stream))
        tokens = tokenizer(row["input"], add_special_tokens=False, return_tensors="pt").input_ids
        if tokens.shape[-1] < max(args.lengths):
            raise ValueError(f"{file}: source shorter than requested context; use existing longer RULER input")
        inputs.append(tokens)
        sources.append({"file": str(file.resolve()), "index": row.get("index"), "input_sha256": hashlib.sha256(row["input"].encode()).hexdigest(), "source_tokens": tokens.shape[-1]})
    manifest = {"protocol": "e7_real_dense_trace_v2", "model": str(args.model.resolve()), "model_config": cfg, "layers": layers, "lengths": args.lengths, "steps": args.steps, "sources": sources, "dtype": "float16", "attention_backend": "flash_attention_2", "crop": "prefix+suffix to N input tokens; recorded KV is last N positions at each decode step", "independent_source_sequences": len(inputs), "complete": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "trace_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in manifest:
            if key != "complete" and previous.get(key) != manifest[key]:
                raise ValueError(f"Trace protocol changed ({key}); use a fresh trace directory")
        if previous.get("complete") and all(trace_path(args.output_dir, n, sample, step, layer).exists() for n in args.lengths for sample in range(len(inputs)) for step in args.steps for layer in layers + ["output"]):
            print(f"reuse completed traces: {args.output_dir}", flush=True)
            return
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    model = AutoModelForCausalLM.from_pretrained(str(args.model), torch_dtype=torch.float16,
                                                attn_implementation="flash_attention_2", local_files_only=True).eval().cuda()
    context = {"step": 0, "payload": {}}
    handles = []

    def active():
        return context["step"] in args.steps

    for layer_id in layers:
        layer = model.model.layers[layer_id]

        def layer_pre(module, values, kwargs, index=layer_id):
            if active():
                context["payload"][index] = {"x": hidden(values, kwargs).squeeze(1).detach()}

        def o_pre(module, values, index=layer_id):
            if active():
                context["payload"][index]["attn_input"] = values[0].squeeze(1).detach()

        def attn_post(module, values, kwargs, output, index=layer_id):
            if not active():
                return
            from transformers.models.llama.modeling_llama import rotate_half
            value = hidden(values, kwargs)
            cos, sin = kwargs["position_embeddings"]
            query = module.q_proj(value).view(1, 1, cfg["num_attention_heads"], -1).transpose(1, 2)
            query = query * cos[:, None] + rotate_half(query) * sin[:, None]
            cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
            key, val = cache[index]
            context["payload"][index].update(q=query.squeeze(2), k=key[:, :, -context["length"]:], v=val[:, :, -context["length"]:], cos=cos.squeeze(1), sin=sin.squeeze(1))

        def layer_post(module, values, kwargs, output, index=layer_id):
            if active():
                payload = context["payload"].pop(index)
                payload["y"] = output.squeeze(1)
                save_tensors(trace_path(args.output_dir, context["length"], context["sample"], context["step"], index), payload)

        handles.extend([layer.register_forward_pre_hook(layer_pre, with_kwargs=True),
                        layer.self_attn.o_proj.register_forward_pre_hook(o_pre),
                        layer.self_attn.register_forward_hook(attn_post, with_kwargs=True),
                        layer.register_forward_hook(layer_post, with_kwargs=True)])

    def final_pre(module, values):
        if active():
            context["final_hidden"] = values[0].squeeze(1).detach()

    handles.append(model.model.norm.register_forward_pre_hook(final_pre))
    for length in args.lengths:
        for sample, tokens in enumerate(inputs):
            expected = [trace_path(args.output_dir, length, sample, step, layer) for step in args.steps for layer in layers + ["output"]]
            if all(file.exists() for file in expected):
                print(f"skip trace N={length} sample={sample}", flush=True)
                continue
            context.update(length=length, sample=sample, step=0)
            cropped = torch.cat((tokens[:, :length // 2], tokens[:, -(length - length // 2):]), -1).cuda()
            result = model(cropped, use_cache=True, logits_to_keep=1)
            cache = result.past_key_values
            token = result.logits[:, -1].argmax(-1, keepdim=True)
            del result
            for step in range(1, max(args.steps) + 1):
                context["step"] = step
                result = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                next_token = result.logits[:, -1].argmax(-1, keepdim=True)
                if active():
                    save_tensors(trace_path(args.output_dir, length, sample, step, "output"), {"x": context.pop("final_hidden"), "token": token, "next_token": next_token})
                    print(f"saved real trace N={length} sample={sample} step={step} layers={layers}", flush=True)
                token = next_token
                del result
            del cache, cropped
            context["step"] = 0
            torch.cuda.empty_cache()
    for handle in handles:
        handle.remove()
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"E7 v2 real trace preparation complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
