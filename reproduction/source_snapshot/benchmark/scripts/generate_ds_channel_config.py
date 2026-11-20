import argparse
import json
import os
from typing import Iterable, List

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TEXTS = [
    "Long context language models must retrieve precise facts from many irrelevant sentences.",
    "Sparse attention methods estimate which cached tokens are most useful for the next query.",
    "The calibration pass records activation magnitudes for every attention projection channel.",
    "This fallback text is only for smoke tests; use a real calibration corpus for final results.",
]


def iter_texts(args) -> Iterable[str]:
    if args.calib_text_file:
        with open(args.calib_text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
        return

    if args.dataset_name:
        from datasets import load_dataset

        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
        for row in dataset:
            text = row[args.text_field].strip()
            if text:
                yield text
        return

    while True:
        for text in DEFAULT_TEXTS:
            yield text


def build_blocks(tokenizer, args) -> List[torch.Tensor]:
    token_ids = []
    for text in iter_texts(args):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            token_ids.extend(ids)
        if len(token_ids) >= args.num_samples * args.block_size:
            break

    total = len(token_ids) // args.block_size * args.block_size
    token_ids = token_ids[:total]
    if total == 0:
        raise ValueError("No calibration tokens collected.")

    blocks = []
    for i in range(0, total, args.block_size):
        blocks.append(torch.tensor([token_ids[i : i + args.block_size]], dtype=torch.long))
    return blocks[: args.num_samples]


def register_channel_hooks(model, selected_channel: str, output_dict: dict):
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    if selected_channel == "q":
        head_count = config.num_attention_heads
        suffix = "q_proj"
    elif selected_channel == "k":
        head_count = config.num_key_value_heads
        suffix = "k_proj"
    else:
        raise ValueError("--selected-channel must be q or k")

    def hook(module, inputs, output, name):
        if isinstance(output, tuple):
            output = output[0]
        if output.shape[-1] != head_count * head_dim:
            raise ValueError(
                f"{name} has output dim {output.shape[-1]}, expected {head_count * head_dim}"
            )
        stat = output.reshape(-1, head_count, head_dim).abs().mean(dim=0).cpu()
        output_dict[name] = output_dict.get(name, 0) + stat

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(suffix):
            hooks.append(module.register_forward_hook(lambda m, x, y, n=name: hook(m, x, y, n)))
    if not hooks:
        raise ValueError(f"No modules ending with {suffix} found.")
    return hooks


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--selected-channel", choices=["q", "k"], default="q")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--dataset-name", default="mit-han-lab/pile-val-backup")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--calib-text-file")
    parser.add_argument("--fallback-text-only", action="store_true")
    args = parser.parse_args()

    if args.fallback_text_only:
        args.dataset_name = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    blocks = build_blocks(tokenizer, args)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    output_dict = {}
    hooks = register_channel_hooks(model, args.selected_channel, output_dict)
    device = next(model.parameters()).device

    for input_ids in tqdm(blocks, desc="Calibrating channel config"):
        model(input_ids.to(device), use_cache=False)

    for hook in hooks:
        hook.remove()

    channel_config = {}
    for name, stat in output_dict.items():
        _, indices = torch.sort(stat, dim=-1, descending=True)
        channel_config[name] = indices.tolist()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(channel_config, f)

    print(f"Saved {len(channel_config)} channel entries to {args.output_path}")


if __name__ == "__main__":
    main()
