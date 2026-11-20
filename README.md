# GAS: Group-Aware Sparse Attention

Research implementation and experiment scripts for group-aware sparse attention in long-context LLM inference. This repository extends [Twilight](https://github.com/tsinghua-ideal/Twilight), based on upstream commit [`843ebcc61c23021f451364f199cc50368c3cff01`](https://github.com/tsinghua-ideal/Twilight/commit/843ebcc61c23021f451364f199cc50368c3cff01); the upstream README is preserved in [README_TWILIGHT.md](README_TWILIGHT.md). Original copyright notices and third-party attributions remain in the corresponding source files.

This is a research snapshot, not a production-ready inference engine. The model-side Python package is `gas_attention`; `csrc/` and `benchmark/efficiency/` also contain experimental CUDA and replay implementations. Standalone kernel and simulated TPOT results must not be presented as measured end-to-end inference speedups.

## Layout

- `gas_attention/`: attention patch, selectors, estimators, and group-aware refinement.
- `csrc/`: CUDA/C++ attention and refinement prototypes.
- `benchmark/`: Passkey, RULER, LongBench, PG19, and efficiency scripts and configurations.
- `Figures/`: existing paper figure notebooks, inputs, and presentation assets.
- `reproduction/`: frozen figure inputs, selected reference summaries, source snapshot, rendering and audit scripts; see [reproduction/README.md](reproduction/README.md).
- `flash-topk-attention/`: local snapshot of [Flash-TopK-Attention](https://github.com/tsinghua-ideal/flash-topk-attention), based on `d8803b29961c44d77a747636ad4282bd7a9094af` with local compatibility changes. Source headers and upstream attribution are preserved.

## Setup

Use a compatible Python, PyTorch, CUDA toolkit and GPU environment with the dependencies required by the selected experiment. For development from a full checkout:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

This command does not install dependencies. CUDA sources and external components are not fully bundled in a standalone wheel. Model checkpoints, benchmark datasets, large trace files, generated prediction logs, and third-party benchmark vendor builds are not included. Several research scripts and configurations contain original machine-specific paths; review and update them before running. See [reproduction/PIPELINES.md](reproduction/PIPELINES.md) and [reproduction/ENVIRONMENT.md](reproduction/ENVIRONMENT.md).

## Reproduction and provenance

Before using figure data in a paper, read [reproduction/PROVENANCE.md](reproduction/PROVENANCE.md) and the saved figure data audit: some existing figures combine measurement sources or differ from a formal results summary. The `reproduction/source_snapshot/` directory is an immutable historical snapshot from before the Python package rename and intentionally uses the `twilight` namespace.

```bash
python reproduction/figures.py verify
python reproduction/source_snapshot.py verify
python -m unittest discover -s reproduction/tests -v
```

Rendering the notebooks needs the dependencies listed in `reproduction/requirements-figures-verified.txt`; numerical figure regeneration is not the same as re-running the original model experiments. Historical local smoke-test outputs under `reproduction/runs/` are not included in this snapshot.
