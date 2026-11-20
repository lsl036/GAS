# Qasper 8% TPOT Component Projection: Tuned Dense FI

Disconnected component projection, not measured model or serving TPOT. Two Qasper samples are repeated across each batch with two shared physical KV pools; the 32K workload repeats context.

Dense FA2 and FI are remeasured at representative layer 15 on the same Q/K/V layout. The main view consistently uses FI page_size=16 and use_tensor_cores=True across all shapes; page_size=1 with tensor cores is a fixed alternative, not a per-shape fastest selection. The local FI 0.6.11.post1 resolves its FP16 tensor-core path to its `fa2` backend; this is not necessarily an independent algorithm from the separately timed FA2 interface.

Dense formula: historical 32-layer common + LM head + 32 * measured dense attention. Sparse formula: historical common + LM head + 2 * tuned dense FI + 30 * sparse attention. Sparse attention is the average of 28 non-search layer medians and is extrapolated to 30 layers. For Less the table uses 4 * dense FI + 28 * sparse attention to account for two additional search layers; the older uniform 2+30 approximation remains in the CSV.

Selection, estimators, top-p, GAS refinement, Less search, metadata, packing and FI plan are excluded. Common components are old random FP16 matrix CUDA-event timings; attention is measured in separate torch.utils.benchmark runs. This sum is not a measured TPOT or a guaranteed bound.

All sparse paths are FI adapters, not official native Quest, Twilight, GAS or LessIsMore kernels. Less-style K is floor(8% of each actual context) and its support is reconstructed from GAS search states.

## Sparse FI: grouped; dense FI: page16 + TC

| Context | Batch | FA2 | FI dense | Quest | Twilight | GAS | Less, 28 sparse + 4 dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16K | 32 | 51.905 | 42.607 | 21.881 | 19.730 | 19.720 | 21.287 |
| 16K | 64 | 75.859 | 58.931 | 26.360 | 20.936 | 20.913 | 25.136 |
| 16K | 96 | 108.622 | 78.385 | 33.821 | 25.107 | 24.942 | 32.032 |
| 16K | 128 | 134.372 | 96.003 | 39.168 | 27.015 | 26.512 | 36.617 |
| 32K repeated | 32 | 88.799 | 67.555 | 28.902 | 21.616 | 21.440 | 27.106 |
| 32K repeated | 64 | 139.691 | 101.515 | 38.360 | 24.719 | 24.079 | 35.548 |
| 32K repeated | 96 | 202.762 | 138.953 | 50.487 | 30.095 | 28.937 | 45.828 |
| 32K repeated | 128 | 256.335 | 174.151 | 63.269 | 35.118 | 33.277 | 55.897 |

## Sparse FI: per_head; dense FI: page16 + TC

| Context | Batch | FA2 | FI dense | Quest | Twilight | GAS | Less, 28 sparse + 4 dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16K | 32 | 51.905 | 42.607 | 20.856 | 19.134 | 19.134 | 22.324 |
| 16K | 64 | 75.859 | 58.931 | 26.048 | 20.315 | 20.318 | 28.258 |
| 16K | 96 | 108.622 | 78.385 | 34.359 | 24.393 | 24.380 | 37.263 |
| 16K | 128 | 134.372 | 96.003 | 40.388 | 26.165 | 26.001 | 44.200 |
| 32K repeated | 32 | 88.799 | 67.555 | 26.969 | 20.815 | 20.854 | 29.755 |
| 32K repeated | 64 | 139.691 | 101.515 | 39.631 | 23.728 | 23.568 | 42.630 |
| 32K repeated | 96 | 202.762 | 138.953 | 56.448 | 29.041 | 28.573 | 58.676 |
| 32K repeated | 128 | 256.335 | 174.151 | 73.273 | 33.907 | 33.016 | 74.272 |

Units are ms per batch step, not divided by batch size. All per-round dense measurements, page/tensor-core settings and errors are in `dense_audit.csv`; the full alternative projection is in `projection.csv`. 32K/B128 common costs come from the available Quest historical row because the historical dense rows were unavailable.

## Supplement: Paired GAS-Custom Attention Projection (2026-11-14)

This supplement preserves the earlier tables. It substitutes the newly paired GAS attention-call measurements into the **same** sparse projection formula: historical 32-layer common work + historical LM head + 2 tuned dense FI layers + 30 sparse layers. The original 28-layer Qasper mean is extrapolated to 30 sparse layers. The FI head, FI group and GAS-custom sparse inputs are the 28-layer arithmetic means from the same new randomized paired author-timer run, not numbers spliced across the old and new attention runs.

| Context | Batch | GAS-FI head | GAS-FI group | GAS-custom | FI group / custom |
|---|---:|---:|---:|---:|---:|
| 16K | 32 | 19.222 | 19.857 | 18.406 | 1.079x |
| 16K | 64 | 20.377 | 20.984 | 19.645 | 1.068x |
| 16K | 96 | 24.377 | 24.975 | 23.877 | 1.046x |
| 16K | 128 | 26.175 | 26.802 | 25.939 | 1.033x |
| 32K repeated | 32 | 20.857 | 21.437 | 20.141 | 1.064x |
| 32K repeated | 64 | 23.599 | 24.149 | 23.022 | 1.049x |
| 32K repeated | 96 | 28.536 | 29.006 | 28.326 | 1.024x |
| 32K repeated | 128 | 33.009 | 33.416 | 33.322 | 1.003x |

Values are **ms per batch decode step**, not per-request throughput or a measured TPOT. FI group / custom is a ratio of these component-sum projections; at 32K/B128 the FI head projection is lower than custom. The machine-readable [gas_custom_projection.csv](gas_custom_projection.csv) records inputs, original FI-group projection, paired replacements, source files and unrounded values. Recompute with `benchmark/efficiency/summarize_qasper_gas_custom_tpot.py` from the `Twilight_Ours` root.

**Interpretation and limitations:** The common and dense components were measured separately from attention, and the common random-matrix components use CUDA-event timings whereas the new sparse attention inputs use `torch.utils.benchmark`, including host/framework dispatch. Quest selection, estimation, top-p, GAS refinement, dynamic support preparation and FI planning are excluded. The two Qasper requests alternate and share two physical KV pools; 32K is repeated context. This disconnected component sum is not measured end-to-end model/serving latency or a guaranteed bound. The 32K/B128 historical common term is sourced from the Quest row, not a new common measurement.

The custom implementation and FI adapters are different backends. In a separate [CUDA Graph event replay](../qasper_gas_paired_20261114/summary.md) of the same captured masks, custom attention was **slower** than grouped FI in all eight shapes. Do not infer a consistent GPU-kernel speedup from this author-timer projection. See the [paired author-timer attention results](../qasper_gas_author_20261114/summary.md) and [technical handoff](../../../../thoughts/Efficiency/GAS_Custom_Fused_Kernel_Technical_Handoff_20261114.md).
