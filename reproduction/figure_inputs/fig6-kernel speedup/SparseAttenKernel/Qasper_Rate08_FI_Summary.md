# Qasper Quest 8%: Per-Head Versus Grouped FI

A100 40GB, two Qasper samples (15 and 54), 28 non-search layers 3-12 and 14-31, teacher-forced question token. The 32K input is repeated-context Qasper, not a natural 32K task. Source BF16 Q/K/V cast to FP16 before timing. Each batch alternates the two requests and shares two physical KV pools; it is not an independent-request batch.

Quest 8%, Twilight p=0.85 and GAS p=0.8 + exception16 use saved masks on the GAS trajectory; Less-style K=floor(0.08 * actual length), 25% recent, is reconstructed from search-layer GAS Q/K. It is not official LessIsMore model inference or kernel.

Values are arithmetic means across 28 layer medians; each median is from three round means of 50 calls, five warmups, using the author torch.utils.benchmark helper. Selection, mask construction, FI plan, layout and model work are excluded.

Per-head FI is a paged decode adapter; grouped FI is a paged prefill adapter with four query positions and exact per-head custom masks. They are different backend paths. They are not native Quest, Twilight, GAS or LessIsMore kernels and these data are not TPOT.

| Context | Batch | Quest head / group | Twilight head / group | GAS head / group | Less 8% head / group |
|---|---:|---:|---:|---:|---:|
| 16K | 32 | 152.76 / 186.92 | 95.36 / 115.22 | 95.34 / 114.86 | 153.39 / 116.34 |
| 16K | 64 | 285.50 / 295.90 | 94.42 / 115.09 | 94.50 / 114.34 | 286.12 / 174.62 |
| 16K | 96 | 430.58 / 412.67 | 98.39 / 122.18 | 97.95 / 116.68 | 429.49 / 242.67 |
| 16K | 128 | 578.72 / 538.05 | 104.64 / 132.95 | 99.16 / 116.19 | 582.47 / 311.65 |
| 32K repeated | 32 | 303.07 / 367.51 | 97.94 / 124.64 | 99.26 / 118.79 | 305.92 / 211.34 |
| 32K repeated | 64 | 636.26 / 593.90 | 106.16 / 139.19 | 100.83 / 117.87 | 596.03 / 343.09 |
| 32K repeated | 96 | 1037.28 / 838.58 | 123.73 / 158.85 | 108.12 / 120.27 | 920.41 / 461.57 |
| 32K repeated | 128 | 1454.91 / 1121.45 | 142.72 / 183.08 | 113.00 / 121.70 | 1250.39 / 594.16 |

## Logical Supports

| Context | Method | Tokens/head | Union/group |
|---|---|---:|---:|
| 16K | quest | 1307.59 | 2368.19 |
| 16K | twilight | 100.17 | 277.13 |
| 16K | gas | 72.19 | 92.97 |
| 16K | less_style_8pct | 1310.00 | 1310.00 |
| 32K repeated | quest | 2618.80 | 4692.43 |
| 32K repeated | twilight | 169.86 | 467.94 |
| 32K repeated | gas | 122.57 | 144.27 |
| 32K repeated | less_style_8pct | 2620.50 | 2620.50 |

Exact-mask per-head and grouped outputs were compared for all batch rows; FP32 references were checked on both distinct requests. Group union is a logical support count, not measured HBM traffic.
These eight shapes use only two samples and one position; 28 layer observations are correlated. Previous Fig.6 data used physically copied KV and CUDA Graph timing and cannot be directly merged.

Per-layer results and per-round means:
- `/data/home/lsl/SparseAtt/Twilight_Ours/benchmark/results_efficiency/qasper_rate08_grouped_16k_b32_20261113/results.csv`
- `/data/home/lsl/SparseAtt/Twilight_Ours/benchmark/results_efficiency/qasper_rate08_grouped_16k_b64_20261113/results.csv`
- `/data/home/lsl/SparseAtt/Twilight_Ours/benchmark/results_efficiency/qasper_rate08_grouped_16k_b96_20261113/results.csv`
- `/data/home/lsl/SparseAtt/Twilight_Ours/benchmark/results_efficiency/qasper_rate08_grouped_16k_b128_20261113/results.csv`
- `/data/home/lsl/SparseAtt/Twilight_Ours/benchmark/results_efficiency/qasper_rate08_grouped_32k_b32_20261113/results.csv`
- `/data/home/lsl/SparseAtt/Twilight_Ours/benchmark/results_efficiency/qasper_rate08_grouped_32k_b64_20261113/results.csv`
- `/data/home/lsl/SparseAtt/Twilight_Ours/benchmark/results_efficiency/qasper_rate08_grouped_32k_b96_20261113/results.csv`
- `/data/home/lsl/SparseAtt/Twilight_Ours/benchmark/results_efficiency/qasper_rate08_grouped_32k_b128_20261113/results.csv`



## Optimized Kernel

| Context | Batch | GAS-FI head | GAS-FI group | GAS-custom |
| ------- | ----- | ----------- | ------------ | ---------- |
| 16K     | 32    | 98.29       | 119.44       | **71.08**  |
| 16K     | 64    | 96.46       | 116.71       | **72.07**  |
| 16K     | 96    | 97.86       | 117.79       | **81.19**  |
| 16K     | 128   | 104.95      | 125.86       | **97.09**  |
| 32K     | 32    | 99.35       | 118.68       | **75.47**  |
| 32K     | 64    | 101.88      | 120.21       | **82.64**  |
| 32K     | 96    | 106.88      | 122.54       | **99.89**  |
| 32K     | 128   | 112.77      | 126.35       | 123.21     |
