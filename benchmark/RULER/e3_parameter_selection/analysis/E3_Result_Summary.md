# E3 P0 Result Summary

## Status and scope

E3-P0 is complete. The experiment evaluates local sensitivity to the per-head
Twilight threshold `p_t` while fixing group-max `p_g=0.80`, exception budget
`m=16`, Quest budget 8192, chunk size 16, and the 4-bit fake min-max estimator.

- Model: LLaMA-3.1-8B-Instruct
- Hardware: NVIDIA A100-PCIE-40GB
- Context length: 65,536 tokens
- Tasks: `niah_multikey_3`, `niah_multivalue`, and `vt`
- Samples: 50 paired samples per task and configuration
- New evaluations: 300; reused E2 A3 evaluations: 150
- Bootstrap: 10,000 task-stratified paired replicates, seed 20260929
- Generation: batch size 1, temperature 0.0, top-k 32, top-p 1.0

P0-D reuses the complete E2 A3 predictions and synchronized KV records. P0-L
and P0-H use the same frozen E2 manifests, sample IDs, and order. All 450
configuration-sample pairs have scores and primary loaded-KV measurements.

## Macro results

| ID | `p_t` | `p_g` | `m` | Score | Loaded KV | Delta Q | Delta K | Delta Q 95% CI | Delta K 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0-L | 0.80 | 0.80 | 16 | 62.47 | 1193.75 | -3.23 | -11.39% | [-7.00, 0.27] | [-13.51%, -9.23%] |
| P0-D | 0.85 | 0.80 | 16 | 65.70 | 1347.13 | +0.00 | +0.00% | [0.00, 0.00] | [0.00%, 0.00%] |
| P0-H | 0.90 | 0.80 | 16 | 69.43 | 1516.08 | +3.73 | +12.54% | [-0.73, 8.20] | [9.62%, 15.25%] |

`Delta Q` and `Delta K` are relative to P0-D. Loaded KV is the mean number of
group-level unique token slots, not HBM bytes, KV-cache capacity, or latency.

## Task-level results

| Task | P0-L score / KV | P0-D score / KV | P0-H score / KV |
|---|---:|---:|---:|
| `niah_multikey_3` | 46.0 / 1669.95 | 54.0 / 1840.60 | 64.0 / 2060.93 |
| `niah_multivalue` | 95.0 / 1186.82 | 95.5 / 1387.39 | 95.5 / 1627.97 |
| `vt` | 46.4 / 724.47 | 47.6 / 813.39 | 48.8 / 859.33 |
| Equal-task macro | 62.47 / 1193.75 | 65.70 / 1347.13 | 69.43 / 1516.08 |

The mean quality movement is driven mainly by `niah_multikey_3`.
`niah_multivalue` is nearly saturated across this local range, while `vt`
changes only modestly.

## Selection conclusion

All three measured means are Pareto non-dominated: lowering `p_t` reduces both
quality and loaded KV, while raising it increases both. P0-D therefore remains
a defensible middle operating point, but this local sweep does not establish a
unique optimum.

- P0-L saves 11.39% loaded KV relative to P0-D. Its mean quality is 3.23 points
  lower, but the paired quality interval includes zero.
- P0-H uses 12.54% more loaded KV relative to P0-D. Its mean quality is 3.73
  points higher, but the paired quality interval also includes zero.
- The loaded-KV intervals exclude zero in both comparisons, so the direction
  of the load tradeoff is clear on this sample set.
- A quality confidence interval covering zero is inconclusive; it is not
  evidence of equivalence.

Neither frozen P1 trigger fires. P0-L does not improve both quality and KV over
P0-D, and P0-H does not have a strictly positive quality interval. A P2
confirmation run is unnecessary while retaining `p_t=0.85` as the default. If
the paper changes the default to P0-L or P0-H, that candidate must first be
frozen and evaluated on the independent P2 manifest.

![E3 quality-KV scatter](quality_kv_scatter.png)

## Output and delivery

1. Execution and configuration: `../run_e3.sh`, `../experiment_plan.json`, and
   `../configs/`. Exact commands, model/code hashes, GPU details, package
   versions, and manifest hashes are under `../metadata/`.
2. Raw predictions and synchronized KV records: `../pred/P0-L`,
   `../pred/P0-H`, and the read-only E2 A3 link `../pred/P0-D`.
3. Paired data: `paired_samples.csv` contains 450 rows with configuration,
   task, sample position/ID, score, loaded KV, and available diagnostics.
   `complement_count_mean` is unavailable and remains empty; no value was
   estimated.
4. Summary data: `summary.csv` contains 12 rows, comprising three task rows
   and one equal-task macro row for each configuration, with paired confidence
   intervals. `quality_kv_scatter.png` is the single paper-facing plot.

The repository was not clean at run time. Reproduction should therefore use
the recorded source/config SHA-256 hashes together with Git commit
`843ebcc61c23021f451364f199cc50368c3cff01`, rather than the commit alone.

## Paper-ready wording

> At 64K context, varying the per-head threshold from 0.80 to 0.90 produced a
> monotonic local quality-load curve: macro score increased from 62.47 to
> 69.43 while mean group-level loaded KV increased from 1193.75 to 1516.08.
> The default `p_t=0.85` attained 65.70 score at 1347.13 loaded KV and was not
> dominated by either neighboring point; paired quality intervals against both
> neighbors included zero, so we retain it as a balanced operating point rather
> than claiming a unique optimum.

