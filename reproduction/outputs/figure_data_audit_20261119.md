# Figure Data Audit

This audit compares the immutable figure-input snapshot with available saved summaries. A matching number is not proof of a connected, fair or representative experiment. The snapshot is not modified.

## Fig. 4: Snapshot Versus 32K Formal Summary

| Method | Snapshot score | Formal score | Snapshot KV | Formal KV |
|---|---:|---:|---:|---:|
| Full attention | 64.58 | 60.58 | 259300.00 | 259300.02 |
| DS KV-calibrated | 42.40 | 42.40 | 1458.00 | 1757.92 |
| Twilight KV-calibrated | 46.76 | 46.76 | 1367.00 | 1767.00 |

Comparison source: `benchmark/RULER/fig4a_llama31/fig4a_summary.csv`. Other curve points are not verified by this comparison.

## Fig. 6: GAS-FI Series Provenance

| Context | Batch | Snapshot GAS | Earlier FI head | Earlier FI group | Matches |
|---|---:|---:|---:|---:|---|
| 16K | 32 | 95.34 | 95.34 | 114.86 | head |
| 32K | 32 | 99.26 | 99.26 | 118.79 | head |
| 16K | 64 | 94.50 | 94.50 | 114.34 | head |
| 32K | 64 | 117.87 | 100.83 | 117.87 | group |
| 16K | 96 | 97.95 | 97.95 | 116.68 | head |
| 32K | 96 | 120.27 | 108.12 | 120.27 | group |
| 16K | 128 | 99.16 | 99.16 | 116.19 | head |
| 32K | 128 | 126.35 | 113.00 | 121.70 | neither |

Source: `benchmark/results_efficiency/qasper_rate08_grouped_summary_20261113/summary.csv`. The notebook also includes GAS-custom from a later paired run. Different source runs must not be silently presented as one paired measurement.

## Fig. 7: Projection Sources

Compare each displayed array with old grouped projection, tuned grouped projection, and custom projection at 0.011 ms tolerance.

| Array | Matches old | Matches tuned | Matches custom |
|---|---:|---:|---:|
| fa2 | 8/8 | 0/8 | 0/8 |
| flashinfer | 0/8 | 8/8 | 0/8 |
| quest | 8/8 | 0/8 | 0/8 |
| lessismore | 8/8 | 0/8 | 0/8 |
| quest_twi | 8/8 | 0/8 | 0/8 |
| gas | 0/8 | 0/8 | 8/8 |

Counts do not establish provenance by themselves; some values can coincide. Less-style uses `projected_step_ms` (30 sparse), not the separate search-aware column. The custom projection mixes separately measured common/dense and sparse components; it is not measured TPOT.

## Fig. 8: Explicit Scaling And Labels

- Snapshot contains `BREAKDOWN_MS[method][:, [0, 1, 2, 4]] *= 0.9`: **true**.
- The notebook labels `common_ms` as `Top-k Selector`; its adjacent README describes common transformer-layer computation, not a selector measurement.
- The backup notebook uses a different active synthetic component table than its `REPORTED_TOTAL_US` values. Its source comments alone do not prove those active values were measured.
- Treat these as manuscript-input snapshots only; do not present them as an audited measured pipeline or overwrite them without an explicit provenance decision.
