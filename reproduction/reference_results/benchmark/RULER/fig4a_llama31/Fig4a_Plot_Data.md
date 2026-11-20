# Fig. 4(a) LLaMA Plot Data

Model: Meta-Llama-3.1-8B-Instruct; context: 32K; five RULER tasks; 250 formal samples per method.
The x-axis is mean group-level loaded KV and the y-axis is macro RULER score.
Full attention is a quality reference; GAS is the highlighted method.

| Method | Macro score | Mean loaded KV | KV ratio to GAS | Score delta vs GAS | Match status |
|---|---:|---:|---:|---:|---|
| Full attention | 60.58 | 259300.02 | 193.605 | -1.46 | reference |
| GAS | 62.04 | 1339.32 | 1.000 | +0.00 | reference |
| Quest (8%) | 56.42 | 37151.71 | 27.739 | -5.62 | formal |
| DS (8%) | 60.08 | 46935.32 | 35.044 | -1.96 | formal |
| Twilight (default) | 57.92 | 4600.00 | 3.435 | -4.12 | formal |
| Quest (KV-calibrated) | 22.14 | 1309.70 | 0.978 | -39.90 | in_range |
| DS (KV-calibrated) | 42.40 | 1757.92 | 1.313 | -19.64 | nearest_above |
| Twilight (KV-calibrated) | 46.76 | 1767.00 | 1.319 | -15.28 | nearest_above |

GAS reference: score=62.04, mean loaded KV=1339.32.
`in_range` means the calibration candidate was within 1.00--1.05 of the calibration GAS target.
`nearest_above` means no candidate was in range; the closest measured candidate above the target was selected.
