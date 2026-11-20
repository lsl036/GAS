| Method                 | Macro score | Mean loaded KV | KV ratio to GAS | Score delta vs GAS | Status              |
| ---------------------- | ----------: | -------------: | --------------: | -----------------: | ------------------- |
| Full attention         |       64.58 |         259300 |        193.605× |              -1.46 | quality reference   |
| GAS                    |       62.04 |           1339 |          1.000× |               0.00 | main method         |
| Quest 8%               |       56.42 |          37152 |         27.739× |              -5.62 | selector baseline   |
| DS 8%                  |       60.08 |          46935 |         35.044× |              -1.96 | selector baseline   |
| Twilight default       |       57.92 |           4600 |          3.435× |              -4.12 | original refinement |
| Quest KV-calibrated    |       22.14 |           1310 |          0.978× |             -39.90 | in-range            |
| DS KV-calibrated       |       42.40 |           1458 |          1.313× |             -19.64 | nearest-above       |
| Twilight KV-calibrated |       46.76 |           1367 |          1.319× |             -15.28 | nearest-above       |

Model: Meta-Llama-3.1-8B-Instruct; context: 32K; five RULER tasks; 250 formal samples per method.
The x-axis is mean group-level loaded KV and the y-axis is macro RULER score.
Full attention is a quality reference; GAS is the highlighted method.



GAS reference: score=62.04, mean loaded KV=1339.32.
`in_range` means the calibration candidate was within 1.00--1.05 of the calibration GAS target.
`nearest_above` means no candidate was in range; the closest measured candidate above the target was selected.



Model: `Meta-Llama-3.1-8B-Instruct`; context length: 32K; tasks: `niah_single_1`, `niah_multikey_3`, `niah_multivalue`, `vt`, `cwe`; formal samples: 50 per task and 250 per method.



The GAS point has the highest score among the tested methods while using about 99.48% fewer loaded KV tokens than Full attention. The fixed Quest/DS 8% baselines and Twilight default use 27.7x, 35.0x, and 3.43x the GAS loaded KV, respectively.

The Quest calibrated point is within the formal KV interval but suffers a large quality loss. DS and Twilight had no calibration candidate within the 1.00--1.05 target interval; the nearest measured upper points were retained and labeled `nearest-above`. These points should not be described as exact KV matches.





## Quest 

| Method              | Macro score | Mean loaded KV | KV ratio to GAS | Score delta vs GAS | Status            |
| ------------------- | ----------: | -------------: | --------------: | -----------------: | ----------------- |
| Quest 8%            |       56.42 |          37152 |         27.739× |              -5.62 | selector baseline |
|                     |        52.5 |          19613 |                 |                    |                   |
|                     |        47.5 |          10049 |                 |                    |                   |
|                     |        36.3 |           5105 |                 |                    |                   |
|                     |        32.5 |           2575 |                 |                    |                   |
| Quest KV-calibrated |       22.14 |           1310 |          0.978× |             -39.90 | in-range          |



## DS

| Method           | Macro score | Mean loaded KV | KV ratio to GAS | Score delta vs GAS | Status            |
| ---------------- | ----------: | -------------: | --------------: | -----------------: | ----------------- |
| DS 8%            |       60.08 |          46935 |         35.044× |              -1.96 | selector baseline |
|                  |        59.8 |          25430 |                 |                    |                   |
|                  |        57.2 |          13463 |                 |                    |                   |
|                  |        54.2 |           6975 |                 |                    |                   |
|                  |        46.4 |           3539 |                 |                    |                   |
| DS KV-calibrated |       42.40 |           1458 |          1.313× |             -19.64 | nearest-above     |



## Twilight

| Method                 | Macro score | Mean loaded KV | KV ratio to GAS | Score delta vs GAS | Status              |
| ---------------------- | ----------: | -------------: | --------------: | -----------------: | ------------------- |
| Twilight default       |       57.92 |           4600 |          3.435× |              -4.12 | original refinement |
|                        |        55.9 |           3222 |                 |                    |                     |
|                        |        52.1 |           2702 |                 |                    |                     |
|                        |        48.7 |           1721 |                 |                    |                     |
| Twilight KV-calibrated |       46.76 |           1367 |          1.319× |             -15.28 | nearest-above       |
|                        |       40.01 |           1161 |                 |                    |                     |

## GAS

| Method | Macro score | Mean loaded KV | KV ratio to GAS | Score delta vs GAS | Status      |
| ------ | ----------: | -------------: | --------------: | -----------------: | ----------- |
|        |       66.17 |           1813 |                 |                    |             |
|        |       64.99 |           1535 |                 |                    |             |
| GAS    |       62.04 |           1339 |          1.000× |               0.00 | main method |
|        |       60.44 |           1207 |                 |                    |             |
|        |       59.65 |           1093 |                 |                    |             |



## LessIsMore

| Method | Macro score | Mean loaded KV | KV ratio to GAS | Score delta vs GAS | Status |
| ------ | ----------: | -------------: | --------------: | -----------------: | ------ |
| K=128  |       36.80 |           1024 |                 |                    |        |
| K=168  |       37.90 |           1344 |                 |                    |        |
| K=256  |       41.00 |           2048 |                 |                    |        |
| K=384  |       47.60 |           3072 |                 |                    |        |
| K=512  |       51.40 |           4096 |                 |                    |        |
