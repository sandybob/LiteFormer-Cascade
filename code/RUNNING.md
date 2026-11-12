# LiteFormer-Stack experiment code

One script trains LiteFormer-Stack and all baselines in every setting and writes every number the paper needs.

## 1. Install (once)
```
pip install -r requirements.txt
```

## 2. Self-test (about 2 minutes, synthetic data)
```
python liteformer_stack.py --smoke
```
If this finishes with "Done.", the installation works. Smoke-test numbers are meaningless; never use them in the paper.

## 3. Put the datasets in place
```
data/CICIoT2023/   all CSV files of the CICIoT2023 release (columns: 46 features + "label")
data/N-BaIoT/      files named like 1.benign.csv, 1.mirai.ack.csv, 1.gafgyt.combo.csv ... (Kaggle layout, devices 1–9)
```

## 4. Run everything (10 runs per setting)
```
python liteformer_stack.py --cic-dir data/CICIoT2023 --nbaiot-dir data/N-BaIoT --out results_out
```
Useful options:
- `--cic-frac 0.02`: fraction of CICIoT2023 used (stratified; `--min-per-class 2000` keeps rare attacks)
- `--nbaiot-frac 0.10`: fraction of N-BaIoT used
- `--settings cic34,nb3_device`: run only some settings (all 7: cic34, cic8, cic2, nb3_random, nb3_device, nb11_random, nb11_device)
- `--runs 10`: runs per random-split setting (device settings use leave-one-device-out, one run per device)

Start with `--settings cic34 --runs 2` to see how long one run takes on your machine.

## 5. Send `results_out/` back
It contains:
| File | Goes into the paper as |
|---|---|
| summary_<setting>.csv | Table 1 (and the same layout for every setting); includes benign false-positive rate and two latency columns |
| stats_<setting>.csv | Table 2 (McNemar, Holm, Wilcoxon, Cliff's delta) |
| per_class_recall_<setting>.csv | new per-class recall table / figure |
| friedman.csv, friedman_stats.json | Supplementary Table S2: the eight settings and their ranks |
| config_used.json | Methods values: k, transformer parameter count, T, tau, per-class tau, rho, sample sizes, hold-out protocol, hyperparameters, CPU, versions, latency protocol |
| preds/ | raw predictions (for any later check) |

## Latency
Two columns are reported. `latency_us` is one flow per call (batch size 1), single thread, which is realistic for flow-by-flow detection. `latency_batched_us` is the per-flow cost when 500 flows are scored in one call. Python call overhead makes batch-size-1 latencies much larger than batched ones, so the paper must say which one it reports.

## Important
If the paper's results are regenerated with this script, the paper must report these new numbers, and Methods must describe the settings in config_used.json.

## Editor items (JCMM 261946)

After the runs, fill the editor's pre-review items (per-class split counts, k, hyperparameters, versions,
latency protocol, held-out devices, per-run metrics and McNemar b/c counts) directly from the outputs:

    python fill_editor_items.py --results results_cic,results_nb --out editor_items

This writes editor_items/LiteFormer-Stack_JCMM_Supplementary_Data.xlsx and editor_items/manuscript_text.txt.
Nothing in these files is estimated: every value is read from config_used.json and the saved predictions.
