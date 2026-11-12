# LiteFormer-Cascade

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24166636.svg)](https://doi.org/10.5281/zenodo.24166636)

Code, run configurations and saved test predictions for

> S. Y. Bobade, R. S. Apare, P. N. Mahalle, H. Joshi. *LiteFormer-Cascade: A Calibrated, Class-Aware Confidence Cascade for Cost-Efficient Fine-Grained IoT Botnet Detection.* Journal of Computers, Mechanical and Management (manuscript 261946, revised).

## Contents

| Path | What it is |
|---|---|
| `code/liteformer_stack.py` | The one script that trains every model in every setting and writes every number in Tables 1–4 (see `code/RUNNING.md`) |
| `code/fill_editor_items.py` | Builds the Supplementary Data workbook from the outputs |
| `analysis/revision_analyses.py` | Tables 5 and 6 (tolerance sweep, benign false positives) from the saved predictions only |
| `analysis/out/` | Outputs of `revision_analyses.py` |
| `results/<setting>/config_used.json` | Hyperparameters, software versions, CPU, latency protocol, and for every run: seed, selected k, temperatures, global and per-class thresholds, fast-path fractions, measured component latencies, class counts per split |
| `results/<setting>/summary_*.csv, stats_*.csv, per_class_recall_*.csv` | Tables 1–2 and per-class recall |
| `results/<setting>/preds/*_run<r>.npz` | Saved test predictions of every model and run: `y_true`, `y_pred`; the cascade files also hold the temperature-scaled XGBoost probabilities of every test flow (`p_xgb_cal`) and the routing decision (`routed_full`) |

Settings: `cic34` (CICIoT2023, 34 classes, 10 random splits), `nb3_random`, `nb3_device`, `nb11_random`, `nb11_device` (N-BaIoT, 3 and 11 classes, 10 random splits or 9 leave-one-device-out runs).

## Reproduce Tables 5 and 6 (minutes, no training)

    pip install numpy pandas scikit-learn scipy
    python analysis/revision_analyses.py results/cic34 analysis/out

The script first checks that every saved cascade prediction equals the XGBoost label (fast path) or the full-stack label (routed), then runs the tolerance sweep by two-fold cross-fitting on the test flows and the benign false-positive breakdown.

## Reproduce everything (hours, needs the datasets)

See `code/RUNNING.md`. Datasets: CICIoT2023 (https://www.unb.ca/cic/datasets/iotdataset-2023.html) and N-BaIoT (UCI Machine Learning Repository); they are not redistributed here.

## Licence

MIT (see `LICENSE`). Cite the paper and the Zenodo DOI when you use this material.

## What is in this repository and what is on Zenodo

This repository holds the code, the CICIoT2023 run configuration (`results/cic34/config_used.json`), the CICIoT2023 test predictions of the cascade and the full stack for all 10 runs (`results/cic34/preds/cic34/`, enough to rebuild Tables 5–7), and the Supplementary Data workbook (`supplementary/`). The complete result folders of all settings (predictions of every model, CICIoT2023 and N-BaIoT) are archived with the Zenodo release of this repository.
