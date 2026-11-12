"""
Fill the JCMM pre-review items (manuscript JCMM 261946) from a real run of liteformer_stack.py.

Reads   <results>/config_used.json  and  <results>/preds/<setting>/*.npz
Writes  <out>/LiteFormer-Stack_JCMM_Supplementary_Data.xlsx   (all sheets filled, nothing estimated)
        <out>/manuscript_text.txt                             (sentences for Sections 4.3, 5.1, 5.4, 5.6)

Usage
  python fill_editor_items.py --results results_out --out editor_items
  python fill_editor_items.py --results results_cic,results_nb --out editor_items     (two separate runs)
  python fill_editor_items.py --results smoke_out --out smoke_items --main smoke_random --device smoke_device   (self-test)
"""
import argparse, json, os, glob, re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment

PROP = "LiteFormer-Stack"
BASELINES = ["Random Forest", "XGBoost", "LightGBM", "MLP", "CNN-LSTM", "FT-Transformer only", "Soft-voting ensemble"]
ALL = BASELINES + ["Tree-only stack", "Full stack (ungated)", PROP]
# device numbering of the N-BaIoT CSV files (UCI release: 1.benign.csv ... 9.gafgyt.udp.csv)
NBAIOT = {1: "Danmini doorbell", 2: "Ecobee thermostat", 3: "Ennio doorbell", 4: "Philips B120N/10 baby monitor",
          5: "Provision PT-737E security camera", 6: "Provision PT-838 security camera", 7: "Samsung SNH-1011-N webcam",
          8: "SimpleHome XCS7-1002-WHT security camera", 9: "SimpleHome XCS7-1003-WHT security camera"}


def load_preds(results, setting):
    out = {}
    for f in [g for r in results.split(",") for g in glob.glob(os.path.join(r, "preds", setting, "*_run*.npz"))]:
        m = re.match(r"(.+)_run(\d+)\.npz$", os.path.basename(f))
        if m: out.setdefault(m.group(1), {})[int(m.group(2))] = np.load(f)
    return out


def sheet(wb, name, title, note, header, rows):
    ws = wb.create_sheet(name)
    ws["A1"] = title; ws["A1"].font = Font(bold=True, size=12)
    if note: ws["A2"] = note; ws["A2"].alignment = Alignment(wrap_text=True)
    for j, h in enumerate(header, 1): ws.cell(4, j, h).font = Font(bold=True)
    for i, r in enumerate(rows, 5):
        for j, v in enumerate(r, 1): ws.cell(i, j, v)
    for j in range(1, len(header) + 1): ws.column_dimensions[ws.cell(4, j).column_letter].width = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_out"); ap.add_argument("--out", default="editor_items")
    ap.add_argument("--main", default="cic34"); ap.add_argument("--device", default="nb3_device,nb11_device")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dirs = a.results.split(","); cfg = json.load(open(os.path.join(dirs[0], "config_used.json")))
    for dd in dirs[1:]:                      # results of separate invocations (e.g. CICIoT2023 and N-BaIoT) are merged
        cfg["settings"].update(json.load(open(os.path.join(dd, "config_used.json")))["settings"])
    st = cfg["settings"][a.main]; classes = st["classes"]; runs = st["runs"]; info = st.get("info") or {}
    wb = Workbook(); wb.remove(wb.active); text = []

    # 1. data and per-class split counts (first run; stratified splits differ only by rounding across runs)
    cc = runs[0]["class_counts"]; rows = []
    for i, c in enumerate(classes):
        tr, se, ca, te = cc["train"][i], cc["selection"][i], cc["calibration"][i], cc["test"][i]
        rows.append([c, tr, se, ca, te, tr + se + ca + te])
    rows.append(["TOTAL"] + [sum(r[j] for r in rows) for j in range(1, 6)])
    min_cal = min(cc["calibration"])
    sheet(wb, "class_counts", f"{a.main}: flows per class in each split (run 1)",
          f"Files: {info.get('files')} CSV files; rows in files: {info.get('rows_in_files')}; rows drawn: {info.get('rows_after_sampling')}; "
          f"removed NaN/inf: {info.get('removed_nan_inf')}; removed duplicates: {info.get('removed_duplicates')}; "
          f"rows after cleaning: {info.get('rows_after_cleaning')}. Smallest calibration class: {min_cal} flows.",
          ["Class", "Train", "Selection (D_sel)", "Calibration (D_cal)", "Test", "Total"], rows)
    names = info.get("file_names") or []
    text.append("Section 5.1 — " + (
        f"CICIoT2023: {info.get('files')} CSV files of the public release were used"
        + (f" ({names[0]} … {names[-1]})" if names else "")
        + f"; {info.get('rows_in_files'):,} rows were read, a stratified subset of {info.get('rows_after_sampling'):,} rows was drawn "
          f"(fraction {info.get('sampling_fraction')}, at least {info.get('min_per_class')} rows per class where available), "
          f"{info.get('removed_nan_inf'):,} rows with NaN/∞ values and {info.get('removed_duplicates'):,} duplicates were removed, "
          f"leaving {info.get('rows_after_cleaning'):,} flows. Per-class counts in the training, selection, calibration and test splits are "
          f"given in the Supplementary Data file (sheet class_counts); the smallest class has {min_cal} calibration flows."
        if info.get("rows_in_files") is not None else "CICIoT2023 file information not found in config_used.json"))

    # 2. latency protocol
    sheet(wb, "latency_protocol", "Latency timing protocol (as implemented in liteformer_stack.py)", "",
          ["Item", "Value"],
          [["Protocol", cfg.get("latency_protocol", "")], ["Flows timed per model", runs[0].get("latency_flows")],
           ["Threads", runs[0].get("threads")], ["CPU", cfg.get("cpu")], ["Platform", cfg.get("platform")], ["CPU count", cfg.get("cpu_count")]])
    text.append("Section 5.6 (timing) — " + cfg.get("latency_protocol", "") + ".")

    # 3. k, hyperparameters, versions
    hp = cfg["hyperparameters"]; hrows = []
    for model, v in hp.items():
        if isinstance(v, dict):
            for k_, val in v.items(): hrows.append([model, k_, str(val)])
        else: hrows.append(["general", model, str(v)])
    ks = st.get("k_per_run") or [r["k"] for r in runs]
    hrows.append(["feature selection", f"k selected per run ({a.main})", ", ".join(map(str, ks))])
    hrows.append(["FT-Transformer", "trainable parameters", st.get("ft_params")])
    hrows.append(["calibration", "temperature T (mean over runs)", round(st.get("T_mean", float("nan")), 4)])
    hrows.append(["calibration", "global threshold tau (mean)", round(st.get("tau_mean", float("nan")), 4)])
    sheet(wb, "hyperparameters", "Hyperparameters and selected values", "", ["Model", "Hyperparameter", "Value"], hrows)
    ver = cfg.get("versions", {})
    sheet(wb, "versions", "Software versions", "", ["Package", "Version"], [["python", cfg.get("python")]] + [[k_, v] for k_, v in ver.items()])
    text.append(f"Section 4.3 — The procedure retained k = {', '.join(map(str, sorted(set(ks))))} features on {a.main} "
                f"(per run: {', '.join(map(str, ks))}).")
    text.append("Section 5.6 (versions) — Python " + str(cfg.get("python")) + ", " +
                ", ".join(f"{k_} {v}" for k_, v in ver.items()) + ". Hyperparameters: Supplementary Data file, sheet hyperparameters.")

    # 4. N-BaIoT hold-out devices
    hrows = []
    for sname in a.device.split(","):
        if sname not in cfg["settings"]: continue
        dst = cfg["settings"][sname]; held = [r["held_out_device"] for r in dst["runs"]]
        devs = (dst.get("info") or {}).get("devices", [])
        hrows.append([sname, ", ".join(f"{d} ({NBAIOT.get(d, '?')})" for d in held), len(held), ", ".join(map(str, devs))])
        text.append(f"Section 5.4 ({sname}) — leave-one-device-out over {len(held)} devices: "
                    + "; ".join(f"{NBAIOT.get(d, d)}" for d in held) + f"; one run per held-out device ({len(held)} runs).")
    sheet(wb, "nbaiot_holdout", "N-BaIoT held-out-device protocol", "Device numbers follow the N-BaIoT CSV file names.",
          ["Setting", "Held-out device per run", "Runs", "Devices present"], hrows)

    # 5. per-run metrics and McNemar b, c (from the saved predictions)
    P = load_preds(a.results, a.main); R = sorted(P.get(PROP, {}).keys())
    prow = []
    for m in ALL:
        if m not in P: continue
        acc = [round(accuracy_score(P[m][r]["y_true"], P[m][r]["y_pred"]) * 100, 2) for r in R]
        f1 = [round(f1_score(P[m][r]["y_true"], P[m][r]["y_pred"], average="macro") * 100, 2) for r in R]
        prow.append([m] + acc + f1 + [round(float(np.mean(f1)), 2), round(float(np.std(f1, ddof=1)), 2) if len(f1) > 1 else None])
    sheet(wb, "per_run_metrics", f"Per-run accuracy and macro-F1 ({a.main})", "Mean and SD columns must equal Table 1.",
          ["Model"] + [f"Run {r} Acc" for r in R] + [f"Run {r} Macro-F1" for r in R] + ["Mean Macro-F1", "SD Macro-F1"], prow)
    mrow = []
    for m in BASELINES:
        if m not in P: continue
        b, c = [], []
        for r in R:
            y = P[PROP][r]["y_true"]; wp = P[PROP][r]["y_pred"] != y; wb_ = P[m][r]["y_pred"] != y
            b.append(int(np.sum(wb_ & ~wp))); c.append(int(np.sum(wp & ~wb_)))
        mrow.append([m] + b + c + [float(np.median(b)), float(np.median(c))])
    sheet(wb, "mcnemar_counts", f"McNemar counts per run ({a.main})",
          "b = flows misclassified only by the baseline; c = flows misclassified only by LiteFormer-Stack.",
          ["Baseline"] + [f"Run {r} b" for r in R] + [f"Run {r} c" for r in R] + ["Median b", "Median c"], mrow)

    wb.save(os.path.join(a.out, "LiteFormer-Stack_JCMM_Supplementary_Data.xlsx"))
    open(os.path.join(a.out, "manuscript_text.txt"), "w").write("\n\n".join(text) + "\n")
    print("\n\n".join(text)); print(f"\nWrote {a.out}/LiteFormer-Stack_JCMM_Supplementary_Data.xlsx and manuscript_text.txt")


if __name__ == "__main__":
    main()
