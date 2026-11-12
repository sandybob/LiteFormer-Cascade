"""Revision analyses for LiteFormer-Cascade (JCMM 261946), computed only from the saved test predictions.

Inputs  (one run = one random 70/15/15 split of the CIC-IoT2023 34-class sample)
  preds/LiteFormer-Cascade_run<r>.npz   y_true, y_pred, p_xgb_cal (temperature-scaled XGBoost probabilities
                                        on the TEST flows, despite the name), routed_full
  preds/Full stack (ungated)_run<r>.npz y_true, y_pred (full stack)
  config_used.json                      per-run latencies (t_xgb_us, t_T_us, t_F_us), thresholds, classes

Outputs (csv + printed tables)
  tolerance_sweep.csv      comment 3: macro-F1, fast-path share and Eq.7 latency for several tolerance settings
  tolerance_per_run.csv
  benign_fp_breakdown.csv  comment 4: benign flows misclassified, by predicted class (cascade and XGBoost)
  benign_reweighted.csv    comment 4: alert-level figures at test and release class proportions

Tolerance sweep: the calibration-split probabilities were not saved, so the thresholds for each tolerance
setting are chosen by 2-fold cross-fitting on the test flows (choose on one stratified half, apply to the
other half, then swap), with exactly the selection rule of the paper (choose_thresholds below is copied
from liteformer_stack.py). No flow is ever scored with thresholds chosen on itself.
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

D = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
cfg = json.load(open(os.path.join(D, "config_used.json")))
S = cfg["settings"]["cic34"]; CL = S["classes"]; K = len(CL); BEN = CL.index("BenignTraffic")
GRID = cfg["hyperparameters"]["tau_grid"]; MINC = cfg["hyperparameters"]["min_cal_per_class"]
RUNS = {r["run"]: r for r in S["runs"]}
PD = os.path.join(D, "preds", "cic34") if os.path.isdir(os.path.join(D, "preds", "cic34")) else os.path.join(D, "preds")

def choose_thresholds(pT, yhat_T, yhat_F, y, f1_tol, err_tol):
    conf = pT.max(1); f_full = f1_score(y, yhat_F, average="macro") * 100
    ok = [abs(f1_score(y, np.where(conf >= t, yhat_T, yhat_F), average="macro") * 100 - f_full) <= f1_tol for t in GRID]
    tau = next((t for i, t in enumerate(GRID) if all(ok[i:])), 1.01)
    tau_c = {}
    for c in range(K):
        mask = yhat_T == c
        if mask.sum() < MINC: tau_c[c] = 1.01; continue
        cand = [t for t in GRID if t >= tau]; good = []
        for t in cand:
            acc = mask & (conf >= t)
            good.append(True if acc.sum() == 0 else (np.mean(yhat_T[acc] != y[acc]) - np.mean(yhat_F[acc] != y[acc]) <= err_tol))
        tau_c[c] = next((t for i, t in enumerate(cand) if all(good[i:])), 1.01)
    return tau, tau_c

def load(r):
    c = np.load(os.path.join(PD, f"LiteFormer-Cascade_run{r}.npz"))
    f = np.load(os.path.join(PD, f"Full stack (ungated)_run{r}.npz"))
    assert (c["y_true"] == f["y_true"]).all()
    y, pX, yF = c["y_true"], c["p_xgb_cal"], f["y_pred"]; yX = pX.argmax(1)
    # consistency: routed flows of the saved cascade must carry the full-stack label, fast flows the XGBoost label
    rt = c["routed_full"]
    assert (c["y_pred"][rt] == yF[rt]).all() and (c["y_pred"][~rt] == yX[~rt]).all()
    return y, pX, yX, yF, c["y_pred"], rt

def lat(r, rho):
    rec = RUNS[r]; tX = rec["cascade"]["t_xgb_us"]; tFull = rec["t_T_us"] + rec["t_F_us"]
    return tX + (1 - rho) * (tFull - tX)                     # Eq. 7: E[t] = t_X + (1 - rho') t_R

SETTINGS = [("0.00", 0.0, 0.005), ("0.05", 0.05, 0.005), ("0.10 (paper)", 0.1, 0.005), ("0.20", 0.2, 0.005),
            ("0.30", 0.3, 0.005), ("0.50", 0.5, 0.005), ("1.00", 1.0, 0.005),
            ("0.10, eps=0.25", 0.1, 0.0025), ("0.10, eps=1.0", 0.1, 0.01), ("0.10, no per-class check", 0.1, 1.0)]

rows, per = [], []
data = {r: load(r) for r in sorted(RUNS)}
for name, ft, et in SETTINGS:
    for r, (y, pX, yX, yF, yC, rt) in data.items():
        yp = np.empty_like(y); fast = np.zeros(len(y), bool); fb = 0
        for a, b in StratifiedKFold(2, shuffle=True, random_state=1000 + r).split(pX, y):
            tau, tc = choose_thresholds(pX[a], yX[a], yF[a], y[a], ft, et)
            fb += int(all(v > 1.0 for v in tc.values()))
            fa = pX[b].max(1) >= np.array([tc[c] for c in yX[b]])
            yp[b] = np.where(fa, yX[b], yF[b]); fast[b] = fa
        rho = fast.mean()
        per.append(dict(setting=name, run=r, macro_f1=f1_score(y, yp, average="macro") * 100,
                        full_f1=f1_score(y, yF, average="macro") * 100, rho=rho * 100, latency_us=lat(r, rho),
                        halves_all_full=fb, benign_fpr=np.mean(yp[y == BEN] != BEN) * 100))
P = pd.DataFrame(per); P["delta"] = P.macro_f1 - P.full_f1
# reference rows from the saved runs
ref = []
for r, (y, pX, yX, yF, yC, rt) in data.items():
    ref.append(dict(setting="Deployed (D_cal, paper)", run=r, macro_f1=f1_score(y, yC, average="macro") * 100,
                    full_f1=f1_score(y, yF, average="macro") * 100, rho=(1 - rt.mean()) * 100, latency_us=lat(r, 1 - rt.mean()),
                    halves_all_full=int(rt.all()), benign_fpr=np.mean(yC[y == BEN] != BEN) * 100))
    ref.append(dict(setting="XGBoost alone", run=r, macro_f1=f1_score(y, yX, average="macro") * 100,
                    full_f1=f1_score(y, yF, average="macro") * 100, rho=100.0, latency_us=RUNS[r]["cascade"]["t_xgb_us"],
                    halves_all_full=0, benign_fpr=np.mean(yX[y == BEN] != BEN) * 100))
    ref.append(dict(setting="Full stack alone", run=r, macro_f1=f1_score(y, yF, average="macro") * 100,
                    full_f1=f1_score(y, yF, average="macro") * 100, rho=0.0, latency_us=RUNS[r]["t_T_us"] + RUNS[r]["t_F_us"],
                    halves_all_full=0, benign_fpr=np.mean(yF[y == BEN] != BEN) * 100))
R = pd.DataFrame(ref); R["delta"] = R.macro_f1 - R.full_f1
ALL = pd.concat([R, P]); ALL.to_csv(os.path.join(OUT, "tolerance_per_run.csv"), index=False)
order = ["XGBoost alone", "Deployed (D_cal, paper)"] + [s[0] for s in SETTINGS] + ["Full stack alone"]
g = ALL.groupby("setting")
T = pd.DataFrame({"macro_f1_mean": g.macro_f1.mean(), "macro_f1_sd": g.macro_f1.std(), "delta_vs_full": g.delta.mean(),
                  "rho_mean": g.rho.mean(), "rho_min": g.rho.min(), "rho_max": g.rho.max(), "latency_us": g.latency_us.mean(),
                  "x_xgb_cost": g.latency_us.mean() / ALL[ALL.setting == "XGBoost alone"].latency_us.mean(),
                  "halves_all_full": g.halves_all_full.sum(), "benign_fpr": g.benign_fpr.mean()}).loc[order]
T.round(3).to_csv(os.path.join(OUT, "tolerance_sweep.csv"))
pd.set_option("display.width", 200); print(T.round(2))
# run 10 (fall-back run) under each setting
print("\nrun 10 (fell back under D_cal):"); print(ALL[ALL.run == 10].set_index("setting").loc[order, ["macro_f1", "rho", "latency_us"]].round(2))

# ---------------------------------------------------------------- comment 4: benign false positives
REL = {"DDoS-ICMP_Flood": 7200504, "DDoS-UDP_Flood": 5412287, "DDoS-TCP_Flood": 4497667, "DDoS-PSHACK_Flood": 4094755,
       "DDoS-SYN_Flood": 4059190, "DDoS-RSTFINFlood": 4045285, "DDoS-SynonymousIP_Flood": 3598138, "DoS-UDP_Flood": 3318595,
       "DoS-TCP_Flood": 2671445, "DoS-SYN_Flood": 2028834, "BenignTraffic": 1098195, "Mirai-greeth_flood": 991866,
       "Mirai-udpplain": 890576, "Mirai-greip_flood": 751682, "DDoS-ICMP_Fragmentation": 452489, "MITM-ArpSpoofing": 307593,
       "DDoS-UDP_Fragmentation": 286925, "DDoS-ACK_Fragmentation": 285104, "DNS_Spoofing": 178911, "Recon-HostDiscovery": 134378,
       "Recon-OSScan": 98259, "Recon-PortScan": 82284, "DoS-HTTP_Flood": 71864, "VulnerabilityScan": 37382, "DDoS-HTTP_Flood": 28790,
       "DDoS-SlowLoris": 23426, "DictionaryBruteForce": 13064, "BrowserHijacking": 5859, "CommandInjection": 5409,
       "SqlInjection": 5245, "XSS": 3846, "Backdoor_Malware": 3218, "Recon-PingSweep": 2262, "Uploading_Attack": 1252}
assert set(REL) == set(CL), set(REL) ^ set(CL)
print("\nrelease total:", sum(REL.values()))
rel = np.array([REL[c] for c in CL], float)

def breakdown(label, getter):
    cnt = np.zeros(K); nb = 0
    for r, d in data.items():
        y, yp = d[0], getter(d); b = y == BEN; nb += b.sum(); cnt += np.bincount(yp[b], minlength=K)
    cnt[BEN] = 0
    df = pd.DataFrame({"predicted_class": CL, "benign_flows": cnt.astype(int), "share_of_benign_%": cnt / nb * 100,
                       "share_of_false_alarms_%": cnt / cnt.sum() * 100})
    df = df[df.benign_flows > 0].sort_values("benign_flows", ascending=False); df.insert(0, "model", label)
    return df, nb
bc, nb = breakdown("LiteFormer-Cascade", lambda d: d[4]); bx, _ = breakdown("XGBoost", lambda d: d[2]); bf, _ = breakdown("Full stack", lambda d: d[3])
B = pd.concat([bc, bx, bf]); B.round(3).to_csv(os.path.join(OUT, "benign_fp_breakdown.csv"), index=False)
print(f"\nbenign test flows over 10 runs: {nb}"); print(bc.round(2).to_string(index=False))
# family view
fam = lambda c: c.split("-")[0] if "-" in c else ("Web" if c in ("SqlInjection", "XSS", "CommandInjection", "Uploading_Attack",
      "BrowserHijacking", "Backdoor_Malware") else "BruteForce" if c == "DictionaryBruteForce" else "Spoofing" if c in ("DNS_Spoofing",) else c)
print(bc.assign(family=bc.predicted_class.map(fam)).groupby("family")["share_of_false_alarms_%"].sum().sort_values(ascending=False).round(1))

# alert-level figures at test and release proportions
def alert_stats(getter, label):
    out = []
    for r, d in data.items():
        y, yp = d[0], getter(d); n_t = np.bincount(y, minlength=K).astype(float)
        for wname, w in (("test", np.ones(K)), ("release", rel / rel.sum() * len(y) / n_t)):
            wy = w[y]; alarm = yp != BEN; isben = y == BEN
            fp = (wy * (alarm & isben)).sum(); tp = (wy * (alarm & ~isben)).sum(); fn = (wy * (~alarm & ~isben)).sum()
            tot = wy.sum(); benign_share = (wy * isben).sum() / tot
            # reweighted multi-class figures
            acc = (wy * (yp == y)).sum() / tot
            out.append(dict(model=label, run=r, proportions=wname, benign_share_pct=benign_share * 100,
                            benign_fpr_pct=fp / (wy * isben).sum() * 100, false_alarm_share_pct=fp / (fp + tp) * 100,
                            false_alarms_per_million_flows=fp / tot * 1e6, attack_detection_rate_pct=tp / (tp + fn) * 100,
                            missed_attacks_per_million_flows=fn / tot * 1e6, weighted_accuracy_pct=acc * 100))
    return out
A = pd.DataFrame(alert_stats(lambda d: d[4], "LiteFormer-Cascade") + alert_stats(lambda d: d[2], "XGBoost") + alert_stats(lambda d: d[3], "Full stack"))
AS = A.groupby(["model", "proportions"]).mean(numeric_only=True).drop(columns="run")
AS.round(4).to_csv(os.path.join(OUT, "benign_reweighted.csv")); print(); print(AS.round(3))
