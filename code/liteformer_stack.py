"""
LiteFormer-Stack — full experiment pipeline (paper Sections 4–5).

One command trains every model in every setting, R runs each, and writes ALL numbers
the paper needs into an output folder:

  summary_<setting>.csv        Table 1 style: accuracy, macro-F1 mean/std/corrected-CI, MCC, kappa,
                               benign false-positive rate, per-flow latency (µs) for every model
  per_class_recall_<setting>.csv   per-class recall (mean over runs) for every model
  stats_<setting>.csv          Table 2: McNemar (median chi2, median p, runs p<0.05), Holm-adjusted
                               Wilcoxon p, Cliff's delta, vs LiteFormer-Stack
  friedman.csv                 mean macro-F1 per model per setting, average ranks, Friedman,
                               Iman-Davenport, Nemenyi CD (Supplementary Table S2)
  config_used.json             every value the paper reports in Methods: k, transformer parameter
                               count, T, tau, per-class tau, rho, t_T, t_F, sample sizes, hold-out
                               protocol, hyperparameters, CPU, library versions
  preds/<setting>/<model>_run<r>.npz   y_true, y_pred (+ tree-path probabilities for the gate)

Usage
  python liteformer_stack.py --cic-dir data/CICIoT2023 --nbaiot-dir data/N-BaIoT --out results_out
  python liteformer_stack.py --smoke            # 2-minute self-test on synthetic data

v3 (2026-10-25): adds LiteFormer-Cascade (XGBoost first, full stack only for uncertain flows),
macro_f1_present (macro-F1 over classes present in the test set) and stats_cascade_<setting>.csv.

IMPORTANT: if you re-run the experiments with this script, the paper must report the numbers this
script produces (they will not be identical to an earlier run).
"""
import pickle, argparse, json, os, platform, re, sys, time, warnings, glob
from collections import defaultdict
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score, recall_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import lightgbm as lgb, xgboost as xgb
import torch, torch.nn as nn

warnings.filterwarnings("ignore")
PROP = "LiteFormer-Stack"
CASC = "LiteFormer-Cascade"     # XGBoost first; full stack only for flows the calibrated XGBoost gate finds uncertain

# ----------------------------------------------------------------------------- hyperparameters
# These are the values actually used; config_used.json records them for the paper (Section 5.6).
HP = dict(
    rf=dict(n_estimators=200, max_depth=None, class_weight="balanced"),
    et=dict(n_estimators=200, max_depth=None, class_weight="balanced"),
    lgbm=dict(n_estimators=300, learning_rate=0.05, num_leaves=63, class_weight="balanced"),
    xgb=dict(n_estimators=300, learning_rate=0.1, max_depth=8, tree_method="hist"),
    mlp=dict(hidden_layer_sizes=(128, 64), max_iter=50, early_stopping=True),
    ft=dict(m=32, layers=2, heads=4, ff=64, dropout=0.1, lr=1e-3, batch=256, epochs=30, patience=3, gamma=2.0),
    cnnlstm=dict(channels=32, hidden=64, lr=1e-3, batch=256, epochs=20, patience=3),
    folds=5, k_grid=[10, 15, 20, 25, 30, 40], corr=0.95, tau_grid=[round(x, 2) for x in np.arange(0.50, 1.00, 0.01)],
    f1_tol=0.1, err_tol=0.005, min_cal_per_class=30, latency_flows=500,
)

# ----------------------------------------------------------------------------- data loading
CIC8 = [("DDoS", "DDoS"), ("DoS", "DoS"), ("Mirai", "Mirai"), ("Recon", "Recon"), ("VulnerabilityScan", "Recon"),
        ("MITM", "Spoofing"), ("DNS_Spoofing", "Spoofing"), ("DictionaryBruteForce", "BruteForce"),
        ("BrowserHijacking", "Web"), ("XSS", "Web"), ("Uploading", "Web"), ("SqlInjection", "Web"),
        ("CommandInjection", "Web"), ("Backdoor", "Web"), ("Benign", "Benign")]

def cic8(label):
    for pre, cat in CIC8:
        if label.startswith(pre): return cat
    raise ValueError(f"unknown CICIoT2023 label {label}")

def stratified_sample(df, label_col, frac, min_per_class, seed=0):
    parts = []
    for _, g in df.groupby(label_col):
        n = max(min(len(g), min_per_class), int(round(len(g) * frac)))
        parts.append(g.sample(n=min(n, len(g)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

def load_cic(cic_dir, frac, min_per_class):
    files = sorted(glob.glob(os.path.join(cic_dir, "*.csv")))
    if not files: raise SystemExit(f"no CSV files in {cic_dir}")
    chunks = []; rows_in_files = 0
    for f in files:
        d = pd.read_csv(f); rows_in_files += len(d)
        lab = [c for c in d.columns if c.lower() == "label"][0]
        d = d.rename(columns={lab: "label"})
        chunks.append(stratified_sample(d, "label", frac, min_per_class // len(files) + 1, 0))       # per-file thinning keeps memory low
    df = pd.concat(chunks, ignore_index=True)
    total_rows_read = sum(len(c) for c in chunks)
    df = stratified_sample(df, "label", 1.0, min_per_class)
    feats = [c for c in df.columns if c != "label"]
    return df[feats], df["label"].astype(str), dict(files=len(files), file_names=[os.path.basename(f) for f in files],
                                                    rows_in_files=int(rows_in_files), rows_after_sampling=len(df),
                                                    sampling_fraction=frac, min_per_class=min_per_class,
                                                    rows_read_after_per_file_sampling=total_rows_read)

def load_nbaiot(nb_dir, frac):
    rows = []
    for f in sorted(glob.glob(os.path.join(nb_dir, "*.csv"))):
        name = os.path.basename(f)[:-4]                                # e.g. "1.mirai.ack" or "3.benign"
        m = re.match(r"^(\d+)\.(benign|mirai|gafgyt)(?:\.(\w+))?$", name)
        if not m: continue
        dev, fam, atk = m.group(1), m.group(2), m.group(3)
        d = pd.read_csv(f)
        if frac < 1: d = d.sample(frac=frac, random_state=0)
        d["device"] = int(dev); d["y3"] = fam if fam != "gafgyt" else "bashlite"
        d["y11"] = "benign" if fam == "benign" else f"{'bashlite' if fam == 'gafgyt' else 'mirai'}_{atk}"
        rows.append(d)
    if not rows: raise SystemExit(f"no N-BaIoT files like '1.benign.csv' in {nb_dir}")
    df = pd.concat(rows, ignore_index=True)
    feats = [c for c in df.columns if c not in ("device", "y3", "y11")]
    return df, feats

def clean(X, y, extra=None, stats=None):
    n0 = len(y)
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = ~X.isna().any(axis=1)
    X, y = X[keep], y[keep]
    extra = extra[keep] if extra is not None else None
    dup = X.assign(_y=y.values).duplicated()
    X, y = X[~dup].reset_index(drop=True), y[~dup].reset_index(drop=True)
    extra = extra[~dup].reset_index(drop=True) if extra is not None else None
    if stats is not None:
        stats.update(rows_before_cleaning=int(n0), removed_nan_inf=int((~keep).sum()), removed_duplicates=int(dup.sum()), rows_after_cleaning=int(len(y)))
    return X, y, extra

# ----------------------------------------------------------------------------- torch models
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FTTransformer(nn.Module):
    def __init__(s, k, K, m, layers, heads, ff, dropout):
        super().__init__()
        s.W = nn.Parameter(torch.randn(k, m) * 0.02); s.b = nn.Parameter(torch.zeros(k, m))
        s.cls = nn.Parameter(torch.zeros(1, 1, m))
        enc = nn.TransformerEncoderLayer(m, heads, ff, dropout, batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(enc, layers); s.norm = nn.LayerNorm(m); s.head = nn.Linear(m, K)
    def forward(s, x):
        e = x.unsqueeze(-1) * s.W + s.b                      # Eq. (1): e_j = x_j W_j + b_j
        e = torch.cat([s.cls.expand(len(x), -1, -1), e], 1)
        return s.head(s.norm(s.enc(e)[:, 0]))

class CNNLSTM(nn.Module):
    def __init__(s, k, K, channels, hidden):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(1, channels, 3, padding=1), nn.ReLU())
        s.lstm = nn.LSTM(channels, hidden, batch_first=True); s.head = nn.Linear(hidden, K)
    def forward(s, x):
        h = s.conv(x.unsqueeze(1)).transpose(1, 2)
        _, (hn, _) = s.lstm(h); return s.head(hn[-1])

def focal_loss(logits, y, alpha, gamma):
    logp = torch.log_softmax(logits, 1); p = logp.exp()
    lp, pt = logp.gather(1, y[:, None]).squeeze(1), p.gather(1, y[:, None]).squeeze(1)
    return (-alpha[y] * (1 - pt) ** gamma * lp).mean()             # Eq. (3)

class TorchClf:
    def __init__(s, kind, K, seed, hp):
        s.kind, s.K, s.seed, s.hp = kind, K, seed, hp
    def fit(s, X, y):
        torch.manual_seed(s.seed); np.random.seed(s.seed)
        Xtr, Xes, ytr, yes = train_test_split(X, y, test_size=0.1, stratify=y, random_state=s.seed)  # inner early-stopping split
        k = X.shape[1]
        s.net = (FTTransformer(k, s.K, s.hp["m"], s.hp["layers"], s.hp["heads"], s.hp["ff"], s.hp["dropout"]) if s.kind == "ft"
                 else CNNLSTM(k, s.K, s.hp["channels"], s.hp["hidden"])).to(DEV)
        w = compute_class_weight("balanced", classes=np.arange(s.K), y=np.r_[y, np.arange(s.K)])
        alpha = torch.tensor(w / w.mean(), dtype=torch.float32, device=DEV)
        opt = torch.optim.Adam(s.net.parameters(), lr=s.hp["lr"])
        Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)
        best, bad, state = 1e9, 0, None
        for ep in range(s.hp["epochs"]):
            s.net.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), s.hp["batch"]):
                idx = perm[i:i + s.hp["batch"]]; xb, yb = Xt[idx].to(DEV), yt[idx].to(DEV)
                out = s.net(xb)
                loss = focal_loss(out, yb, alpha, s.hp["gamma"]) if s.kind == "ft" else nn.functional.cross_entropy(out, yb, weight=alpha)
                opt.zero_grad(); loss.backward(); opt.step()
            vl = nn.functional.cross_entropy(torch.tensor(s._logits(Xes)), torch.tensor(yes, dtype=torch.long)).item()
            if vl < best - 1e-4: best, bad, state = vl, 0, {kk: v.detach().clone() for kk, v in s.net.state_dict().items()}
            else:
                bad += 1
                if bad >= s.hp["patience"]: break
        if state: s.net.load_state_dict(state)
        return s
    def _logits(s, X):
        s.net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                out.append(s.net(torch.tensor(X[i:i + 4096], dtype=torch.float32, device=getattr(s, "dev", DEV))).cpu().numpy())
        return np.vstack(out)
    def predict_proba(s, X):
        z = s._logits(X); z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
    def n_params(s): return int(sum(p.numel() for p in s.net.parameters()))
    def to_cpu(s):                       # latency is always timed on the CPU, even if training used a GPU
        s.net.to("cpu"); s.dev = torch.device("cpu")

# ----------------------------------------------------------------------------- sklearn-style wrappers
def full_proba(model, X, K):
    """predict_proba aligned to classes 0..K-1 even if a fold missed a class."""
    p = model.predict_proba(X); cls = getattr(model, "classes_", np.arange(p.shape[1]))
    out = np.zeros((len(X), K)); out[:, np.asarray(cls, dtype=int)] = p; return out

def make(name, K, seed, threads=-1):
    if name == "Random Forest": return RandomForestClassifier(**HP["rf"], n_jobs=threads, random_state=seed)
    if name == "Extra Trees": return ExtraTreesClassifier(**HP["et"], n_jobs=threads, random_state=seed)
    if name == "LightGBM": return lgb.LGBMClassifier(**HP["lgbm"], n_jobs=threads, random_state=seed, verbose=-1)
    if name == "XGBoost": return xgb.XGBClassifier(**HP["xgb"], n_jobs=threads, random_state=seed, objective="multi:softprob" if K > 2 else "binary:logistic")
    if name == "MLP": return MLPClassifier(**HP["mlp"], random_state=seed)
    if name == "CNN-LSTM": return TorchClf("cnnlstm", K, seed, HP["cnnlstm"])
    if name == "FT-Transformer only": return TorchClf("ft", K, seed, HP["ft"])
    raise ValueError(name)

def fit(model, X, y, K):
    if isinstance(model, xgb.XGBClassifier):
        # XGBoost needs every class 0..K-1 present; class balance through sample weights
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model

# ----------------------------------------------------------------------------- feature selection (Sec. 4.3)
def select_features(Xtr, ytr, Xsel, ysel, seed, K):
    n = min(len(Xtr), 50000); idx = np.random.RandomState(seed).choice(len(Xtr), n, replace=False)
    mi = mutual_info_classif(Xtr[idx], ytr[idx], random_state=seed)
    order = list(np.argsort(-mi)); corr = np.corrcoef(Xtr[idx].T); keep = []
    for j in order:
        if all(abs(corr[j, i]) <= HP["corr"] for i in keep if np.isfinite(corr[j, i])): keep.append(j)
    grid = sorted({min(k, len(keep)) for k in HP["k_grid"] + [len(keep)]}); scores = {}
    for k in grid:
        m = fit(make("LightGBM", K, seed), Xtr[:, keep[:k]], ytr, K)
        scores[k] = f1_score(ysel, m.predict(Xsel[:, keep[:k]]), average="macro") * 100
    best = max(scores.values()); k = min(k for k, v in scores.items() if v >= best - HP["f1_tol"])
    return keep[:k], dict(k=k, k_scores=scores, after_corr_filter=len(keep))

# ----------------------------------------------------------------------------- stacking + gate (Sec. 4.6–4.7)
BASE = ["FT-Transformer only", "LightGBM", "XGBoost", "Extra Trees"]

def oof_stack(Xtr, ytr, K, seed):
    oof = {b: np.zeros((len(Xtr), K)) for b in BASE}
    for tr, va in StratifiedKFold(HP["folds"], shuffle=True, random_state=seed).split(Xtr, ytr):
        for b in BASE:
            oof[b][va] = full_proba(fit(make(b, K, seed), Xtr[tr], ytr[tr], K), Xtr[va], K)
    return oof

def softmax_T(s, T):
    z = s / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def fit_temperature(s, y):
    nll = lambda lt: -np.mean(np.log(softmax_T(s, np.exp(lt))[np.arange(len(y)), y] + 1e-12))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))

def choose_thresholds(pT, yhat_T, yhat_F, y, K):
    conf = pT.max(1); grid = HP["tau_grid"]; f_full = f1_score(y, yhat_F, average="macro") * 100
    ok = [abs(f1_score(y, np.where(conf >= t, yhat_T, yhat_F), average="macro") * 100 - f_full) <= HP["f1_tol"] for t in grid]
    tau = next((t for i, t in enumerate(grid) if all(ok[i:])), 1.01)
    tau_c = {}
    for c in range(K):
        mask = yhat_T == c
        if mask.sum() < HP["min_cal_per_class"]: tau_c[c] = 1.01; continue
        cand = [t for t in grid if t >= tau]; good = []
        for t in cand:
            acc = mask & (conf >= t)
            good.append(True if acc.sum() == 0 else (np.mean(yhat_T[acc] != y[acc]) - np.mean(yhat_F[acc] != y[acc]) <= HP["err_tol"]))
        tau_c[c] = next((t for i, t in enumerate(cand) if all(good[i:])), 1.01)
    return tau, tau_c

# ----------------------------------------------------------------------------- latency (single core, one flow at a time)
def time_per_flow(fn, X, n):
    X = X[:n]; fn(X[:1])                                   # warm-up
    t0 = time.perf_counter()
    for i in range(len(X)): fn(X[i:i + 1])
    return (time.perf_counter() - t0) / len(X) * 1e6

def time_batched(fn, X, n, reps=5):
    """per-flow cost when n flows are scored in one call (median of reps)."""
    X = X[:n]; fn(X[:2]); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(X); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / len(X) * 1e6

def set_single_thread(models):
    torch.set_num_threads(1)
    for m in models:
        if hasattr(m, "to_cpu"): m.to_cpu()
        for attr in ("n_jobs",):
            if hasattr(m, attr):
                try: m.set_params(**{attr: 1})
                except Exception: pass

# ----------------------------------------------------------------------------- one run of one setting
def run_once(Xtr, ytr, Xva, yva, Xte, yte, K, seed, benign, out_pred, run, record):
    qt = QuantileTransformer(n_quantiles=min(1000, len(Xtr)), output_distribution="normal", random_state=seed).fit(Xtr)
    Xtr, Xva, Xte = qt.transform(Xtr), qt.transform(Xva), qt.transform(Xte)
    Xsel, Xcal, ysel, ycal = train_test_split(Xva, yva, test_size=0.5, stratify=yva, random_state=seed)
    S, finfo = select_features(Xtr, ytr, Xsel, ysel, seed, K)
    Xtr, Xsel, Xcal, Xte = Xtr[:, S], Xsel[:, S], Xcal[:, S], Xte[:, S]
    res, preds = {}, {}
    # single-model baselines
    models = {}
    for name in ["Random Forest", "XGBoost", "LightGBM", "MLP", "CNN-LSTM", "FT-Transformer only", "Extra Trees"]:
        models[name] = fit(make(name, K, seed), Xtr, ytr, K)
    # stacking: OOF meta-features, meta-learners, temperature, thresholds
    oof = oof_stack(Xtr, ytr, K, seed)
    zF = np.hstack([oof[b] for b in BASE]); zT = np.hstack([oof[b] for b in BASE[1:]])
    gF = LogisticRegression(max_iter=2000).fit(zF, ytr); gT = LogisticRegression(max_iter=2000).fit(zT, ytr)
    base_p = lambda X: {b: full_proba(models[b], X, K) for b in BASE}
    def logits(g, z):
        d = g.decision_function(z)
        return d.reshape(len(z), -1) if K > 2 else np.c_[-d, d] / 2      # binary: softmax([-d/2, d/2]) = sigmoid(d)
    def stack(P, which):
        z = np.hstack([P[b] for b in (BASE if which == "F" else BASE[1:])]); g = gF if which == "F" else gT
        return full_proba(g, z, K), logits(g, z)
    Pc = base_p(Xcal); pF_cal, _ = stack(Pc, "F"); _, sT_cal = stack(Pc, "T")
    T = fit_temperature(sT_cal, ycal); pT_cal = softmax_T(sT_cal, T)
    tau, tau_c = choose_thresholds(pT_cal, pT_cal.argmax(1), pF_cal.argmax(1), ycal, K)
    sX_cal = np.log(Pc["XGBoost"] + 1e-12); TX = fit_temperature(sX_cal, ycal); pX_cal = softmax_T(sX_cal, TX)
    tauX, tauX_c = choose_thresholds(pX_cal, pX_cal.argmax(1), pF_cal.argmax(1), ycal, K)
    # test-set predictions
    P = base_p(Xte); pF, _ = stack(P, "F"); _, sT = stack(P, "T"); pT = softmax_T(sT, T)
    yT, yF = pT.argmax(1), pF.argmax(1); thr = np.array([tau_c[c] for c in yT]); fast = pT.max(1) >= thr
    preds[PROP] = np.where(fast, yT, yF); preds["Full stack (ungated)"] = yF; preds["Tree-only stack"] = yT
    pX = softmax_T(np.log(P["XGBoost"] + 1e-12), TX); yX = pX.argmax(1)
    fastX = pX.max(1) >= np.array([tauX_c[c] for c in yX]); preds[CASC] = np.where(fastX, yX, yF)
    preds["Soft-voting ensemble"] = np.mean([P[b] for b in BASE], 0).argmax(1)
    for name in ["Random Forest", "XGBoost", "LightGBM", "MLP", "CNN-LSTM", "FT-Transformer only"]:
        preds[name] = full_proba(models[name], Xte, K).argmax(1)
    # latency
    set_single_thread(list(models.values()) + [gF, gT]); n = HP["latency_flows"]
    one = lambda name: (lambda x: models[name].predict_proba(x))
    lat = {name: time_per_flow(one(name), Xte, n) for name in ["Random Forest", "XGBoost", "LightGBM", "MLP", "CNN-LSTM", "FT-Transformer only"]}
    def tree_path(x):
        z = np.hstack([full_proba(models[b], x, K) for b in BASE[1:]]); return softmax_T(logits(gT, z), T)
    def full_path(x):
        return full_proba(gF, np.hstack([full_proba(models[b], x, K) for b in BASE]), K)
    lat["Tree-only stack"] = time_per_flow(tree_path, Xte, n)
    lat["Full stack (ungated)"] = time_per_flow(full_path, Xte, n)
    lat["Soft-voting ensemble"] = time_per_flow(lambda x: np.mean([full_proba(models[b], x, K) for b in BASE], 0), Xte, n)
    def gated(x):
        p = tree_path(x)[0]; c = int(np.argmax(p))
        return p if p.max() >= tau_c[c] else full_path(x)
    lat[PROP] = time_per_flow(gated, Xte, n)
    lat_et = time_per_flow(one("Extra Trees"), Xte, n)
    def rest_full(x, px):                                   # stage 2 reuses the XGBoost probabilities of stage 1
        return full_proba(gF, np.hstack([px if b == "XGBoost" else full_proba(models[b], x, K) for b in BASE]), K)
    def cascade(x):
        px = full_proba(models["XGBoost"], x, K); p = softmax_T(np.log(px + 1e-12), TX)[0]; c = int(np.argmax(p))
        return p if p.max() >= tauX_c[c] else rest_full(x, px)
    lat[CASC] = time_per_flow(cascade, Xte, n)
    # batched per-flow cost (n flows per call), the same models, one thread
    latb = {name: time_batched(one(name), Xte, n) for name in ["Random Forest", "XGBoost", "LightGBM", "MLP", "CNN-LSTM", "FT-Transformer only"]}
    latb["Tree-only stack"] = time_batched(tree_path, Xte, n); latb["Full stack (ungated)"] = time_batched(full_path, Xte, n)
    latb["Soft-voting ensemble"] = time_batched(lambda x: np.mean([full_proba(models[b], x, K) for b in BASE], 0), Xte, n)
    def gated_batch(x):
        p = tree_path(x); c = p.argmax(1); route = p.max(1) < np.array([tau_c[int(i)] for i in c])
        if route.any(): p[route] = full_path(x[route])
        return p
    latb[PROP] = time_batched(gated_batch, Xte, n)
    def cascade_batch(x):
        px = full_proba(models["XGBoost"], x, K); p = softmax_T(np.log(px + 1e-12), TX); c = p.argmax(1)
        route = p.max(1) < np.array([tauX_c[int(i)] for i in c])
        if route.any(): p[route] = rest_full(x[route], px[route])
        return p
    latb[CASC] = time_batched(cascade_batch, Xte, n)
    rho = float(fast.mean()); rhoX = float(fastX.mean())
    # metrics
    for name, yp in preds.items():
        m = dict(accuracy=accuracy_score(yte, yp) * 100, macro_f1=f1_score(yte, yp, average="macro") * 100,
                 macro_f1_present=f1_score(yte, yp, labels=np.unique(yte), average="macro") * 100,
                 mcc=matthews_corrcoef(yte, yp), kappa=cohen_kappa_score(yte, yp), latency_us=lat[name], latency_batched_us=latb[name],
                 recall=recall_score(yte, yp, labels=range(K), average=None, zero_division=0) * 100)
        if benign is not None:
            b = yte == benign; m["benign_fpr"] = float(np.mean(yp[b] != benign) * 100) if b.any() else np.nan
        res[name] = m
        os.makedirs(out_pred, exist_ok=True)
        extra = dict(p_tree_cal=pT, routed_full=~fast) if name == PROP else dict(p_xgb_cal=pX, routed_full=~fastX) if name == CASC else {}
        np.savez_compressed(os.path.join(out_pred, f"{name}_run{run}.npz"), y_true=yte, y_pred=yp, **extra)
    record.update(k=finfo["k"], k_scores=finfo["k_scores"], features_after_corr_filter=finfo["after_corr_filter"],
                  ft_params=models["FT-Transformer only"].n_params(), temperature=T, tau=tau,
                  tau_c={int(c): v for c, v in tau_c.items()}, classes_always_full=[int(c) for c, v in tau_c.items() if v > 1.0],
                  rho=rho, t_T_us=lat["Tree-only stack"], t_F_us=lat["Full stack (ungated)"] - lat["Tree-only stack"],
                  t_T_batched_us=latb["Tree-only stack"], t_F_batched_us=latb["Full stack (ungated)"] - latb["Tree-only stack"],
                  n_train=len(ytr), n_sel=len(ysel), n_cal=len(ycal), n_test=len(yte),
                  class_counts=dict(train=np.bincount(ytr, minlength=K).tolist(), selection=np.bincount(ysel, minlength=K).tolist(),
                                    calibration=np.bincount(ycal, minlength=K).tolist(), test=np.bincount(yte, minlength=K).tolist()),
                  threads=1, latency_flows=n,
                  cascade=dict(temperature=TX, tau=tauX, tau_c={int(c): v for c, v in tauX_c.items()},
                               classes_always_full=[int(c) for c, v in tauX_c.items() if v > 1.0], rho=rhoX,
                               t_xgb_us=lat["XGBoost"], t_et_us=lat_et))
    return res, preds

# ----------------------------------------------------------------------------- statistics
def corrected_ci(x, ratio):
    R = len(x); return stats.t.ppf(0.975, R - 1) * np.std(x, ddof=1) * np.sqrt(1 / R + ratio) if R > 1 else np.nan

def mcnemar(y, a, b):
    """a = proposed, b = baseline. Returns chi2, p (exact binomial when b+c<25)."""
    nb = int(np.sum((a == y) & (b != y))); nc = int(np.sum((a != y) & (b == y)))
    if nb + nc == 0: return 0.0, 1.0
    if nb + nc < 25: return float("nan"), float(stats.binomtest(min(nb, nc), nb + nc, 0.5).pvalue)
    chi = (abs(nb - nc) - 1) ** 2 / (nb + nc); return chi, float(stats.chi2.sf(chi, 1))

def holm(p):
    names = sorted(p, key=p.get); m = len(names); adj, run = {}, 0
    for i, n in enumerate(names): run = max(run, min(1, (m - i) * p[n])); adj[n] = run
    return adj

def cliffs(x, y): return (sum(a > b for a in x for b in y) - sum(a < b for a in x for b in y)) / (len(x) * len(y))

NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

# ----------------------------------------------------------------------------- settings
def make_settings(args):
    S = {}
    if args.smoke:
        from sklearn.datasets import make_classification
        K = args.smoke_classes; w = ([.5, .2, .15, .1, .04, .01] if K == 6 else [.7, .3])
        X, y = make_classification(6000, 30, n_informative=12, n_classes=K, n_clusters_per_class=1, weights=w, random_state=0)
        X = pd.DataFrame(X); y = pd.Series(np.array(["Benign", "A", "B", "C", "D", "E"][:K])[y])
        dev = pd.Series(np.random.RandomState(0).randint(1, 4, len(y)))
        S["smoke_random"] = dict(X=X, y=y, mode="random"); S["smoke_device"] = dict(X=X, y=y, mode="device", device=dev)
        return S
    if args.cic_dir:
        X, y, info = load_cic(args.cic_dir, args.cic_frac, args.min_per_class); cst = {}; X, y, _ = clean(X, y, stats=cst)
        info.update(cst)
        S["cic34"] = dict(X=X, y=y, mode="random", info=info)
        S["cic8"] = dict(X=X, y=y.map(cic8), mode="random", info=info)
        S["cic2"] = dict(X=X, y=y.map(lambda s: "Benign" if s.startswith("Benign") else "Attack"), mode="random", info=info)
    if args.nbaiot_dir:
        df, feats = load_nbaiot(args.nbaiot_dir, args.nbaiot_frac)
        for lab in ("y3", "y11"):
            cst = {}; X, y, dev = clean(df[feats], df[lab], df["device"], stats=cst)
            info = dict(sampling_fraction=args.nbaiot_frac, devices=sorted(dev.unique().tolist()), **cst)
            n = "nb3" if lab == "y3" else "nb11"
            S[f"{n}_random"] = dict(X=X, y=y, mode="random", info=info)
            S[f"{n}_device"] = dict(X=X, y=y, mode="device", device=dev, info=info)
    if args.settings: S = {k: v for k, v in S.items() if k in args.settings.split(",")}
    return S

def splits(setting, R, seed0):
    """random: R stratified 70/15/15 splits.  device: leave-one-device-out (one run per device)."""
    y = setting["y"].values
    vc = pd.Series(y).value_counts(); rare = set(vc[vc < 2].index)
    if rare: raise SystemExit(f"classes with <2 rows after cleaning: {sorted(rare)} - check sampling")
    if setting["mode"] == "random":
        for r in range(R):
            idx = np.arange(len(y)); tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed0 + r)
            va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed0 + r)
            yield r + 1, seed0 + r, tr, va, te, None
    else:
        dev = setting["device"].values
        for r, d in enumerate(sorted(np.unique(dev))):
            te = np.where(dev == d)[0]; rest = np.where(dev != d)[0]
            tr, va = train_test_split(rest, test_size=0.15 / 0.85, stratify=y[rest], random_state=seed0 + r)
            yield r + 1, seed0 + r, tr, va, te, int(d)

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cic-dir"); ap.add_argument("--nbaiot-dir"); ap.add_argument("--out", default="results_out")
    ap.add_argument("--runs", type=int, default=10); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cic-frac", type=float, default=0.02, help="stratified fraction of CICIoT2023 rows to use")
    ap.add_argument("--min-per-class", type=int, default=2000, help="keep at least this many rows of each class")
    ap.add_argument("--nbaiot-frac", type=float, default=0.10); ap.add_argument("--settings", help="comma list, e.g. cic34,nb3_device")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic self-test")
    ap.add_argument("--smoke-classes", type=int, default=6)
    a = ap.parse_args()
    if a.smoke:
        a.runs = 2; HP["folds"] = 2; HP["latency_flows"] = 50
        HP["ft"].update(epochs=3); HP["cnnlstm"].update(epochs=2); HP["rf"]["n_estimators"] = HP["et"]["n_estimators"] = 30
        HP["lgbm"]["n_estimators"] = HP["xgb"]["n_estimators"] = 50; HP["k_grid"] = [8, 12]
    if not (a.smoke or a.cic_dir or a.nbaiot_dir): ap.error("give --cic-dir and/or --nbaiot-dir (or --smoke)")
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(hyperparameters=HP, python=platform.python_version(), cpu=platform.processor() or platform.machine(),
               platform=platform.platform(), cpu_count=os.cpu_count(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=__import__("sklearn").__version__,
                             lightgbm=lgb.__version__, xgboost=xgb.__version__, torch=torch.__version__),
               latency_protocol=f"latency_us: time.perf_counter(), mean over {HP['latency_flows']} test flows, one flow per call (batch size 1), single CPU thread (torch.set_num_threads(1), n_jobs=1), 1 warm-up call; "
                                 f"latency_batched_us: {HP['latency_flows']} flows per call, median of 5 repetitions, divided by the number of flows",
               split_protocol="random: stratified 70/15/15 per run, validation split 50/50 into selection and calibration; "
                              "device: leave-one-device-out, one run per held-out device, remaining devices split 85/15 train/validation",
               settings={})
    settings = make_settings(a); mean_f1 = {}
    for sname, st in settings.items():
        print(f"\n=== {sname} ({len(st['y'])} rows, {st['y'].nunique()} classes, mode={st['mode']}) ===", flush=True)
        le = LabelEncoder().fit(st["y"]); y = le.transform(st["y"]); X = st["X"].values.astype(np.float32); K = len(le.classes_)
        benign = next((i for i, c in enumerate(le.classes_) if c.lower().startswith("benign")), None)
        runs, recs, all_preds = defaultdict(list), [], []
        for r, seed, tr, va, te, dev in splits(st, a.runs, a.seed):
            t0 = time.time(); rec = dict(run=r, seed=seed, held_out_device=dev)
            ck = os.path.join(a.out, "ckpt", sname, f"run{r}.pkl")
            if os.path.exists(ck):                                   # resume: reuse a run finished in an earlier session
                with open(ck, "rb") as fh: res, preds, rec = pickle.load(fh)
                print(f"  run {r}: loaded from checkpoint", flush=True)
            else:
                res, preds = run_once(X[tr], y[tr], X[va], y[va], X[te], y[te], K, seed, benign, os.path.join(a.out, "preds", sname), r, rec)
                os.makedirs(os.path.dirname(ck), exist_ok=True)
                with open(ck + ".tmp", "wb") as fh: pickle.dump((res, preds, rec), fh)
                os.replace(ck + ".tmp", ck)
            for n, m in res.items(): runs[n].append(m)
            recs.append(rec); all_preds.append((y[te], preds))
            print(f"  run {r}: {PROP} macro-F1 {res[PROP]['macro_f1']:.2f}  rho {rec['rho']:.2f}  k {rec['k']} | "
                  f"{CASC} macro-F1 {res[CASC]['macro_f1']:.2f}  rho {rec['cascade']['rho']:.2f}  "
                  f"lat {res[CASC]['latency_us']:.0f} vs {res[PROP]['latency_us']:.0f} us  ({time.time() - t0:.0f}s)", flush=True)
        ratio = 0.15 / 0.70 if st["mode"] == "random" else float(np.mean([len(np.where(st["device"].values == d)[0]) for d in np.unique(st["device"])])) / len(y)
        rows = []
        for n, ms in runs.items():
            f = [m["macro_f1"] for m in ms]
            row = dict(model=n, accuracy=np.mean([m["accuracy"] for m in ms]), macro_f1=np.mean(f), macro_f1_std=np.std(f, ddof=1) if len(f) > 1 else np.nan,
                       macro_f1_present=np.mean([m["macro_f1_present"] for m in ms]),
                       ci95=corrected_ci(f, ratio), mcc=np.mean([m["mcc"] for m in ms]), kappa=np.mean([m["kappa"] for m in ms]),
                       latency_us=np.mean([m["latency_us"] for m in ms]), latency_batched_us=np.mean([m["latency_batched_us"] for m in ms]))
            if benign is not None: row["benign_fpr"] = np.nanmean([m["benign_fpr"] for m in ms])
            rows.append(row)
        summ = pd.DataFrame(rows).sort_values("macro_f1").round(4); summ.to_csv(os.path.join(a.out, f"summary_{sname}.csv"), index=False)
        pd.DataFrame({n: np.mean([m["recall"] for m in ms], 0) for n, ms in runs.items()}, index=le.classes_).round(2)\
          .assign(test_support=np.bincount(all_preds[0][0], minlength=K)).to_csv(os.path.join(a.out, f"per_class_recall_{sname}.csv"))
        # Table 2 statistics, once for each proposed variant
        for prop_name, fname in ((PROP, f"stats_{sname}.csv"), (CASC, f"stats_cascade_{sname}.csv")):
            base = [n for n in runs if n != prop_name]
            mc = {n: [mcnemar(yt, p[prop_name], p[n]) for yt, p in all_preds] for n in base}
            fp = [m["macro_f1"] for m in runs[prop_name]]
            wil = {n: (stats.wilcoxon(fp, [m["macro_f1"] for m in runs[n]]).pvalue if len(fp) > 5 and np.any(np.array(fp) != np.array([m["macro_f1"] for m in runs[n]])) else np.nan) for n in base}
            hw = holm({n: v for n, v in wil.items() if np.isfinite(v)}) if any(np.isfinite(v) for v in wil.values()) else {}
            hm = holm({n: float(np.median([p for _, p in mc[n]])) for n in base})
            pd.DataFrame([dict(baseline=n, mcnemar_chi2_median=np.nanmedian([c for c, _ in mc[n]]), mcnemar_p_median=np.median([p for _, p in mc[n]]),
                               runs_p_lt_005=f"{sum(p < 0.05 for _, p in mc[n])}/{len(mc[n])}", mcnemar_p_median_holm=hm[n],
                               wilcoxon_p_holm=hw.get(n, np.nan), cliffs_delta=cliffs(fp, [m['macro_f1'] for m in runs[n]])) for n in base])\
              .to_csv(os.path.join(a.out, fname), index=False)
        mean_f1[sname] = {n: np.mean([m["macro_f1"] for m in ms]) for n, ms in runs.items()}
        cfg["settings"][sname] = dict(mode=st["mode"], n_rows=len(y), n_classes=K, classes=list(map(str, le.classes_)), info=st.get("info"), runs=recs,
                                      k_per_run=[r_["k"] for r_ in recs], ft_params=recs[0]["ft_params"],
                                      T_mean=float(np.mean([r_["temperature"] for r_ in recs])), tau_mean=float(np.mean([r_["tau"] for r_ in recs])),
                                      rho_mean=float(np.mean([r_["rho"] for r_ in recs])), rho_std=float(np.std([r_["rho"] for r_ in recs], ddof=1)) if len(recs) > 1 else None,
                                      cascade_rho_mean=float(np.mean([r_["cascade"]["rho"] for r_ in recs])),
                                      cascade_T_mean=float(np.mean([r_["cascade"]["temperature"] for r_ in recs])))
        json.dump(cfg, open(os.path.join(a.out, "config_used.json"), "w"), indent=1, default=float)
    # Friedman over settings (Supplementary Table S2)
    models = ["LiteFormer-Stack", "LiteFormer-Cascade", "Soft-voting ensemble", "LightGBM", "XGBoost", "Random Forest", "FT-Transformer only", "CNN-LSTM", "MLP"]
    if len(mean_f1) >= 2:
        M = pd.DataFrame(mean_f1).T[models]; ranks = M.rank(axis=1, ascending=False).mean(); N, k = M.shape
        chi = 12 * N / (k * (k + 1)) * (np.sum(ranks ** 2) - k * (k + 1) ** 2 / 4); FF = (N - 1) * chi / (N * (k - 1) - chi)
        cd = NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
        out = M.round(3); out.loc["average_rank"] = ranks.round(3)
        out.to_csv(os.path.join(a.out, "friedman.csv"))
        json.dump(dict(N=N, k=k, chi2_F=chi, p=stats.chi2.sf(chi, k - 1), iman_davenport_F=FF, p_FF=stats.f.sf(FF, k - 1, (k - 1) * (N - 1)), nemenyi_CD=cd),
                  open(os.path.join(a.out, "friedman_stats.json"), "w"), indent=1, default=float)
    print(f"\nDone. All tables and config_used.json are in {a.out}/")

if __name__ == "__main__":
    main()
