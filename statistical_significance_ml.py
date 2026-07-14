"""
Statistical significance testing for ML models via k-fold cross-validation.
Produces k R² scores per (model, PC), enabling:
  - One-sample tests vs 0 (is the model better than predicting the mean?)
  - Pairwise tests between models (paired by fold)
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from scipy import stats

from torch_geometric.data import Batch
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from engineered_features import FeatureCalculator

# ── Config ────────────────────────────────────────────────────────────────────
N_FOLDS   = 10
N_PCS     = 15          # number of target PCs
ALPHA     = 0.05
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def RF(): return RandomForestRegressor(n_jobs=-1)

MODELS    = {
    "SVR":                   SVR,
    "RandomForest":          RF,
    "LinearRegression":      LinearRegression,
    "Ridge":                 Ridge,
}

# ── Load & format data ────────────────────────────────────────────────────────
def get_features(dataset, batch_size=16):
    fc = FeatureCalculator()
    x, y = [], []
    for i in tqdm(range(0, len(dataset), batch_size), leave=False):
        samples = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        batch = Batch.from_data_list(samples).to(DEVICE)
        x.append(fc.forward(batch.x, batch.edge_index, batch.meta, batch=batch.batch).cpu())
        y.append(batch.y.cpu())
    return torch.cat(x).numpy(), torch.cat(y).numpy()

print("Loading datasets…")
train_set = torch.load(".data_cache/data_src_train.pt", weights_only=False)
val_set   = torch.load(".data_cache/data_src_val.pt",   weights_only=False)
test_set  = torch.load(".data_cache/data_src_test.pt",  weights_only=False)

X_train, y_train = get_features(train_set)
X_val,   y_val   = get_features(val_set)
X_test,  y_test  = get_features(test_set)

# Pool everything for cross-validation
X = np.concatenate([X_train, X_val, X_test], axis=0)
y = np.concatenate([y_train, y_val, y_test], axis=0)
print(f"Full dataset: X={X.shape}, y={y.shape}")

# ── Cross-validation ──────────────────────────────────────────────────────────
# results[model_name][pc_idx] = list of k R² values
results = {name: [[] for _ in range(N_PCS)] for name in MODELS}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
folds = list(kf.split(X))

print(f"\nRunning {N_FOLDS}-fold CV for {len(MODELS)} models × {N_PCS} PCs…")
for name, ModelClass in MODELS.items():
    print(f"  {name}")
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        for pc in range(N_PCS):
            model = ModelClass()
            model.fit(X_tr, X_tr[:, pc])          # predicts PC from features
            y_pred = model.predict(X_te)
            r2 = r2_score(y_te[:, pc], y_pred)
            results[name][pc].append(r2)

# ── Aggregate: mean R² per (model, fold) across PCs ──────────────────────────
# Shape: (n_models, n_folds)
model_names = list(MODELS.keys())
scores = np.array([
    [np.mean([results[name][pc][fold] for pc in range(N_PCS)]) for fold in range(N_FOLDS)]
    for name in model_names
])  # shape: (n_models, n_folds)

# ── Report ────────────────────────────────────────────────────────────────────
def sig_label(p, corrected_alpha):
    if p < 0.001:           return "***"
    elif p < 0.01:          return "**"
    elif p < corrected_alpha: return "*"
    else:                   return "ns"

SEP = "=" * 70

print(f"\n{SEP}")
print(f"RESULTS  ({N_FOLDS}-fold CV, R² averaged over {N_PCS} PCs)")
print(f"{SEP}")

print(f"\n{'Model':<22} {'Mean R²':>9} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 58)
for i, name in enumerate(model_names):
    s = scores[i]
    print(f"{name:<22} {s.mean():>9.4f} {s.std():>8.4f} {s.min():>8.4f} {s.max():>8.4f}")

# ── One-sample tests vs 0 ─────────────────────────────────────────────────────
n_models = len(model_names)
ca = ALPHA / n_models   # Bonferroni
print(f"\n--- One-sample tests vs R²=0  (Bonferroni α={ca:.4f}) ---")
print(f"{'Model':<22} {'t-stat':>8} {'t p':>9} {'W-stat':>8} {'W p':>9} {'sig(t)':>7} {'sig(W)':>7}")
print("-" * 72)
for i, name in enumerate(model_names):
    s = scores[i]
    t_stat, t_p = stats.ttest_1samp(s, popmean=0, alternative="two-sided")
    try:
        w_stat, w_p = stats.wilcoxon(s, alternative="two-sided")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    print(f"{name:<22} {t_stat:>8.3f} {t_p:>9.4f} {w_stat:>8.1f} {w_p:>9.4f} "
          f"{sig_label(t_p, ca):>7} {sig_label(w_p, ca):>7}")

# ── Pairwise tests between models ─────────────────────────────────────────────
pairs = list(combinations(range(n_models), 2))
ca2   = ALPHA / len(pairs)
print(f"\n--- Pairwise tests (paired by fold, Bonferroni α={ca2:.4f}) ---")
print(f"{'Pair':<42} {'t-stat':>8} {'t p':>9} {'W-stat':>8} {'W p':>9} {'sig(t)':>7} {'sig(W)':>7}")
print("-" * 96)
for i, j in pairs:
    diff = scores[i] - scores[j]
    t_stat, t_p = stats.ttest_rel(scores[i], scores[j])
    try:
        w_stat, w_p = stats.wilcoxon(scores[i], scores[j], alternative="two-sided")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    pair_str = f"{model_names[i]} vs {model_names[j]}"
    print(f"{pair_str:<42} {t_stat:>8.3f} {t_p:>9.4f} {w_stat:>8.1f} {w_p:>9.4f} "
          f"{sig_label(t_p, ca2):>7} {sig_label(w_p, ca2):>7}")

# ── Per-PC breakdown (vs 0 only) ──────────────────────────────────────────────
print(f"\n--- Per-PC one-sample t-test vs 0 (uncorrected, for exploration) ---")
col_w = 24
header = f"{'PC':<5}" + "".join(f"{n:>{col_w}}" for n in model_names)
print(header)
print("-" * (5 + col_w * n_models))
for pc in range(N_PCS):
    row = f"{pc+1:<5}"
    for name in model_names:
        s = np.array(results[name][pc])
        t_stat, t_p = stats.ttest_1samp(s, popmean=0, alternative="two-sided")
        sl = sig_label(t_p, ALPHA)
        cell = f"{s.mean():>+6.5f}±{s.std():.5f} ({sl:>3})"
        row += f"  {cell:>{col_w - 2}}"
    print(row)

print(f"\n{SEP}")
print("Sig codes: *** p<0.001  ** p<0.01  * corrected α  ns not significant")
print(f"Pairwise Bonferroni corrects for {len(pairs)} comparisons; vs-0 corrects for {n_models} models.")

# ── Save fold-level scores ─────────────────────────────────────────────────────
df_scores = pd.DataFrame(scores.T, columns=model_names)
df_scores.index.name = "fold"
df_scores.to_csv("cv_r2_scores.csv")
print(f"\nFold-level R² scores saved to cv_r2_scores.csv")
