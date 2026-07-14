import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

df = pd.read_csv("ml_results.csv")

MODELS = ["SVR", "RandomForestRegressor", "LinearRegression", "Ridge"]
METRICS = {"r2": "R² score", "mse": "MSE"}

print("=" * 60)
print("STATISTICAL COMPARISON OF ML MODELS")
print(f"N = {len(df)} principal components")
print("=" * 60)

for metric_key, metric_label in METRICS.items():
    print(f"\n{'─' * 60}")
    print(f"Metric: {metric_label}")
    print(f"{'─' * 60}")

    data = {}
    for model in MODELS:
        col = f"{model}_{metric_key}"
        data[model] = df[col].dropna().values

    # Descriptive stats
    print(f"\n{'Model':<30} {'Mean':>10} {'Std':>10} {'Median':>10}")
    print("-" * 62)
    for model in MODELS:
        vals = data[model]
        print(f"{model:<30} {vals.mean():>10.4f} {vals.std():>10.4f} {np.median(vals):>10.4f}")

    # One-sample tests against 0
    n_models = len(MODELS)
    corrected_alpha_vs0 = 0.05 / n_models
    print(f"\nOne-sample tests vs 0 (Bonferroni α={corrected_alpha_vs0:.4f}):")
    print(f"  {'Model':<30} {'t-stat':>8} {'t p-val':>9} {'W-stat':>8} {'W p-val':>9} {'sig (t)':>8} {'sig (W)':>8}")
    print("  " + "-" * 86)
    for model in MODELS:
        vals = data[model]
        t_stat, t_p = stats.ttest_1samp(vals, popmean=0, alternative="two-sided")
        try:
            w_stat, w_p = stats.wilcoxon(vals, alternative="two-sided")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        sig_t = "***" if t_p < 0.001 else ("**" if t_p < 0.01 else ("*" if t_p < corrected_alpha_vs0 else "ns"))
        sig_w = "***" if w_p < 0.001 else ("**" if w_p < 0.01 else ("*" if w_p < corrected_alpha_vs0 else "ns"))
        print(f"  {model:<30} {t_stat:>8.3f} {t_p:>9.4f} {w_stat:>8.1f} {w_p:>9.4f} {sig_t:>8} {sig_w:>8}")

    # Align lengths for paired tests (use rows where all models have data)
    aligned_cols = [f"{m}_{metric_key}" for m in MODELS]
    df_clean = df[aligned_cols].dropna()
    aligned = {m: df_clean[f"{m}_{metric_key}"].values for m in MODELS}
    n = len(df_clean)
    print(f"\nPaired samples available: {n}")

    # Friedman test (non-parametric repeated-measures ANOVA)
    arrays = [aligned[m] for m in MODELS]
    stat, p = stats.friedmanchisquare(*arrays)
    print(f"\nFriedman test: χ²={stat:.4f}, p={p:.4f}", end="")
    print(" ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else " (ns)")))

    # Pairwise Wilcoxon signed-rank tests with Bonferroni correction
    pairs = list(combinations(MODELS, 2))
    alpha = 0.05
    corrected_alpha = alpha / len(pairs)
    print(f"\nPairwise Wilcoxon signed-rank tests (Bonferroni α={corrected_alpha:.4f}):")
    print(f"  {'Pair':<50} {'statistic':>10} {'p-value':>10} {'sig':>5}")
    print("  " + "-" * 78)
    for m1, m2 in pairs:
        stat, p = stats.wilcoxon(aligned[m1], aligned[m2])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < corrected_alpha else "ns"))
        pair_str = f"{m1} vs {m2}"
        print(f"  {pair_str:<50} {stat:>10.2f} {p:>10.4f} {sig:>5}")

    # One-way repeated-measures via paired t-tests (parametric)
    print(f"\nPairwise paired t-tests (Bonferroni α={corrected_alpha:.4f}):")
    print(f"  {'Pair':<50} {'t-stat':>10} {'p-value':>10} {'sig':>5}")
    print("  " + "-" * 78)
    for m1, m2 in pairs:
        stat, p = stats.ttest_rel(aligned[m1], aligned[m2])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < corrected_alpha else "ns"))
        pair_str = f"{m1} vs {m2}"
        print(f"  {pair_str:<50} {stat:>10.4f} {p:>10.4f} {sig:>5}")

print(f"\n{'=' * 60}")
print("Significance codes: *** p<0.001  ** p<0.01  * p<0.05  ns not significant")
print("Note: Bonferroni correction applied for", len(pairs), "pairwise comparisons per metric.")
