"""
Statistical significance testing for DL training results.

Cases are (model, n_pcs) groups. For each case:
  - Per-PC one-sample tests vs R²=0  (is each PC predicted better than chance?)
  - Pairwise tests between cases with the same n_pcs  (which architecture is better?)

Uses the most recent 10 trials per case, sorted by timestamp.
Trials with the same seed are paired across models for the pairwise tests.
"""
import json
import glob
import re
import pathlib
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
from collections import defaultdict

# Load paper-formatting helpers (hyphen in filename requires importlib)
_spec = importlib.util.spec_from_file_location(
    "configure_fig",
    pathlib.Path(__file__).resolve().parent.parent / "PaperFormatting" / "configure-fig.py",
)
_cfig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfig)
PaperPlotter, CONFIG = _cfig.PaperPlotter, _cfig.CONFIG

RESULTS_DIR = "results"
N_TRIALS    = 10
ALPHA       = 0.05

PCA_WEIGHTS = np.array([
    0.33153246, 0.13009581, 0.11191947, 0.08375275, 0.07213102,
    0.06736816, 0.05068327, 0.04505967, 0.03534769, 0.0220468,
    0.01866559, 0.01226277, 0.01028329, 0.00615714, 0.0026941,
])

# ── Load all summary JSONs ────────────────────────────────────────────────────
records = []
for path in glob.glob(f"{RESULTS_DIR}/summary_*.json"):
    with open(path) as f:
        d = json.load(f)
    # extract timestamp from filename for recency sorting
    m = re.search(r"summary_(\d{6}_\d{6})_", path)
    d["_ts"]   = m.group(1) if m else ""
    d["_path"] = path
    d["n_pcs"] = len(d["r2_per_joint"])
    records.append(d)

# ── Group by (model, n_pcs), keep most recent N_TRIALS ───────────────────────
groups = defaultdict(list)
for r in records:
    groups[(r["model"], r["n_pcs"])].append(r)

cases = {}
for key, runs in groups.items():
    runs_sorted = sorted(runs, key=lambda r: r["_ts"], reverse=True)[:N_TRIALS]
    if len(runs_sorted) < N_TRIALS:
        print(f"Warning: {key} only has {len(runs_sorted)} trials (need {N_TRIALS}), skipping.")
        continue
    # shape: (n_trials, n_pcs)
    r2_matrix = np.array([r["r2_per_joint"] for r in runs_sorted])
    # sort by seed so paired tests across models align on the same random init
    seeds  = [r["seed"] for r in runs_sorted]
    order  = np.argsort(seeds)
    cases[key] = {"r2": r2_matrix[order], "seeds": [seeds[i] for i in order]}

case_keys = sorted(cases.keys())
print(f"Cases loaded: {len(case_keys)}")
for k in case_keys:
    print(f"  {k[0]:25s}  n_pcs={k[1]}  trials={cases[k]['r2'].shape[0]}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def sig(p, threshold):
    if p < 0.001:       return "**"
    elif p < 0.01:      return "* "
    # elif p < threshold: return "*  "
    else:               return "ns"

def weighted_r2_bar(r2_matrix):
    """Weighted average R² per trial using PCA explained-variance weights."""
    n_pcs = r2_matrix.shape[1]
    w = PCA_WEIGHTS[:n_pcs]
    return r2_matrix @ w

SEP  = "=" * 72
SEP2 = "-" * 72

# ── Per-case: one-sample tests vs 0 per PC ───────────────────────────────────
print(f"\n{SEP}")
print("ONE-SAMPLE TESTS vs R²=0  (per PC, Bonferroni corrected within each case)")
print(SEP)

for key in case_keys:
    model, n_pcs = key
    r2 = cases[key]["r2"]   # (n_trials, n_pcs)
    corrected_alpha = ALPHA / n_pcs

    print(f"\n{model}  |  n_pcs={n_pcs}  |  Bonferroni α={corrected_alpha:.4f}")
    print(f"  {'PC':<5} {'mean R²':>9} {'std':>7} {'t-stat':>8} {'t p':>8} {'sig(t)':>7}  {'W-stat':>7} {'W p':>8} {'sig(W)':>7}")
    print("  " + SEP2)

    for pc in range(n_pcs):
        vals = r2[:, pc]
        t_s, t_p = stats.ttest_1samp(vals, popmean=0, alternative="two-sided")
        try:
            w_s, w_p = stats.wilcoxon(vals, alternative="two-sided")
        except ValueError:
            w_s, w_p = float("nan"), float("nan")
        print(f"  {pc+1:<5} {vals.mean():>+9.4f} {vals.std():>7.4f} "
              f"{t_s:>8.3f} {t_p:>8.4f} {sig(t_p, corrected_alpha):>7}  "
              f"{w_s:>7.1f} {w_p:>8.4f} {sig(w_p, corrected_alpha):>7}")

    print("  " + SEP2)
    rbar = weighted_r2_bar(r2)   # (n_trials,)
    t_s, t_p = stats.ttest_1samp(rbar, popmean=0, alternative="two-sided")
    try:
        w_s, w_p = stats.wilcoxon(rbar, alternative="two-sided")
    except ValueError:
        w_s, w_p = float("nan"), float("nan")
    print(f"  {'R̄':<5} {rbar.mean():>+9.3f} {rbar.std():>7.3f} "
          f"{t_s:>8.3f} {t_p:>8.4f} {sig(t_p, corrected_alpha):>7}  "
          f"{w_s:>7.1f} {w_p:>8.4f} {sig(w_p, corrected_alpha):>7}")

# ── Pairwise tests between same-n_pcs cases, per PC ─────────────────────────
by_npcs = defaultdict(list)
for key in case_keys:
    by_npcs[key[1]].append(key)

print(f"\n{SEP}")
print("PAIRWISE TESTS between models  (same n_pcs, per PC, Bonferroni corrected)")
print(SEP)

for n_pcs, keys in sorted(by_npcs.items()):
    if len(keys) < 2:
        continue
    pairs = list(combinations(keys, 2))
    corrected_alpha = ALPHA / len(pairs)

    print(f"\nn_pcs={n_pcs}  |  {len(pairs)} pairs  |  Bonferroni α={corrected_alpha:.4f}")

    for pc in range(n_pcs):
        print(f"\n  PC {pc+1}")
        print(f"  {'Pair':<52} {'t-stat':>7} {'t p':>8} {'sig(t)':>7}  {'W-stat':>7} {'W p':>8} {'sig(W)':>7}")
        print("  " + SEP2)

        for k1, k2 in pairs:
            a = cases[k1]["r2"][:, pc]
            b = cases[k2]["r2"][:, pc]
            t_s, t_p = stats.ttest_rel(a, b)
            try:
                w_s, w_p = stats.wilcoxon(a, b, alternative="two-sided")
            except ValueError:
                w_s, w_p = float("nan"), float("nan")
            label = f"{k1[0]} vs {k2[0]}"
            print(f"  {label:<52} {t_s:>7.3f} {t_p:>8.4f} {sig(t_p, corrected_alpha):>7}  "
                  f"{w_s:>7.1f} {w_p:>8.4f} {sig(w_p, corrected_alpha):>7}")

    print(f"\n  R̄  (weighted average across PCs)")
    print(f"  {'Pair':<52} {'t-stat':>7} {'t p':>8} {'sig(t)':>7}  {'W-stat':>7} {'W p':>8} {'sig(W)':>7}")
    print("  " + SEP2)
    for k1, k2 in pairs:
        a = weighted_r2_bar(cases[k1]["r2"])
        b = weighted_r2_bar(cases[k2]["r2"])
        t_s, t_p = stats.ttest_rel(a, b)
        try:
            w_s, w_p = stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            w_s, w_p = float("nan"), float("nan")
        label = f"{k1[0]} vs {k2[0]}"
        print(f"  {label:<52} {t_s:>7.3f} {t_p:>8.4f} {sig(t_p, corrected_alpha):>7}  "
              f"{w_s:>7.1f} {w_p:>8.4f} {sig(w_p, corrected_alpha):>7}")

print(f"\n{SEP}")
print("Sig codes: *** p<0.001  ** p<0.01  * corrected α  ns not significant")
print(f"Pairwise tests are paired by seed (same random init across models).")

# ── Plot R̄ comparisons ────────────────────────────────────────────────────────
npcs_groups = [(n, ks) for n, ks in sorted(by_npcs.items())]
fig, axes = plt.subplots(1, len(npcs_groups),
                         figsize=(4.5 * len(npcs_groups), 5), squeeze=False,
                         sharey=True)

# Pass 1: precompute per-subplot data and bracket assignments so we can
# determine a single shared y-range before drawing anything.
subplots = []
for n_pcs, keys in npcs_groups:
    keys_sorted = sorted(keys)
    n_models    = len(keys_sorted)
    rbar_data   = {k: weighted_r2_bar(cases[k]["r2"]) for k in keys_sorted}
    data        = [rbar_data[k] for k in keys_sorted]
    palette     = CONFIG.data_palette(n_models)

    assignments = []
    if n_models >= 2:
        idx_pairs          = list(combinations(range(n_models), 2))
        corrected_alpha_pl = ALPHA / len(idx_pairs)
        occupied           = []
        for i1, i2 in sorted(idx_pairs, key=lambda p: p[1] - p[0]):
            k1, k2 = keys_sorted[i1], keys_sorted[i2]
            _, t_p = stats.ttest_rel(rbar_data[k1], rbar_data[k2])
            s      = sig(t_p, corrected_alpha_pl).strip()
            if s == "ns":
                continue
            x1, x2 = i1 + 1, i2 + 1
            level   = 0
            while any(ox1 <= x2 and ox2 >= x1 and ol == level
                      for ox1, ox2, ol in occupied):
                level += 1
            occupied.append((x1, x2, level))
            assignments.append((x1, x2, level, s))

    subplots.append(dict(n_pcs=n_pcs, keys_sorted=keys_sorted,
                         n_models=n_models, rbar_data=rbar_data,
                         data=data, palette=palette, assignments=assignments))

# Global y-limits shared across all panels
all_vals    = np.concatenate([np.concatenate(sp["data"]) for sp in subplots])
global_ymin = all_vals.min()
global_ymax = all_vals.max()
step        = max((global_ymax - global_ymin) * 0.14, 0.01)
base_y      = global_ymax + step * 0.4
max_level   = max(
    (max((a[2] for a in sp["assignments"]), default=-1) for sp in subplots),
    default=-1,
)
final_top = base_y + (max_level + 1.8) * step

# Pass 2: draw all subplots with the shared scale already known.
rng = np.random.default_rng(0)
for ax_idx, sp in enumerate(subplots):
    ax          = axes[0, ax_idx]
    pp          = PaperPlotter(fig, ax)
    keys_sorted = sp["keys_sorted"]
    n_models    = sp["n_models"]
    rbar_data   = sp["rbar_data"]
    data        = sp["data"]
    palette     = sp["palette"]

    bp = ax.boxplot(
        data, patch_artist=True, widths=0.45, showfliers=False,
        medianprops=dict(color=CONFIG.TEXT_COLOR, linewidth=1.5),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    for i, k in enumerate(keys_sorted):
        jitter = rng.uniform(-0.12, 0.12, size=len(rbar_data[k]))
        ax.scatter(i + 1 + jitter, rbar_data[k],
                   s=22, alpha=0.85, color=palette[i],
                   edgecolors="white", linewidths=0.3, zorder=3)

    for x1, x2, level, s in sp["assignments"]:
        h   = base_y + level * step
        tip = step * 0.22
        ax.plot([x1, x1, x2, x2], [h, h + tip, h + tip, h],
                lw=0.9, c=CONFIG.TEXT_COLOR, clip_on=False)
        ax.text((x1 + x2) / 2, h + tip, s,
                ha="center", va="bottom", fontsize=8.5, color=CONFIG.TEXT_COLOR)

    ax.axhline(0, color=CONFIG.EDGE_COLOR, linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xticks(range(1, n_models + 1))
    ax.set_xticklabels([k[0] for k in keys_sorted], rotation=35, ha="right", fontsize=8)
    pp.set_title(f"n_pcs = {sp['n_pcs']}")
    if ax_idx == 0:
        pp.set_ylabel("R̄  (PCA-weighted R²)")

# Set once — sharey propagates it to all panels.
axes[0, 0].set_ylim(bottom=global_ymin - step * 0.3, top=final_top)

fig.suptitle("R̄ across models  (Bonferroni-corrected paired t-test)",
             color=CONFIG.TEXT_COLOR, fontsize=11, y=1.01)
fig.tight_layout()
out_path = f"{RESULTS_DIR}/rbar_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved → {out_path}")

# ── Plot 2: R̄ vs n_pcs per model ─────────────────────────────────────────────
by_model = defaultdict(list)
for key in case_keys:
    by_model[key[0]].append(key)

all_model_names = sorted(by_model)
palette_m = CONFIG.data_palette(len(all_model_names))

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
pp2 = PaperPlotter(fig2, ax2)

for mi, model_name in enumerate(all_model_names):
    keys_m = sorted(by_model[model_name], key=lambda k: k[1])
    color  = palette_m[mi]

    xs    = [k[1] for k in keys_m]
    rbars = [weighted_r2_bar(cases[k]["r2"]) for k in keys_m]
    means = np.array([r.mean() for r in rbars])
    stds  = np.array([r.std()  for r in rbars])

    ax2.plot(xs, means, marker="o", markersize=5, linewidth=1.4,
             color=color, label=model_name, zorder=3)
    ax2.fill_between(xs, means - stds, means + stds,
                     color=color, alpha=0.15, zorder=2)

    # Annotate significant consecutive n_pcs steps
    for i in range(len(keys_m) - 1):
        a, b   = rbars[i], rbars[i + 1]
        sa, sb = cases[keys_m[i]]["seeds"], cases[keys_m[i + 1]]["seeds"]
        common = [s for s in sa if s in sb]
        if len(common) >= 5:
            ia = [sa.index(s) for s in common]
            ib = [sb.index(s) for s in common]
            _, t_p = stats.ttest_rel(a[ia], b[ib])
        else:
            _, t_p = stats.ttest_ind(a, b)
        s = sig(t_p, ALPHA).strip()
        if s == "ns":
            continue
        mid_x = (xs[i] + xs[i + 1]) / 2
        ann_y = max(means[i] + stds[i], means[i + 1] + stds[i + 1])
        ax2.text(mid_x, ann_y, s, ha="center", va="bottom",
                 fontsize=8, color=color, fontweight="bold")

ax2.axhline(0, color=CONFIG.EDGE_COLOR, linewidth=0.9, linestyle="--", zorder=1)
ax2.set_xticks(sorted({k[1] for k in case_keys}))
pp2.set_xlabel("n_pcs")
pp2.set_ylabel("R̄  (PCA-weighted R²)")
pp2.set_title("R̄ vs number of PCs  (mean ± std, 10 trials)")
ax2.legend(facecolor="white", edgecolor=CONFIG.EDGE_COLOR,
           fontsize=8, labelcolor=CONFIG.TEXT_COLOR)
fig2.tight_layout()
out_path2 = f"{RESULTS_DIR}/rbar_vs_npcs.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved → {out_path2}")
