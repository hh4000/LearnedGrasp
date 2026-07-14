"""
Gaze-channel ablation: compare 3PC GNN results WITH vs WITHOUT the gaze channel.

The gaze channel is not stored as an explicit flag, but it shows up in the model
config (params/params_<run_id>.json) as extra node-input features:

    node features = pos(3) + normals(3) + gaze gaussian-weights(P)

  • with gaze    → larger input dim  (GCN models: 11,  EdgeConv/v2: 2*11 = 22)
  • without gaze → smaller input dim (GCN models:  6,  EdgeConv/v2: 2*6  = 12)

For each model we read the first conv layer's input dimension from the architecture
string, group the 3PC runs by that dimension, and label the LARGER dim "with gaze"
and the SMALLER "without gaze". No magic constants — it adapts per model.

With-gaze and without-gaze batches share the same master seed, so trials with the
same seed are paired across the two conditions for the paired tests.

Run:
    python gaze_ablation_comparison.py
"""
import json
import glob
import re
import pathlib
import importlib.util
from collections import defaultdict

import numpy as np
from scipy import stats

RESULTS_DIR = "results"
PARAMS_DIR  = "params"
N_PCS       = 3          # 3PC prediction
ALPHA       = 0.05

# PCA explained-variance ratios (same list used in dl_statistical_significance.py).
# Only the first N_PCS are used, renormalised to sum to 1, to form a weighted R̄.
PCA_WEIGHTS = np.array([
    0.33153246, 0.13009581, 0.11191947, 0.08375275, 0.07213102,
    0.06736816, 0.05068327, 0.04505967, 0.03534769, 0.0220468,
    0.01866559, 0.01226277, 0.01028329, 0.00615714, 0.0026941,
])

# ── Optional paper-formatted plotting (graceful fallback to plain matplotlib) ──
try:
    import matplotlib.pyplot as plt
    _spec = importlib.util.spec_from_file_location(
        "configure_fig",
        pathlib.Path(__file__).resolve().parent.parent.parent / "PaperFormatting" / "configure-fig.py",
    )
    _cfig = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_cfig)
    PaperPlotter, CONFIG = _cfig.PaperPlotter, _cfig.CONFIG
    _HAVE_PAPER = True
except Exception:
    PaperPlotter = CONFIG = None
    _HAVE_PAPER = False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None


# ── Helpers ────────────────────────────────────────────────────────────────────
def first_conv_input_dim(architecture: str) -> int | None:
    """First node-input feature dimension declared in the architecture string.

    GCNConv layers print `GCNConv(in, out)`; EdgeConv etc. print the input width
    as `in_features=<n>` on their first Linear. We take whichever appears first.
    """
    m = re.search(r"GCNConv\((\d+)|in_features=(\d+)", architecture)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def sig(p: float) -> str:
    if np.isnan(p):     return "  ?"
    if p < 0.001:       return "**"
    if p < 0.01:        return " *"
    return " ns"


def weighted_rbar(r2_matrix: np.ndarray) -> np.ndarray:
    """PCA-explained-variance-weighted mean R² per trial. (n_trials, n_pcs) → (n_trials,)."""
    w = PCA_WEIGHTS[:r2_matrix.shape[1]]
    w = w / w.sum()
    return r2_matrix @ w


# ── Load 3PC summaries and attach their config input dim ─────────────────────────
records = []
missing_params = 0
for path in glob.glob(f"{RESULTS_DIR}/summary_*.json"):
    with open(path) as f:
        d = json.load(f)
    if len(d.get("r2_per_joint", [])) != N_PCS:
        continue  # not a 3PC run

    pj = pathlib.Path(PARAMS_DIR) / f"params_{d['run_id']}.json"
    if not pj.exists():
        missing_params += 1
        continue
    with open(pj) as f:
        params = json.load(f)
    dim = first_conv_input_dim(params.get("architecture", ""))
    if dim is None:
        missing_params += 1
        continue

    m = re.match(r"(\d{6}_\d{6})", d["run_id"])
    d["_ts"]      = m.group(1) if m else ""
    d["_in_dim"]  = dim
    d["_r2"]      = np.array(d["r2_per_joint"], dtype=float)
    records.append(d)

print("=" * 78)
print("GAZE-CHANNEL ABLATION — 3PC GNN results  (with gaze vs without gaze)")
print("=" * 78)
print(f"3PC summaries found: {len(records)}"
      + (f"   ({missing_params} skipped: no params / unparseable architecture)"
         if missing_params else ""))

# ── Group by model, then split each model's runs by input dim ────────────────────
by_model: dict[str, list[dict]] = defaultdict(list)
for r in records:
    by_model[r["model"]].append(r)

SEP  = "-" * 78
comparisons = []   # (model, gaze_runs, nogaze_runs, gaze_dim, nogaze_dim)

print(f"\nDetected groups per model (input dim → #3PC runs):")
for model in sorted(by_model):
    dims = defaultdict(int)
    for r in by_model[model]:
        dims[r["_in_dim"]] += 1
    dims_str = "  ".join(f"in={d}:{n}" for d, n in sorted(dims.items()))
    print(f"  {model:20s}  {dims_str}")

    distinct = sorted(dims)
    if len(distinct) < 2:
        continue
    gaze_dim, nogaze_dim = max(distinct), min(distinct)   # larger dim carries gaze
    gaze_runs   = [r for r in by_model[model] if r["_in_dim"] == gaze_dim]
    nogaze_runs = [r for r in by_model[model] if r["_in_dim"] == nogaze_dim]
    comparisons.append((model, gaze_runs, nogaze_runs, gaze_dim, nogaze_dim))

if not comparisons:
    print("\nNo model has BOTH a with-gaze and a without-gaze 3PC group yet.")
    print("Once the no-gaze batch (n_in=6) finishes, re-run this script.")
    raise SystemExit(0)


def stack(runs: list[dict]) -> tuple[np.ndarray, list[int]]:
    """Return (r2_matrix (n_trials, N_PCS), seeds) for a run list."""
    return np.array([r["_r2"] for r in runs]), [r["seed"] for r in runs]


def paired_arrays(a_runs, b_runs, col):
    """Values for condition A and B paired on shared seeds; falls back to unpaired."""
    a_by_seed = {r["seed"]: r for r in a_runs}
    b_by_seed = {r["seed"]: r for r in b_runs}
    shared = sorted(set(a_by_seed) & set(b_by_seed))
    if len(shared) >= 2:
        a = np.array([col(a_by_seed[s]) for s in shared])
        b = np.array([col(b_by_seed[s]) for s in shared])
        return a, b, True, len(shared)
    a = np.array([col(r) for r in a_runs])
    b = np.array([col(r) for r in b_runs])
    return a, b, False, 0


def compare_metric(label, gaze_runs, nogaze_runs, col):
    """Print one row comparing a scalar metric between conditions."""
    a, b, paired, n_shared = paired_arrays(gaze_runs, nogaze_runs, col)
    ga = np.array([col(r) for r in gaze_runs])
    nb = np.array([col(r) for r in nogaze_runs])
    if paired:
        diff = a - b
        t_s, t_p = stats.ttest_rel(a, b)
        try:
            w_s, w_p = stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            w_p = float("nan")
        tag = f"paired n={n_shared}"
    else:
        diff = ga.mean() - nb.mean()
        diff = np.array([diff])
        t_s, t_p = stats.ttest_ind(ga, nb)
        try:
            w_s, w_p = stats.mannwhitneyu(ga, nb, alternative="two-sided")
        except ValueError:
            w_p = float("nan")
        tag = "unpaired"
    print(f"  {label:<8} {ga.mean():>+9.4f} {ga.std():>8.4f} "
          f"{nb.mean():>+9.4f} {nb.std():>8.4f} {diff.mean():>+9.4f} "
          f"{t_p:>8.4f} {sig(t_p):>4} {w_p:>8.4f} {sig(w_p):>4}  {tag}")


for model, gaze_runs, nogaze_runs, gaze_dim, nogaze_dim in comparisons:
    print(f"\n{'=' * 78}")
    print(f"MODEL: {model}")
    print(f"  with gaze    : in_dim={gaze_dim:<3}  trials={len(gaze_runs)}")
    print(f"  without gaze : in_dim={nogaze_dim:<3}  trials={len(nogaze_runs)}")
    print("=" * 78)

    g_mat, _ = stack(gaze_runs)
    n_mat, _ = stack(nogaze_runs)

    print(f"\n  {'metric':<8} {'gaze μ':>9} {'gaze σ':>8} {'no-gaze μ':>9} {'no-gaze σ':>8} "
          f"{'Δ(g-ng)':>9} {'t p':>8} {'sig':>4} {'W p':>8} {'sig':>4}")
    print("  " + SEP)

    # Per-PC R²
    for pc in range(N_PCS):
        compare_metric(f"PC{pc+1} R²", gaze_runs, nogaze_runs, lambda r, pc=pc: r["_r2"][pc])

    # Weighted R̄ across the 3 PCs
    compare_metric("R̄ (wgt)", gaze_runs, nogaze_runs,
                   lambda r: float(weighted_rbar(r["_r2"][None, :])[0]))
    # Unweighted mean R²
    compare_metric("R̄ (mean)", gaze_runs, nogaze_runs, lambda r: float(r["r2_mean_test"]))
    # Test loss (lower is better)
    compare_metric("test loss", gaze_runs, nogaze_runs, lambda r: float(r["test_loss"]))

print(f"\n{'=' * 78}")
print("Δ(g-ng) > 0 means WITH gaze scores higher (better for R², worse for loss).")
print("Sig codes: *** p<0.001  ** p<0.01  * p<0.05  ns not significant")
print("Paired tests use trials sharing a seed across the two conditions.")

# ── Plot: per-PC + R̄ boxplots, gaze vs no-gaze, ONE FIGURE PER MODEL ────────────
if plt is None:
    print("\n(matplotlib unavailable — skipping plot)")
    raise SystemExit(0)

from matplotlib.patches import Patch

metrics = [(f"PC{pc+1}", lambda m, pc=pc: m[:, pc]) for pc in range(N_PCS)]
metrics.append(("R̄", lambda m: weighted_rbar(m)))

if _HAVE_PAPER:
    pal = CONFIG.data_palette(2)
    text_color, edge_color = CONFIG.TEXT_COLOR, CONFIG.EDGE_COLOR
else:
    pal = ["#4C72B0", "#DD8452"]
    text_color, edge_color = "black", "gray"

handles = [Patch(facecolor=pal[0], alpha=0.65, label="with gaze"),
           Patch(facecolor=pal[1], alpha=0.65, label="without gaze")]

rng = np.random.default_rng(0)
print()
for model, gaze_runs, nogaze_runs, gaze_dim, nogaze_dim in comparisons:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    g_mat, _ = stack(gaze_runs)
    n_mat, _ = stack(nogaze_runs)

    positions, data, colors, centers, labels = [], [], [], [], []
    for i, (name, fn) in enumerate(metrics):
        base = i * 3
        gv, nv = fn(g_mat), fn(n_mat)
        data += [gv, nv]
        positions += [base + 1, base + 2]
        colors += [pal[0], pal[1]]
        centers.append(base + 1.5)
        labels.append(name)

    bp = ax.boxplot(data, positions=positions, widths=0.7,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=text_color, linewidth=1.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    for pos, vals, c in zip(positions, data, colors):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(pos + jitter, vals, s=18, color=c,
                   edgecolors="white", linewidths=0.3, zorder=3)

    # Significance stars per metric (paired weighted/per-PC where possible)
    for i, (name, fn) in enumerate(metrics):
        a, b, paired, _ = paired_arrays(
            gaze_runs, nogaze_runs,
            (lambda r, k=i: weighted_rbar(r["_r2"][None, :])[0]) if name == "R̄"
            else (lambda r, k=i: r["_r2"][k]),
        )
        if paired and len(a) >= 2:
            _, p = stats.ttest_rel(a, b)
        else:
            _, p = stats.ttest_ind(fn(g_mat), fn(n_mat))
        s = sig(p).strip()
        if s == "ns":
            continue
        top = max(np.max(fn(g_mat)), np.max(fn(n_mat)))
        ax.text(i * 3 + 1.5, top, s, ha="center", va="bottom",
                fontsize=10, color=text_color, fontweight="bold")

    ax.axhline(0, color=edge_color, linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              facecolor="white", edgecolor=edge_color)
    title = f"{model} — 3PC test R²\n(gaze in={gaze_dim} vs no-gaze in={nogaze_dim})"
    if _HAVE_PAPER:
        pp = PaperPlotter(fig, ax)
        pp.set_title(title)
        pp.set_ylabel("test R²")
    else:
        ax.set_title(title)
        ax.set_ylabel("test R²")

    fig.tight_layout()
    out_path = f"{RESULTS_DIR}/gaze_ablation_3pc_{model}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")

# ── Plot 2: PCA-weighted R̄ only — gaze vs no-gaze, all models in one panel ──────
fig2, ax2 = plt.subplots(figsize=(1.6 * len(comparisons) + 2, 5))

positions, data, colors, centers, labels = [], [], [], [], []
for i, (model, gaze_runs, nogaze_runs, gaze_dim, nogaze_dim) in enumerate(comparisons):
    g_mat, _ = stack(gaze_runs)
    n_mat, _ = stack(nogaze_runs)
    base = i * 3
    data += [weighted_rbar(g_mat), weighted_rbar(n_mat)]
    positions += [base + 1, base + 2]
    colors += [pal[0], pal[1]]
    centers.append(base + 1.5)
    labels.append(model.replace("MANOGraspGNN", ""))

bp = ax2.boxplot(data, positions=positions, widths=0.7,
                 patch_artist=True, showfliers=False,
                 medianprops=dict(color=text_color, linewidth=1.4))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.65)
for pos, vals, c in zip(positions, data, colors):
    jitter = rng.uniform(-0.18, 0.18, size=len(vals))
    ax2.scatter(pos + jitter, vals, s=20, color=c,
                edgecolors="white", linewidths=0.3, zorder=3)

# Paired significance star per model (on the weighted R̄)
for i, (model, gaze_runs, nogaze_runs, _, _) in enumerate(comparisons):
    a, b, paired, _ = paired_arrays(
        gaze_runs, nogaze_runs, lambda r: weighted_rbar(r["_r2"][None, :])[0])
    if paired and len(a) >= 2:
        _, p = stats.ttest_rel(a, b)
    else:
        g_mat, _ = stack(gaze_runs); n_mat, _ = stack(nogaze_runs)
        _, p = stats.ttest_ind(weighted_rbar(g_mat), weighted_rbar(n_mat))
    s = sig(p).strip()
    if s == "ns":
        continue
    top = max(np.max(data[2 * i]), np.max(data[2 * i + 1]))
    ax2.text(centers[i], top + 0.004, s, ha="center", va="bottom",
             fontsize=11, color=text_color, fontweight="bold")

ax2.axhline(0, color=edge_color, linewidth=0.9, linestyle="--", zorder=1)
ax2.set_xticks(centers)
ax2.set_xticklabels(labels)
ax2.legend(handles=handles, loc="best", fontsize=8,
           facecolor="white", edgecolor=edge_color)
if _HAVE_PAPER:
    pp2 = PaperPlotter(fig2, ax2)
    pp2.set_ylabel("R̄  (PCA-weighted test R²)")
    pp2.set_title("Gaze-channel ablation — PCA-weighted R̄ (3PC)")
else:
    ax2.set_ylabel("R̄  (PCA-weighted test R²)")
    ax2.set_title("Gaze-channel ablation — PCA-weighted R̄ (3PC)")

fig2.tight_layout()
out_path2 = f"{RESULTS_DIR}/gaze_ablation_rbar.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Plot saved → {out_path2}")
