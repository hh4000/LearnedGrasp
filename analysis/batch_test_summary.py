"""
Summarise batches of DL test trials that share the same parameters (except seed).

A "batch" is the set of trials whose parameter signature is identical once the
per-trial fields (run_id, seed, trial, date) are removed. The model architecture
is part of that signature, so different architectures never share a batch.

Data sources (matched on the `run_id` field, which is identical in both):
  results/summary_<run_id>.json   - per-trial test metrics (r2_per_joint, seed, ...)
  params/params_<ts>.json         - full parameters for that run (architecture, ...)

For every batch the report gives:
  - Model architecture, with number of input and output channels (parsed from
    the architecture string: input = in-dim of the first conv/GNN layer,
    output = final out_features, i.e. the number of predicted PCs).
  - PC-weighted R2 (mean & std over trials), using the PCA explained-variance
    weights WITHOUT renormalising them to sum to 1 (i.e. PCA_WEIGHTS[:n_pcs]).
  - Per-PC R2 (mean & std over trials) for each principal component.
  - Number of trials in the batch.
  - Date of the first and last trial.
  - Number of unique seeds in the batch.

All means/stds are rounded to three significant figures.

Usage:
    python batch_test_summary.py                 # report + CSV for all batches
    python batch_test_summary.py --min-trials 5  # only batches with >=5 trials
    python batch_test_summary.py --no-csv        # console report only
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np

RESULTS_DIR = "results"
PARAMS_DIR = "params"

# PCA explained-variance ratios (same list used in dl_statistical_significance.py).
# Only the first n_pcs are used and they are deliberately NOT renormalised to
# sum to 1 -> a batch using fewer PCs gets a correspondingly smaller weighted R2.
PCA_WEIGHTS = np.array([
    0.33153246, 0.13009581, 0.11191947, 0.08375275, 0.07213102,
    0.06736816, 0.05068327, 0.04505967, 0.03534769, 0.0220468,
    0.01866559, 0.01226277, 0.01028329, 0.00615714, 0.0026941,
])

# Per-trial fields that must NOT be part of the batch signature.
EXCLUDE_FROM_KEY = {"run_id", "seed", "trial", "date"}


def sig3(x):
    """Format a number to three significant figures."""
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{float(x):.3g}"


def load_params():
    """Map run_id -> params dict (the params file carries the authoritative run_id)."""
    by_run = {}
    for path in glob.glob(os.path.join(PARAMS_DIR, "params_*.json")):
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        rid = d.get("run_id")
        if rid:
            by_run[rid] = d
    return by_run


def parse_channels(arch):
    """(in_channels, out_channels) parsed from the printed architecture string.

    in_channels  : in-dim of the first learnable layer
                   (GCNConv(<in>, .) / Conv2d(<in>, .) / first in_features=<in>).
    out_channels : the final out_features (number of predicted PCs); falls back
                   to the last Conv2d output for purely convolutional models.
    """
    in_ch = out_ch = None

    first = re.search(r"GCNConv\((\d+),", arch) \
        or re.search(r"Conv2d\((\d+),", arch) \
        or re.search(r"in_features=(\d+)", arch)
    if first:
        in_ch = int(first.group(1))

    outs = re.findall(r"out_features=(\d+)", arch)
    if outs:
        out_ch = int(outs[-1])
    else:
        conv_outs = re.findall(r"Conv2d\(\d+,\s*(\d+)", arch)
        if conv_outs:
            out_ch = int(conv_outs[-1])

    return in_ch, out_ch


def parse_date(params, run_id):
    """Best-effort trial timestamp: params 'date' (ISO) else the run_id prefix."""
    raw = params.get("date")
    if raw:
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    m = re.match(r"(\d{6}_\d{6})", run_id or "")
    if m:
        try:
            return datetime.strptime(m.group(1), "%y%m%d_%H%M%S")
        except ValueError:
            pass
    return None


def batch_key(params):
    """Canonical signature of a batch: all params except the per-trial fields."""
    sig = {k: v for k, v in params.items() if k not in EXCLUDE_FROM_KEY}
    return json.dumps(sig, sort_keys=True, default=str)


def model_name(params, run_id):
    """Model name, falling back to the architecture string or run_id when absent."""
    if params.get("model"):
        return params["model"]
    arch = params.get("architecture", "")
    if arch:
        return arch.split("(", 1)[0]
    m = re.search(r"_(MANO\w+?)_t\d+$", run_id or "")
    return m.group(1) if m else "?"


def collect_batches(params_by_run):
    """Group every summary trial into its batch."""
    batches = defaultdict(lambda: {"trials": [], "params": None})
    for path in glob.glob(os.path.join(RESULTS_DIR, "summary_*.json")):
        try:
            s = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        rid = s.get("run_id")
        params = params_by_run.get(rid)
        if params is None:
            continue  # no parameters -> cannot place it in a batch
        key = batch_key(params)
        b = batches[key]
        b["params"] = params  # representative (identical across the batch)
        b["trials"].append({
            "run_id": rid,
            "seed": s.get("seed"),
            "r2": np.asarray(s.get("r2_per_joint", []), dtype=float),
            "date": parse_date(params, rid),
        })
    return batches


def summarise(batch):
    """Compute the per-batch summary metrics."""
    trials = batch["trials"]
    params = batch["params"]

    r2 = np.vstack([t["r2"] for t in trials])  # (n_trials, n_pcs)
    n_trials, n_pcs = r2.shape

    # PC-weighted R2 per trial, weights NOT renormalised to sum to 1.
    w = PCA_WEIGHTS[:n_pcs]
    weighted = r2 @ w  # (n_trials,)

    seeds = [t["seed"] for t in trials if t["seed"] is not None]
    dates = [t["date"] for t in trials if t["date"] is not None]
    in_ch, out_ch = parse_channels(params.get("architecture", ""))

    return {
        "model": model_name(params, trials[0]["run_id"]),
        "in_channels": in_ch,
        "out_channels": out_ch,
        "n_pcs": n_pcs,
        "n_trials": n_trials,
        "n_unique_seeds": len(set(seeds)) if len(seeds) == n_trials else None,
        "date_first": min(dates) if dates else None,
        "date_last": max(dates) if dates else None,
        "weighted_mean": weighted.mean(),
        "weighted_std": weighted.std(),
        "per_pc_mean": r2.mean(axis=0),
        "per_pc_std": r2.std(axis=0),
        "params": params,
    }


def param_signature_line(params):
    """One-line human-readable digest of the discriminating parameters."""
    opt = params.get("optimizer", {})
    lr = opt.get("lr", opt.get("kwargs", {}).get("lr"))
    wd = opt.get("weight_decay", opt.get("kwargs", {}).get("weight_decay"))
    loss = params.get("loss_function")
    if isinstance(loss, dict):
        loss = loss.get("name")
    parts = [
        f"loss={loss}",
        f"opt={opt.get('name')}",
        f"lr={lr}",
        f"wd={wd}",
        f"drop_gnn={params.get('dropout_gnn')}",
        f"drop_fc={params.get('dropout_fc')}",
        f"max_epochs={params.get('max_epochs')}",
    ]
    if "dropout_2d" in params:
        parts.append(f"drop_2d={params['dropout_2d']}")
    es = params.get("early_stopping")
    if isinstance(es, dict):
        parts.append(f"early_stop(pat={es.get('patience')},min={es.get('min_epochs')})")
    return "  ".join(str(p) for p in parts)


def fmt_date(d):
    return d.strftime("%Y-%m-%d %H:%M:%S") if d else "?"


def print_report(summaries):
    sep = "=" * 78
    print(sep)
    print(f"BATCH TEST SUMMARY  ({len(summaries)} batches)")
    print("PC-weighted R2 uses PCA explained-variance weights, NOT normalised to sum to 1.")
    print("All means/stds rounded to 3 significant figures.")
    print(sep)

    for s in summaries:
        in_ch = s["in_channels"] if s["in_channels"] is not None else "?"
        out_ch = s["out_channels"] if s["out_channels"] is not None else "?"
        print(f"\n{s['model']}   |   in_ch={in_ch}  out_ch={out_ch}   |   "
              f"n_pcs={s['n_pcs']}   |   trials={s['n_trials']}")
        print(f"  {param_signature_line(s['params'])}")
        seeds = s["n_unique_seeds"]
        print(f"  unique seeds : {seeds if seeds is not None else 'unknown'}")
        print(f"  date range   : {fmt_date(s['date_first'])}  ->  {fmt_date(s['date_last'])}")
        print(f"  PC-weighted R2 : mean={sig3(s['weighted_mean'])}  std={sig3(s['weighted_std'])}")
        print("  per-PC R2 :")
        for i in range(s["n_pcs"]):
            print(f"      PC{i + 1:<2} mean={sig3(s['per_pc_mean'][i]):>8}  std={sig3(s['per_pc_std'][i]):>8}")


def write_csv(summaries, path):
    import csv
    max_pcs = max(s["n_pcs"] for s in summaries)
    fields = [
        "model", "in_channels", "out_channels", "n_pcs", "n_trials", "n_unique_seeds",
        "date_first", "date_last",
        "pc_weighted_r2_mean", "pc_weighted_r2_std",
    ]
    for i in range(max_pcs):
        fields += [f"pc{i + 1}_r2_mean", f"pc{i + 1}_r2_std"]
    fields += ["param_signature"]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            row = {
                "model": s["model"],
                "in_channels": s["in_channels"] if s["in_channels"] is not None else "?",
                "out_channels": s["out_channels"] if s["out_channels"] is not None else "?",
                "n_pcs": s["n_pcs"],
                "n_trials": s["n_trials"],
                "n_unique_seeds": s["n_unique_seeds"] if s["n_unique_seeds"] is not None else "unknown",
                "date_first": fmt_date(s["date_first"]),
                "date_last": fmt_date(s["date_last"]),
                "pc_weighted_r2_mean": sig3(s["weighted_mean"]),
                "pc_weighted_r2_std": sig3(s["weighted_std"]),
                "param_signature": param_signature_line(s["params"]),
            }
            for i in range(s["n_pcs"]):
                row[f"pc{i + 1}_r2_mean"] = sig3(s["per_pc_mean"][i])
                row[f"pc{i + 1}_r2_std"] = sig3(s["per_pc_std"][i])
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-trials", type=int, default=1,
                    help="only report batches with at least this many trials")
    ap.add_argument("--no-csv", action="store_true", help="skip writing the CSV file")
    args = ap.parse_args()

    params_by_run = load_params()
    batches = collect_batches(params_by_run)

    summaries = [summarise(b) for b in batches.values()
                 if len(b["trials"]) >= args.min_trials]
    # Newest batches first, then by model.
    summaries.sort(key=lambda s: (s["date_last"] or datetime.min, s["model"]), reverse=True)

    if not summaries:
        print("No batches found.")
        return

    print_report(summaries)

    if not args.no_csv:
        stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        out = os.path.join(RESULTS_DIR, f"batch_test_summary_{stamp}.csv")
        write_csv(summaries, out)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()