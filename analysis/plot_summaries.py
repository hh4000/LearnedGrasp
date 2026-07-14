import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import seaborn as sns
import json, os
from pathlib import Path
from scipy.stats import t, ttest_ind, ttest_1samp, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statannotations.Annotator import Annotator
def p_to_stars(p, *thresholds):
    def p_to_stars_inner(p, thresholds=(1e-3, 1e-2, 5e-2)):
        """Map a p-value to a significance string."""
        for i, t in enumerate(thresholds):
            if p < t:
                return "*"*(len(thresholds)-i)
        return "ns"
    return p_to_stars_inner(p, thresholds)
DATA_PATH = Path("results")
def annotate_tukey(ax, df, value_col, group_col, order, alpha=0.05):
    """Run Tukey HSD on one feature and annotate significant pairs."""
    tukey = pairwise_tukeyhsd(df[value_col], df[group_col], alpha=alpha)
    tukey_df = pd.DataFrame(
        data=tukey.summary().data[1:],
        columns=tukey.summary().data[0],
    )
    sig = tukey_df[tukey_df["reject"]]
    pairs   = list(zip(sig["group1"], sig["group2"]))
    pvalues = sig["p-adj"].astype(float).tolist()

    if pairs:
        annotator = Annotator(
            ax, pairs, data=df, x=group_col, y=value_col, order=order,
        )
        annotator.configure(text_format="star", loc="inside", verbose=0)
        annotator.set_pvalues_and_annotate(pvalues)

    return tukey_df
def annotate_vs_zero(ax, df, value_col, group_col, order, popmean=0.0,
                     show_ns=False):
    """One-sample t-test of each group's values vs. popmean; annotate above box."""
    results = []
    for g in order:
        vals = df.loc[df[group_col] == g, value_col].to_numpy()
        t, p = ttest_1samp(vals, popmean=popmean, nan_policy="omit")
        results.append((g, t, p))

    # place annotations just above the data range of each group
    y_min, y_max = ax.get_ylim()
    pad = 0.03 * (y_max - y_min)

    for x_idx, (g, t, p) in enumerate(results):
        label = p_to_stars(p, *(1e-4, 1e-3, 1e-2, 5e-2))
        if label == "ns" and not show_ns:
            continue
        group_max = df.loc[df[group_col] == g, value_col].max()
        ax.text(
            x_idx, group_max + pad, label,
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color="C3",
        )

    return pd.DataFrame(results, columns=["group", "t", "p"])

data_full = {}
for f in DATA_PATH.glob("*.json"):
    if not f.stem.startswith("summary"):continue
    data = json.load(open(f))
    model = data.get("model")
    r2 = data.get("r2_per_joint")
    if len(r2) not in data_full:
        data_full[len(r2)] = {}
    if model not in data_full[len(r2)]:
        data_full[len(r2)][model] = {"data": []}
    data_full[len(r2)][model]["data"].append(r2)
for n in data_full:
    for m in data_full[n]:
        data_full[n][m]["data"] = np.array(data_full[n][m]["data"])
print(data_full)
p_lims = [0.05, 0.01, 0.001]

def get_t_stat(m1,m2,s1,s2,c1,c2):
    s_d = np.sqrt(s1**2/c1+s2**2/c2)
    return np.abs(m1-m2)/s_d
for n in data_full:
    for k,v in data_full[n].items():
        data: np.ndarray = v['data']
        m = data.mean(axis=0)
        s = data.std(axis=0, ddof=1)
        ci95=t(df=len(data)).cdf(0.975) * s/(len(data)**0.5)
        data_full[n][k].update(dict(mean=m, std=s, ci_width=ci95))
w_bar = 0.9
for n, df in data_full.items():
    names = sorted(df.keys())
    means = [df[k]['mean'] for k in names]
    cis   = [df[k]['ci_width'] for k in names]
    counts= [len(df[k]['data']) for k in names]
    x_ticks = [i%n*(w_bar/n) + i//n - (n//2)*w_bar/n for i in range(n*len(names))]
    data_arrs = [df[k]['data'] for k in names]
    f, p = f_oneway(*data_arrs, axis = 0)
    print(f, p)
    bar_h = np.array(means).flatten()
    bar_ci = np.array(cis).flatten()
    plt.figure()
    plt.xticks(ticks= range(len(names)), labels=names, rotation = 90)
    plt.bar(x_ticks, bar_h, width=w_bar/n, edgecolor='black', yerr=bar_ci)
    plt.tight_layout()
    plt.ylim(-0.1, 0.5)
    group = []
    for name,count in zip(names, counts):
        for _ in range(count):
            group.append(name)
    datastack = np.concatenate([df[n]['data'] for n in names], axis = 0)
    print(len(group))
    print(datastack.shape)
    df = pd.DataFrame(dict(
        group=group,
        **{f'v{i}': datastack[:,i] for i in range(datastack.shape[1])}
    ))
    y_lims = [
        min(df[[f'v{i}' for i in range(datastack.shape[1])]].values.min() - 0.1, -0.1),
        max(df[[f'v{i}' for i in range(datastack.shape[1])]].values.max() + 0.1, 0.1)
    ]
    print(y_lims)

    fig, axes = plt.subplots(ncols=n)
    if n == 1: axes = [axes]
    for m, ax in enumerate(axes):
        print(f"PC{m+1}: ")
        result = pairwise_tukeyhsd(datastack[:,m], group, alpha=0.01)
        print(result)
        tukey_df = pd.DataFrame(data = result.summary().data[1:], columns = result.summary().data[0])
        sig = tukey_df[tukey_df["reject"]]
        pairs = list(zip(sig["group1"], sig["group2"]))
        p_vals = sig['p-adj'].astype(float).tolist()

        sns.boxplot(data = df, x = "group", y = f"v{m}", ax = ax)
        ax.axhline(0, color='gray',linestyle='--', linewidth = 1, alpha = 0.7)
        annotate_vs_zero(ax, df, value_col=f"v{m}", group_col=f"group", order = names)
        annotate_tukey(ax,df, value_col=f'v{m}', group_col=f'group', order = names)
        ax.set_ylim(y_lims)

    plt.show()

# data_total = np.concatenate([data_full[name]['data'] for name in data_full])
# mean_total = data_total.mean(axis=0)
# std_total = data_total.std(axis=0, ddof=1)
# ci = t(df=len(data_total)-1).cdf(0.975) * std_total/len(data_total)**0.5
# print("\n\n"
#       f"SUMMARIZED MEAN: {mean_total:.4f} ± {ci:.6f}")
# plt.bar(names, means,yerr=cis, align='center', capsize=5)
# plt.xticks(names)
# plt.show()