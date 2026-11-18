"""Plan sec. 37: does the 1/L step rule change results where the fixed step was stable? (B, C: trajectory CSVs;
D: realdata CSVs.) Same datasets and seeds under both rules.

Usage: analyze_steprule.py traj <fixed dir> <invL dir> <glob> <cell cols>   |   analyze_steprule.py real <fixed dir> <invL dir>
Prints, at step 6000 (and 12000 for invL): per arm, paired invL - fixed in union-F1 and ELBO; the J - Ens union-F1 gap
under each rule; invL 12000 - 6000 (convergence). '*' = 95 % CI excludes 0; 'e' = CI inside +-0.02 (F1) / +-0.5 (ELBO).
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)


def short(x, band):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return "n/a"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); m = x.mean()
    return f"{m:+.3f}{'e' if (m - h > -band and m + h < band) else ('*' if (m - h > 0 or m + h < 0) else ' ')}"


def load(d, g):
    return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, g)) if os.path.getsize(f) > 0], ignore_index=True)


mode = sys.argv[1]
if mode == "traj":
    a, b = load(sys.argv[2], sys.argv[4]), load(sys.argv[3], sys.argv[4])
    cells = sys.argv[5].split(",")
    key = ["prior", "overlap", "K"] + cells + ["seed", "arm"]
    grp = ["prior"] + cells
    A, B6, B12 = a[a.step == 6000].set_index(key), b[b.step == 6000].set_index(key), b[b.step == 12000].set_index(key)
    for m, band in (("union_f1", 0.02), ("elbo", 0.5)):
        dd = (B6[m] - A[m].reindex(B6.index)).rename("d").reset_index()
        print(f"\n== invL - fixed at step 6000, {m}, per arm")
        print(dd.groupby(grp + ["arm"]).d.apply(lambda x: short(x, band)).unstack("arm").to_string())
    for name, X in (("fixed 6000", A), ("invL 6000", B6), ("invL 12000", B12)):
        v = X.union_f1.unstack("arm")
        print(f"\n== J - Ens union-F1, {name}")
        print((v.F3J - v.F3Ens).rename("d").reset_index().groupby(grp).d.apply(lambda x: short(x, 0.02)).unstack(cells[-1]).to_string())
    dd = (B12.union_f1 - B6.union_f1).rename("d").reset_index()
    print("\n== invL convergence: union-F1 at 12000 - 6000, per arm")
    print(dd.groupby(grp + ["arm"]).d.apply(lambda x: short(x, 0.02)).unstack("arm").to_string())
    print("\n== mean effective step (invL), per arm and prior")
    print(b[b.step == 6000].groupby(["arm", "prior"] + cells[-1:]).step_mean.mean().unstack(cells[-1]).to_string())
else:
    a, b = load(sys.argv[2], "D_*.csv"), load(sys.argv[3], "D_*.csv")
    M = ["nsol", "f1_best", "f1_mean", "f1_min", "dissim", "n_distinct_raw", "elbo"]
    for (ds, arm), g in b.groupby(["dataset", "arm"]):
        f = a[(a.dataset == ds) & (a.arm == arm)].set_index("seed"); g = g.set_index("seed")
        print(f"{ds:12s} {arm:6s} " + "  ".join(f"{m} {g[m].mean():.3f} (fixed {f[m].mean():.3f})" for m in M))
