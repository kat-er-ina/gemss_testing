"""Plan secs. 29 (B: noise x n) and 30 (C: classification): joint fit (F3J) vs ensemble of independent fits (F3Ens)
on the RQ1 generator, paired over seeds (same dataset, same torch seed).

Usage: analyze_pairs.py <dir with trajectory CSVs> <glob, e.g. 'B_*.csv'> <cell columns, e.g. noise,n> [step]
Metrics: union_f1 (clustered to 3 solutions), sol_f1 (planted supports recovered), n_distinct (distinct raw top-D
supports among the K components), seconds. Difference = J - Ens; '*' = 95 % CI excludes 0; 'e' = CI inside the
band (+-0.02 for F1, +-0.2 for n_distinct).
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
BAND = dict(union_f1=0.02, sol_f1=0.02, n_distinct=0.2)


def short(x, band):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return "n/a"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); m = x.mean()
    tag = "e" if (m - h > -band and m + h < band) else ("*" if (m - h > 0 or m + h < 0) else " ")
    return f"{m:+.3f}{tag}"


fs = [f for f in glob.glob(os.path.join(sys.argv[1], sys.argv[2])) if os.path.getsize(f) > 0]
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
cells = sys.argv[3].split(",")
step = int(sys.argv[4]) if len(sys.argv) > 4 else int(d.step.max())
d = d[d.step == step]
print(f"files {len(fs)}, rows at step {step}: {len(d)} | arms {d.arm.value_counts().to_dict()}")
key = ["prior", "overlap", "K"] + cells + ["seed"]
for m, band in BAND.items():
    v = d.pivot_table(index=key, columns="arm", values=m)
    diff = (v.F3J - v.F3Ens).rename("d").reset_index()
    print(f"\n== {m}: J - Ens (paired over seeds), rows {cells[0]}, columns {cells[1:] or 'none'}")
    for (pr, ov, K), g in diff.groupby(["prior", "overlap", "K"]):
        t = g.groupby(cells).d.apply(lambda x: short(x, band))
        print(f"-- prior {pr}, overlap {ov}, K {K}")
        print((t.unstack(cells[1:]) if len(cells) > 1 else t).to_string())
print("\n== absolute union_f1 (mean over seeds), J | Ens")
a = d.pivot_table(index=["prior", "overlap", "K"] + cells, columns="arm", values="union_f1").round(3)
print(a.to_string())
print("\n== fit seconds (median), by arm")
print(d.groupby(["arm", "prior"] + cells[-1:]).seconds.median().round(1).unstack(cells[-1]).to_string())
