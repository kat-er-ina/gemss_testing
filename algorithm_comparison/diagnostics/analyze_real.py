"""Plan sec. 31 (experiment D): joint fit vs ensemble on real data, paired over torch seeds.

Usage: analyze_real.py <dir with D_*.csv>
Per dataset: mean of each arm and the paired J - Ens difference ('*' = 95 % CI excludes 0) for the number of
solutions after clustering, best / mean / worst end-model F1 over the solutions, dissimilarity of the solutions,
distinct raw supports among the 12 components, ELBO; plus how reproducible each arm's solutions are across seeds
(mean pairwise Jaccard of the union of selected features between seeds) and how much the two arms' unions overlap.
"""
import glob, itertools, os, sys
import numpy as np, pandas as pd
from scipy import stats

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(sys.argv[1], "D_*.csv")) if os.path.getsize(f) > 0])
M = ["nsol", "f1_best", "f1_mean", "f1_min", "dissim", "n_distinct_raw", "elbo", "perplexity", "fit_seconds"]


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return "n/a"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); m = x.mean()
    return f"{m:+.3f}{'*' if (m - h > 0 or m + h < 0) else ' '}"


for ds, g in d.groupby("dataset"):
    print(f"\n== {ds} (n {g.n.iloc[0]}, p {g.p.iloc[0]}), seeds per arm {g.groupby('arm').size().to_dict()}")
    mean = g.groupby("arm")[M].mean().T.round(3)
    v = {m: g.pivot(index="seed", columns="arm", values=m) for m in M}
    mean["J - Ens"] = [ci(v[m]["F3J"] - v[m]["F3Ens"]) if {"F3J", "F3Ens"} <= set(v[m]) else "n/a" for m in M]
    print(mean.to_string())
    U = {a: [set(int(j) for s in str(r).split(";") for j in s.split(",") if j != "") for r in gg.solutions]
         for a, gg in g.sort_values("seed").groupby("arm")}
    jac = lambda A, B: len(A & B) / max(1, len(A | B))
    for a, us in U.items():
        print(f"   {a}: union of selected features {np.mean([len(u) for u in us]):.1f}; between-seed Jaccard of the union "
              f"{np.mean([jac(x, y) for x, y in itertools.combinations(us, 2)]):.2f}")
    if len(U) == 2:
        print(f"   J vs Ens, same seed: Jaccard of the unions {np.mean([jac(x, y) for x, y in zip(U['F3J'], U['F3Ens'])]):.2f}")
