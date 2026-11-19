"""Plan sec. 48 (first look, no new runs): are the solutions reproducible across data perturbations?

Input: rq2_honest.py results (protocol 'inside' rows store each outer fold's solutions, ranked by inner-CV score).
Each training fold holds 80 % of the samples; two training folds share 75 % of their samples.
For a solution s found in fold f and another fold g: match_g(s) = max Jaccard of s to the solutions found in g.
s is reproduced at threshold tau if match_g(s) >= tau in at least 3 of the 4 other folds.
Reported per method and dataset (means over folds and the 3 repetitions):
  nsol        solutions per fold
  rep80/rep50 reproduced solutions per fold at tau 0.8 (>= 9 of 10 features shared) / 0.5 (about >= 7 of 10)
  match       mean over solutions and other folds of match_g(s)
  rank1_rep50 share of folds whose inner-CV top-ranked solution is reproduced at tau 0.5
  within      mean pairwise Jaccard among a fold's own solutions (a method with near-identical solutions matches
              trivially; read rep against it)
Usage: fold_reproducibility.py <rq2_honest results dir>
"""
import glob, os, sys
import numpy as np, pandas as pd

pd.set_option("display.width", 250)


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


rows = []
for f in glob.glob(os.path.join(sys.argv[1], "*.csv")):
    d = pd.read_csv(f, dtype={"solutions_by_fold": str})
    for _, r in d[d.protocol == "inside"].iterrows():
        folds = [[set(int(float(j)) for j in s.split(",")) for s in fs.split(";") if s] for fs in r.solutions_by_fold.split("|")]
        for fi, sols in enumerate(folds):
            others = [g for gi, g in enumerate(folds) if gi != fi and g]
            for rank, s in enumerate(sols):
                m = [max(jac(s, t) for t in g) for g in others]
                rows.append(dict(dataset=r.dataset, method=r.method, rep=r.rep, fold=fi, rank=rank,
                                 match=np.mean(m), r80=sum(x >= 0.8 for x in m) >= 3, r50=sum(x >= 0.5 for x in m) >= 3))
            if len(sols) > 1:
                w = np.mean([jac(sols[a], sols[b]) for a in range(len(sols)) for b in range(a + 1, len(sols))])
            else:
                w = np.nan
            rows.append(dict(dataset=r.dataset, method=r.method, rep=r.rep, fold=fi, rank=-1, within=w))
x = pd.DataFrame(rows)
s, wn = x[x["rank"] >= 0], x[x["rank"] < 0]
per_fold = s.groupby(["dataset", "method", "rep", "fold"]).agg(nsol=("rank", "size"), rep80=("r80", "sum"),
                                                               rep50=("r50", "sum"), match=("match", "mean"))
out = per_fold.groupby(["dataset", "method"]).mean()
out["rank1_rep50"] = s[s["rank"] == 0].groupby(["dataset", "method"]).r50.mean()
out["within"] = wn.groupby(["dataset", "method"]).within.mean()
order = ["GEMSS_default", "GEMSS_outlier", "RandLasso_ensemble", "EnumLasso", "Masking_logistic", "ALFESE_mi", "BBSSL"]
for ds in ["diabetes", "preterm", "pcos"]:
    print(f"\n== {ds}")
    print(out.loc[ds].reindex([m for m in order if m in out.loc[ds].index]).round(2).to_string())
