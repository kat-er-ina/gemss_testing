"""Paired per-dataset comparison of R9 GEMSS arms (K by ELBO) with Li et al.'s saved MF and mixture.
Usage: paired_r9.py <results/amvi dir> <amvi repo root>"""
import glob, json, os, re, sys
import numpy as np, pandas as pd
from scipy import stats
root, A = sys.argv[1], sys.argv[2]
rec = json.load(open(os.path.join(A, "results/one_triplet/records.json")))
rows = []
for r in rec:
    if r["p"] != 10: continue
    for m, k in (("MFVI", "MF"), ("Mixture", "MIX")):
        x = r["methods"][m]; rows.append(dict(id=r["id"], who=k, tv=x["tv"], pip=x["pip"], mse=x["mse"]))
t2 = pd.read_csv(os.path.join(A, "results/two_triplets/records.csv"))
t2 = t2[t2.method.isin(["baseline", "original"])]
for _, x in t2.iterrows():
    rows.append(dict(id=x.dataset, who={"baseline": "MF", "original": "MIX"}[x.method],
                     tv=x.group_union_tv, pip=x.pip_mae, mse=x.prediction_mse))
theirs = pd.DataFrame(rows).pivot_table(index="id", columns="who", values=["tv", "pip", "mse"])
theirs.columns = [f"{a}_{b}" for a, b in theirs.columns]
fs = glob.glob(os.path.join(root, "r4/R9/*.csv"))
r9 = pd.concat([pd.read_csv(f).assign(tag=re.sub(r"_(one|two)_[0-9.]+\.csv$", "", os.path.basename(f))) for f in fs], ignore_index=True)
b = r9.loc[r9.groupby(["tag", "id"]).elbo.idxmax()].merge(theirs, left_on="id", right_index=True)

def ci(x):
    x = np.asarray(x, float); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():+.3f} [{x.mean()-h:+.3f}, {x.mean()+h:+.3f}]"

for tag, ro in [("ss", "soft"), ("ss", "topD"), ("ss_L2", "soft"), ("ss_L0.5", "soft")]:
    print(f"\n== GEMSS {tag}, {ro} readout, K by ELBO: paired difference GEMSS minus theirs (negative = GEMSS better)")
    g = b[b.tag == tag]
    for (d, rho), gg in g.groupby(["design", "rho"]):
        print(f"  {d} {rho} (n={len(gg)}): TV vs MIX {ci(gg[f'tv_{ro}'] - gg.tv_MIX)}, win {np.mean(gg[f'tv_{ro}'] < gg.tv_MIX):.2f} | "
              f"TV vs MF {ci(gg[f'tv_{ro}'] - gg.tv_MF)} | PIP MAE vs MIX {ci(gg[f'pip_mae_{ro}'] - gg.pip_MIX)} | "
              f"MSE vs MIX {ci(gg.mse - gg.mse_MIX)}")
