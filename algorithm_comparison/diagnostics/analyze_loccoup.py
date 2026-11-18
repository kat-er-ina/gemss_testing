"""Plan sec. 27: localised coupling. Does the joint fit keep its block advantage at high P when its components may
differ only on a kept set of coordinates (the block, or a top-m marginal-score screen), all others shared?

Usage: analyze_loccoup.py <dir with G's G_*.csv> <dir with L_*.csv> [<dir with the control's N_*.csv>]
Same datasets, seeds and starts as experiment G. Gap = TV_arm - TV_Ens(G) on the block (post-hoc weights;
paired; negative = joint better). At block size every localised arm equals F3J (G's row is used).
'*' = 95 % CI excludes 0; 'e' = CI inside +-0.02.
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
LOC = ["F3Jloc_or", "F3Jloc20", "F3Jloc50"]


def short(x, band=0.02):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return "n/a"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); m = x.mean()
    tag = "e" if (m - h > -band and m + h < band) else ("*" if (m - h > 0 or m + h < 0) else " ")
    return f"{m:+.3f}{tag}"


def load(pattern):
    fs = [f for f in glob.glob(pattern) if os.path.getsize(f) > 0]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def prep(d):
    d = d.copy()
    d["cell"] = d.design + "/rho" + d.rho.astype(str) + "/n" + d.n.astype(str)
    return d


g = prep(load(os.path.join(sys.argv[1], "G_*.csv")))
g = g[(g.design != "R") & ((g.P == g.g) | ((g.nuis == "diag") & (g.wmode == "fixed")))]
if "nuis_prior" in g:
    g = g[g.nuis_prior.fillna("ss") == "ss"]
L = prep(load(os.path.join(sys.argv[2], "L_*.csv")))
g = g[g.cell.isin(L.cell.unique())]
KEY = ["init", "cell", "P", "rep"]

print("== L0 reproducibility: F3J rerun vs G (same dataset, seed, start); max |block TV difference|")
j0 = L[L.arm == "F3J"].merge(g[g.arm == "F3J"], on=KEY, suffixes=("", "_G"))
print(f"   rows matched {len(j0)} of {int((L.arm == 'F3J').sum())}; max |dTV| = {np.abs(j0.tv_post - j0.tv_post_G).max():.2e}")

blk = g[(g.P == g.g) & (g.arm == "F3J")]
d = pd.concat([g[g.arm.isin(["F3J", "F3Ens"])], L[L.arm.isin(LOC)]] + [blk.assign(arm=a) for a in LOC], ignore_index=True)
d["Pk"] = np.where(d.P == d.g, "block", d.P.astype(str))
PK = ["block"] + [str(x) for x in sorted(L.P.unique())]
d = d[d.Pk.isin(PK)]
d["Pk"] = pd.Categorical(d.Pk, categories=PK, ordered=True)
ens = d[d.arm == "F3Ens"].set_index(KEY).tv_post
d["gap"] = d.tv_post.values - ens.reindex(pd.MultiIndex.from_frame(d[KEY])).values
J = d[d.arm != "F3Ens"]
ARMS = ["F3J"] + LOC

for init in ("default", "oracle"):
    print(f"\n== L1 [{init} start] gap arm - Ens on the block, by cell and P (F3J = full coupling, from G)")
    print(J[J.init == init].groupby(["cell", "arm", "Pk"], observed=True).gap.apply(short).unstack("Pk").to_string())

m = J.groupby(["init", "cell", "arm", "Pk"], observed=True).gap.mean().unstack("Pk")
ret = (m[PK[-1]] / m["block"]).unstack("arm")[ARMS]
if len(sys.argv) > 3:                                  # the Gaussian-nuisance control's joint fit (sec. 25)
    c = prep(load(os.path.join(sys.argv[3], "N_*.csv")))
    v = c.pivot_table(index=KEY, columns="arm", values="tv_post")
    cg = (v.F3J - v.F3Ens).groupby(["init", "cell", "P"]).mean().unstack("P")
    ret["control J"] = cg[cg.columns.max()] / m["block"].xs("F3J", level="arm")
print(f"\n   retained fraction of the block-size gap at P = {PK[-1]}:")
print(ret.round(2).to_string())

print("\n== L2 screen recall (localised arms): block features kept (of 10), correlated-group features kept (fraction),"
      " signal features (group + feature 3) kept (fraction)")
s = L[L.arm.isin(["F3Jloc20", "F3Jloc50"]) & (L.init == "default")].copy()
s["cols_frac"] = s.scr_cols / s.design.map({"M": 3, "U": 2})
print(s.groupby(["cell", "arm", "P"])[["scr_block", "cols_frac", "scr_rel"]].mean().round(2).unstack("P").to_string())

print(f"\n   gap at P = {PK[-1]} by screen recall (all signal features kept vs not), default start")
t = J[(J.arm.isin(["F3Jloc20", "F3Jloc50"])) & (J.Pk == PK[-1]) & (J.init == "default")].copy()
t["full_recall"] = np.where(t.scr_rel >= 1, "all kept", "missed some")
print(t.groupby(["cell", "arm", "full_recall"]).gap.agg(["mean", "count"]).round(3).unstack("full_recall").to_string())

print("\n== L3 localised joint fit's own block TV, paired against the full joint fit (G) at each P (negative = localised better)")
jj = J.pivot_table(index=KEY + ["Pk"], columns="arm", values="tv_post", observed=True)
for init in ("default", "oracle"):
    x = jj.xs(init, level="init")
    print(f"   [{init}]")
    print(pd.DataFrame({a: (x[a] - x.F3J).groupby(["cell", "Pk"], observed=True).apply(short) for a in LOC})
          .unstack("Pk").to_string())

print("\n== L4 structure (default start, mean over cells x reps): block-distinct components, Mahalanobis^2 on block vs nuisance")
print(J[J.init == "default"].groupby(["arm", "Pk"], observed=True)[["n_groups_block", "d2_block", "d2_nuis"]]
      .mean().round(2).unstack("Pk").to_string())

print("\n== L5 ELBO, localised arm - full joint fit (paired, nats; negative = full joint fit has the higher ELBO)")
ee = J.pivot_table(index=KEY + ["Pk"], columns="arm", values="elbo", observed=True)
x = ee.xs("default", level="init")
print(pd.DataFrame({a: (x[a] - x.F3J).groupby(["cell", "Pk"], observed=True).apply(short) for a in LOC})
      .unstack("Pk").to_string())

print("\n== L6 drift of each arm's own block TV, TV(P) - TV(block), paired (default start)")
b = J[J.Pk == "block"].set_index(["arm"] + KEY[:2] + ["rep"]).tv_post
J = J.assign(drift=J.tv_post.values - b.reindex(pd.MultiIndex.from_frame(J[["arm", "init", "cell", "rep"]])).values)
print(J[J.init == "default"].groupby(["cell", "arm", "Pk"], observed=True).drift.apply(short).unstack("Pk").to_string())
