"""Plan sec. 24: Gaussian-nuisance control. Does the dimension erosion of sec. 22 come from the nuisance being
non-Gaussian (spike-slab marginals that Gaussian components cannot hold) or from dimension as such?

Usage: analyze_gaussnuis.py <dir with experiment G's G_*.csv> <dir with the control's N_*.csv>
Same datasets, seeds and starts in both; only the nuisance prior differs (ss = G, gauss = control).
Gap = TV_J - TV_Ens on the block (post-hoc weights; paired; negative = joint better).
'*' = 95 % CI excludes 0; 'e' = CI inside +-0.02.
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)


def short(x, band=0.02):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return "n/a"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); m = x.mean()
    tag = "e" if (m - h > -band and m + h < band) else ("*" if (m - h > 0 or m + h < 0) else " ")
    return f"{m:+.3f}{tag}"


def load(pattern):
    fs = [f for f in glob.glob(pattern) if os.path.getsize(f) > 0]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


g = load(os.path.join(sys.argv[1], "G_*.csv"))
g = g[(g.design != "R") & ((g.P == g.g) | ((g.nuis == "diag") & (g.wmode == "fixed")))].copy()
if "nuis_prior" not in g:
    g["nuis_prior"] = "ss"
g["nuis_prior"] = g.nuis_prior.fillna("ss")
c = load(os.path.join(sys.argv[2], "N_*.csv"))
blk = g[g.P == g.g].copy()
d = pd.concat([g[g.P > g.g], c, blk.assign(nuis_prior="gauss"), blk], ignore_index=True)   # block shared by both
d["cell"] = d.design + "/rho" + d.rho.astype(str) + "/n" + d.n.astype(str)
d["Pk"] = np.where(d.P == d.g, "block", d.P.astype(str))
PK = ["block"] + [str(x) for x in sorted(d.P[d.P > d.g].unique())]
d["Pk"] = pd.Categorical(d.Pk, categories=PK, ordered=True)
d = d[d.cell.isin(c.design.add("/rho").add(c.rho.astype(str)).add("/n").add(c.n.astype(str)).unique())]
print("rows:", len(d), "| per nuisance prior / init / arm:", d.groupby(["nuis_prior", "init", "arm"]).size().to_dict())


def gaps(df, col="tv_post"):
    v = df.pivot_table(index=["nuis_prior", "init", "cell", "Pk", "rep"], columns="arm", values=col, observed=True)
    return (v.F3J - v.F3Ens).rename("gap").reset_index()


for col, name in (("tv_post", "block TV"), ("elbo", "ELBO (nats; positive = joint better)")):
    G_ = gaps(d, col)
    for init in ("default", "oracle"):
        print(f"\n== N1 [{init} start] gap J - Ens in {name}, by nuisance prior, cell and P")
        print(G_[G_.init == init].groupby(["cell", "nuis_prior", "Pk"], observed=True).gap.apply(short)
              .unstack("Pk").to_string())
    if col == "tv_post":
        m = G_.groupby(["init", "cell", "nuis_prior", "Pk"], observed=True).gap.mean().unstack("Pk")
        print("\n   retained fraction of the block-size gap at P max:")
        print((m[m.columns[-1]] / m["block"]).unstack("nuis_prior").round(2).to_string())

print("\n== N2 joint fit's own block TV, paired control - G at each P (same dataset, seed, start)")
j = d[d.arm == "F3J"].pivot_table(index=["init", "cell", "Pk", "rep"], columns="nuis_prior", values="tv_post",
                                  observed=True)
print((j.gauss - j.ss).rename("d").reset_index().groupby(["init", "cell", "Pk"], observed=True).d.apply(short)
      .unstack("Pk").to_string())

print("\n== N3 where the joint fit's separation goes (default start, mean over cells x reps): Mahalanobis^2 on block"
      " vs nuisance coordinates, block-distinct components")
s = d[(d.arm == "F3J") & (d.init == "default")]
print(s.groupby(["nuis_prior", "Pk"], observed=True)[["d2_block", "d2_nuis", "n_groups", "n_groups_block"]]
      .mean().round(3).to_string())
print("   per cell, block-distinct components of the joint fit")
print(s.groupby(["cell", "nuis_prior", "Pk"], observed=True).n_groups_block.mean().unstack("Pk").round(2).to_string())

print("\n== N4 ensemble's block TV, control vs G (should not change: independent fits, factorised posterior)")
e = d[(d.arm == "F3Ens") & (d.init == "default")]
print(e.groupby(["cell", "nuis_prior", "Pk"], observed=True).tv_post.mean().unstack("Pk").round(3).to_string())
