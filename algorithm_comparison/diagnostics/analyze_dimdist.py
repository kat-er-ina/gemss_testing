"""Plan sec. 21 (experiment G): dimension or distance? Joint (F3J) vs ensemble (F3Ens) against the exact block
posterior, with the block embedded in P dimensions (orthogonal nuisance) and block geometries from connected
(A designs, mini-RQ1 with slab 2) to separated (mini-RQ1 with RQ1's slab 100).

Usage: analyze_dimdist.py <dir with G_*.csv>
Gap = TV_J - TV_Ens on the block (post-hoc weights; paired over replicates; negative = joint better).
'*' = 95 % CI excludes 0; 'equiv' = CI inside +-0.02.
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(sys.argv[1], "G_*.csv")) if os.path.getsize(f) > 0],
              ignore_index=True)
d["cell"] = np.where(d.design == "R", "R/slab" + d.slab.astype(int).astype(str) + "/ys" + d.yscale.astype(str)
                     + "/ov" + d.r_overlap.astype(str),
                     d.design + "/rho" + d.rho.astype(str) + "/n" + d.n.astype(str))
d["Pk"] = np.where(d.P == d.g, "block", d.P.astype(str))
PK = ["block"] + [str(x) for x in sorted(d.P[d.P > d.g].unique())]
d["Pk"] = pd.Categorical(d.Pk, categories=PK, ordered=True)
BLOCK = d.P == d.g                       # block-size rows (no embedding; run once, w fixed, tagged nuis = diag)
print("rows:", len(d), "| per nuis/wmode/init/arm:", d.groupby(["nuis", "wmode", "init", "arm"]).size().to_dict())


def short(x, band=0.02):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return "n/a"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); m = x.mean()
    tag = "e" if (m - h > -band and m + h < band) else ("*" if (m - h > 0 or m + h < 0) else " ")
    return f"{m:+.3f}{tag}"


def gaps(df, col="tv_post", init="default"):
    v = df[df.init == init].pivot_table(index=["cell", "Pk", "rep"], columns="arm", values=col, observed=True)
    if not {"F3J", "F3Ens"} <= set(v.columns):
        return pd.DataFrame(columns=["cell", "Pk", "rep", "gap"])
    return (v["F3J"] - v["F3Ens"]).rename("gap").reset_index()


def sel(nuis, wmode):
    """block rows plus the embedded rows of one nuisance type and prior-weight mode"""
    return d[BLOCK | ((d.nuis == nuis) & (d.wmode == wmode) & ~BLOCK)]


Porder = sorted(d.P.unique())
print("\n== G0 block geometry per cell (exact posterior; block only, w fixed): nearest-mode distance (posterior s.d.),"
      " top-pattern mass, patterns for 90 %")
b0 = d[BLOCK].drop_duplicates(["cell", "rep"])
print(b0.groupby("cell")[["ex_dnn", "ex_maxpat", "ex_n90", "ex_nmodes"]].median().round(2).to_string())

for nuis in ("diag", "coll"):
    for wmode in ("fixed", "scaled"):
        g = gaps(sel(nuis, wmode))
        if not len(g):
            continue
        print(f"\n== G1 [{nuis} nuisance, w {wmode}] gap TV_J - TV_Ens by cell and P (block = no embedding)")
        print(g.groupby(["cell", "Pk"], observed=True).gap.apply(short).unstack("Pk").to_string())

print("\n== G2 retained gap at the largest P vs block (diag, w fixed): mean gap(P_max) / mean gap(block)")
g = gaps(sel("diag", "fixed"))
m = g.groupby(["cell", "Pk"], observed=True).gap.mean().unstack("Pk")
print((m[m.columns[-1]] / m["block"]).round(2).to_string())

print("\n== G3 DISTANCE at block size: gap by nearest-mode distance bin (pooled over cells)")
g = gaps(d[BLOCK]).merge(b0[["cell", "rep", "ex_dnn"]], on=["cell", "rep"])
g["dbin"] = pd.cut(g.ex_dnn.fillna(0), [-1, 1.5, 3, 6, 12, 1e9], labels=["<1.5 (or 1 mode)", "1.5-3", "3-6", "6-12", ">12"])
print(g.groupby("dbin", observed=True).gap.agg([short, "size"]).to_string())
print("   same, per cell:"); print(g.groupby("cell").agg(dnn=("ex_dnn", "median"), gap=("gap", short)).to_string())

print("\n== G4 mechanism by arm and P (diag, w fixed, default start): leak, distinct comps (all / block coords),"
      " nuisance share of between-component distance, nuisance PIP and its error vs exact")
s = sel("diag", "fixed"); s = s[s.init == "default"].copy()
s["nuis_share"] = s.d2_nuis / (s.d2_block + s.d2_nuis)
cols = [c for c in ("leak", "n_groups", "n_groups_block", "nuis_share", "nuis_pip", "nuis_pip_mae", "ex_nuis_ent", "seconds")
        if c in s]
print(s.groupby(["arm", "Pk"], observed=True)[cols].median().round(3).to_string())
print("   per cell, joint fit only: distinct components on block coordinates")
print(s[s.arm == "F3J"].groupby(["cell", "Pk"], observed=True).n_groups_block.mean().unstack("Pk").round(2).to_string())

print("\n== G5 separated basins: oracle-started ensemble members still distinct at the end (block coords, K = 6)")
o = sel("diag", "fixed"); o = o[(o.init == "oracle") & (o.arm == "F3Ens")]
print(o.groupby(["cell", "Pk"], observed=True).n_groups_block.mean().unstack("Pk").round(2).to_string())
g = gaps(sel("diag", "fixed"), init="oracle")
print("   gap under equal oracle starts:"); print(g.groupby(["cell", "Pk"], observed=True).gap.apply(short).unstack("Pk").to_string())

print("\n== G6 prior scaling (diag): what w = w0 p / P does to the block posterior (top-pattern mass) and the gap")
e = d[d.nuis == "diag"].drop_duplicates(["cell", "rep", "Pk", "wmode"])
print(e.groupby(["cell", "wmode", "Pk"], observed=True).ex_maxpat.median().unstack(["wmode", "Pk"]).round(2).to_string())

print("\n== G7 PIP MAE gap on the block (diag, w fixed)")
g = gaps(sel("diag", "fixed"), col="pip_mae_post")
print(g.groupby(["cell", "Pk"], observed=True).gap.apply(short).unstack("Pk").to_string())
