"""Plan sec. 28 (experiment S): spike width. Target spike s.d. 0.01, spike fixed ('none') vs a path 0.1 -> 0.01.

Usage: analyze_spike.py <dir with G's G_*.csv (spike 0.1)> <dir with S_*.csv>
Each fit is scored against the exact posterior at its own target spike. Gap = TV_J - TV_Ens on the block (post-hoc
weights; paired; negative = joint better). '*' = 95 % CI excludes 0; 'e' = CI inside +-0.02.
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


s = load(os.path.join(sys.argv[2], "S_*.csv"))
g = load(os.path.join(sys.argv[1], "G_*.csv"))
g = g[(g.design != "R") & ((g.P == g.g) | ((g.nuis == "diag") & (g.wmode == "fixed")))]
if "nuis_prior" in g:
    g = g[g.nuis_prior.fillna("ss") == "ss"]
g = g[g.P.isin(s.P.unique()) & g.arm.isin(["F3J", "F3Ens"])].assign(spike=0.1, spike_path="G (spike 0.1)")
d = pd.concat([s, g], ignore_index=True)
d["cell"] = d.design + "/rho" + d.rho.astype(str) + "/n" + d.n.astype(str)
d = d[d.cell.isin(s.design.add("/rho").add(s.rho.astype(str)).add("/n").add(s.n.astype(str)).unique())]
d["Pk"] = np.where(d.P == d.g, "block", d.P.astype(str))
PK = ["block"] + [str(x) for x in sorted(s.P[s.P > s.g].unique())]
d["Pk"] = pd.Categorical(d.Pk, categories=PK, ordered=True)
print("rows:", d.groupby(["spike_path", "init", "arm"]).size().to_dict())
key = ["spike_path", "init", "cell", "Pk", "rep"]
v = d.pivot_table(index=key, columns="arm", values="tv_post", observed=True)
gap = (v.F3J - v.F3Ens).rename("gap").reset_index()

for init in ("default", "oracle"):
    print(f"\n== S1 [{init} start] gap J - Ens on the block, by schedule, cell and P")
    print(gap[gap.init == init].groupby(["cell", "spike_path", "Pk"], observed=True).gap.apply(short).unstack("Pk").to_string())
m = gap.groupby(["init", "cell", "spike_path", "Pk"], observed=True).gap.mean().unstack("Pk")
print(f"\n   retained fraction of the block-size gap at P = {PK[-1]} (G = spike 0.1, sec. 22):")
print((m[PK[-1]] / m["block"]).unstack("spike_path").round(2).to_string())

print("\n== S2 absolute block TV per arm (mean over replicates)")
print(d.pivot_table(index=["init", "cell", "arm"], columns=["spike_path", "Pk"], values="tv_post", observed=True)
      .round(3).to_string())

print("\n== S3 path - fixed, paired, each arm's block TV (negative = path better)")
w = d[d.spike_path != "G (spike 0.1)"].pivot_table(index=["init", "arm", "cell", "Pk", "rep"], columns="spike_path",
                                                   values="tv_post", observed=True)
print((w["0.1/4000"] - w["none"]).rename("d").reset_index().groupby(["init", "arm", "cell", "Pk"], observed=True)
      .d.apply(short).unstack("Pk").to_string())

print("\n== S4 path - fixed, paired ELBO (nats; positive = path better)")
w = d[d.spike_path != "G (spike 0.1)"].pivot_table(index=["init", "arm", "cell", "Pk", "rep"], columns="spike_path",
                                                   values="elbo", observed=True)
print((w["0.1/4000"] - w["none"]).rename("d").reset_index().groupby(["init", "arm", "Pk"], observed=True)
      .d.apply(lambda x: short(x, 0.5)).unstack("Pk").to_string())

print("\n== S5 structure (mean over cells x reps): block-distinct components, Mahalanobis^2 on block / nuisance")
print(d.groupby(["init", "arm", "spike_path", "Pk"], observed=True)[["n_groups_block", "d2_block", "d2_nuis"]]
      .mean().round(2).unstack("Pk").to_string())
