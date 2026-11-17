"""Plan sec. 23 (experiment G-amvi): Li et al.'s mixture vs P on experiment G's connected cells, next to GEMSS.

Usage: analyze_amvi_dimdist.py <dir with GA_*.csv> [<dir with experiment G's G_*.csv>]
Each method is scored against the exact posterior of its OWN model (Li et al.: point-mass spike; GEMSS: Gaussian
spike s.d. 0.1), so absolute TVs are not comparable across methods; the dimension TREND is.
'*' = 95 % CI excludes 0; 'e' = CI inside +-0.02 (paired over replicates).
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
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


a = load(os.path.join(sys.argv[1], "GA_*.csv"))
a["cell"] = a.design + "/rho" + a.rho.astype(str) + "/n" + a.n.astype(str)
a["Pk"] = np.where(a.P == a.g, "block", a.P.astype(str))
PK = ["block"] + [str(x) for x in sorted(a.P[a.P > a.g].unique())]
a["Pk"] = pd.Categorical(a.Pk, categories=PK, ordered=True)
err = a.error.fillna("").astype(str).str.len() > 0 if "error" in a else pd.Series(False, index=a.index)
print("rows:", len(a), "| errors:", int(err.sum()))
a = a[~err]
if "arm" not in a:
    a["arm"] = "asis"
print("arms:", a.groupby("arm").size().to_dict())

print("\n== GA0 exact posterior under their prior (block): top-pattern mass, patterns for 90 %; nuisance entropy at P max")
print(a[a.Pk == "block"].groupby("cell")[["ex_maxpat", "ex_n90"]].median().round(2).to_string())
print(a[a.P == a.P.max()].groupby("cell")[["ex_nuis_ent", "ex_nuis_pip"]].median().round(3).to_string())

print("\n== GA0b their split's coordinate choice (from the mean-field start): block coordinates among the 20 it uses,"
      " largest nuisance inclusion probability")
s0 = a[a.arm == a.arm.iloc[0]]
print(s0.pivot_table(index="cell", columns="Pk", values=["split_block_sel", "max_nuis_a"], observed=True).round(2).to_string())

print("\n== GA1 block TV of their mixture per arm, and of their mean field (mean over replicates)")
print(a.pivot_table(index=["cell", "arm"], columns="Pk", values="tv_mix", observed=True).round(3).to_string())
print(s0.pivot_table(index="cell", columns="Pk", values="tv_mf", observed=True).round(3).to_string())

print("\n== GA2 mixture gain on the block: TV_mix - TV_mf (paired; negative = mixture better)")
a["gap"] = a.tv_mix - a.tv_mf
print(a.groupby(["arm", "cell", "Pk"], observed=True).gap.apply(short).unstack("Pk").to_string())
m = a.groupby(["arm", "cell", "Pk"], observed=True).gap.mean().unstack("Pk")
print("   retained at P max:"); print((m[m.columns[-1]] / m["block"]).round(2).to_string())

print("\n== GA3 drift of the mixture's own block TV with P: TV_mix(P) - TV_mix(block), paired by replicate")
b = a[a.Pk == "block"].set_index(["arm", "cell", "rep"]).tv_mix
a["drift"] = a.tv_mix.values - b.reindex(pd.MultiIndex.from_arrays([a.arm, a.cell, a.rep])).values
print(a.groupby(["arm", "cell", "Pk"], observed=True).drift.apply(short).unstack("Pk").to_string())

print("\n== GA4 mixture structure by P: components (returned / candidate), fraction stopped by the time limit,"
      " distinct inclusion patterns (block / all), nuisance share of between-component inclusion differences, gain (nats)")
a["tl"] = (a.status == "time_limit").astype(float)
a["nuis_share"] = a.a_diff_nuis / (a.a_diff_block + a.a_diff_nuis).replace(0, np.nan)
a["secs"] = a.mf_seconds + a.fit_seconds
print(a.groupby(["arm", "cell", "Pk"], observed=True)[["K", "K_cand", "tl", "n_groups_block", "n_groups_all", "nuis_share",
                                                 "gain_mf", "secs"]].mean().round(3).to_string())

print("\n== GA5 nuisance PIP error (mixture | mean field) vs closed form")
print(a[a.P > a.g].pivot_table(index=["cell", "arm"], columns="Pk", values=["nuis_pip_mae_mix", "nuis_pip_mae_mf"],
                               observed=True).round(4).to_string())

if len(sys.argv) > 2:
    d = load(os.path.join(sys.argv[2], "G_*.csv"))
    d = d[(d.design != "R") & (d.init == "default") & ((d.P == d.g) | ((d.nuis == "diag") & (d.wmode == "fixed")))]
    d["cell"] = d.design + "/rho" + d.rho.astype(str) + "/n" + d.n.astype(str)
    v = d.pivot_table(index=["cell", "P", "rep"], columns="arm", values="tv_post")
    gG = (v.F3J - v.F3Ens).groupby(["cell", "P"]).mean().unstack("P")
    jG = d[d.arm == "F3J"].groupby(["cell", "P"]).tv_post.mean().unstack("P")
    print("\n== GA6 side by side, block size -> P max (each against its own exact posterior)")
    cols = {"GEMSS J TV": jG.iloc[:, 0].round(3).astype(str) + " -> " + jG.iloc[:, -1].round(3).astype(str),
            "GEMSS J-Ens retained": (gG.iloc[:, -1] / gG.iloc[:, 0]).round(2)}
    for arm in sorted(a.arm.unique()):
        s1 = a[a.arm == arm]
        gA = s1.groupby(["cell", "P"]).gap.mean().unstack("P"); jA = s1.groupby(["cell", "P"]).tv_mix.mean().unstack("P")
        cols[f"Li {arm} TV"] = jA.iloc[:, 0].round(3).astype(str) + " -> " + jA.iloc[:, -1].round(3).astype(str)
        cols[f"Li {arm} mix-MF retained"] = (gA.iloc[:, -1] / gA.iloc[:, 0]).round(2)
    print(pd.DataFrame(cols).to_string())
