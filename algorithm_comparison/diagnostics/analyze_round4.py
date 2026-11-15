"""Round 4 (plan sec. 11) analysis: R8 (likelihood sharpness on RQ1) and R9 (E3 arms).

Usage: analyze_round4.py <results/amvi dir>
Expects r4/R8, r4/R9, r4/dumps (pulled from RCI) next to r2/, r3/.
"""
import glob, os, re, sys
import numpy as np, pandas as pd
from scipy import stats

root = sys.argv[1]
D, N = 10, 100


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return f"{x.mean():.2f}" if len(x) else "–"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():.2f} ± {h:.2f}"


def kint(s):
    return int(re.search(r"\d+", s).group())


def n_distinct(dump_dir, method, seed, K):
    f = os.path.join(dump_dir, f"{method}_r2_sd{seed}_K{K}.npz")
    if not os.path.exists(f):
        return np.nan, np.nan
    mu = np.load(f)["mu"]
    sups = [tuple(sorted(np.argsort(-np.abs(m))[:D])) for m in mu]
    sets = [set(s) for s in sups]
    dis = np.mean([1 - len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]) if len(sets) > 1 else 0
    return len(set(sups)), dis


# ------------------------------------------------------------------ R8
rows = []
for f in glob.glob(os.path.join(root, "r4/R8/*.csv")):
    name = re.sub(r"_s\d+\.csv$", "", os.path.basename(f))
    d = pd.read_csv(f); d["arm"] = name; d["dump"] = os.path.join(root, "r4/dumps", f"R8_{name}")
    rows.append(d)
ref = [("sss_default", glob.glob(os.path.join(root, "r2/R4/GEMSSjoint_r2_s*.csv")), os.path.join(root, "r2/dumps/R4_GEMSSjoint_r2")),
       ("ss01_default", glob.glob(os.path.join(root, "r2/R3/GEMSSjointSS_v0.01_s*.csv")), os.path.join(root, "r2/dumps/R3_GEMSSjointSS_v0.01"))]
for name, fs, dd in ref:
    for f in fs:
        d = pd.read_csv(f); d["arm"] = name; d["dump"] = dd; rows.append(d)
r8 = pd.concat(rows, ignore_index=True)
r8["K"] = r8.hp.map(kint)
r8["prior"] = r8.arm.str.split("_").str[0]
r8["setting"] = r8.arm.str.split("_", n=1).str[1]
if "lik_scale_raw" not in r8: r8["lik_scale_raw"] = 1.0
r8["lik_scale_raw"] = r8["lik_scale_raw"].fillna(1.0)
r8["elbo_corr"] = r8.elbo - N * np.log(r8.lik_scale_raw)
nd = [n_distinct(dd, m, s, k) for dd, m, s, k in zip(r8.dump, r8.method, r8.seed, r8.K)]
r8["n_distinct"] = [a for a, _ in nd]; r8["raw_dissim"] = [b for _, b in nd]

order = ["lik0.02", "lik0.05", "default", "lik0.3", "lik1.0",
         "yn0.1_def", "yn0.1_lik", "yn0.3_def", "yn0.3_lik", "yn1.0_def", "yn1.0_lik"]
print("== R8: paper rule (K by seed-mean union-F1) | K by ELBO per seed")
print("prior | setting | seeds | best K | union-F1 | dissim | distinct raw (of K) | raw dissim | union-F1, K by ELBO")
for prior in ["sss", "ss01"]:
    for st in order:
        g = r8[(r8.prior == prior) & (r8.setting == st)]
        if g.empty: continue
        kb = g.groupby("K").union_f1.mean().idxmax(); b = g[g.K == kb]
        e = g.loc[g.groupby("seed").elbo.idxmax()]
        print(f"{prior} | {st} | {g.seed.nunique()} | {kb} | {ci(b.union_f1)} | {b.dissim.mean():.2f} | "
              f"{b.n_distinct.mean():.1f} | {b.raw_dissim.mean():.2f} | {ci(e.union_f1)}")
    print()
print("distinct raw supports by K (no response noise):")
for prior in ["sss", "ss01"]:
    g = r8[(r8.prior == prior) & r8.setting.isin(order[:5])]
    print(prior); print(g.pivot_table(index="setting", columns="K", values="n_distinct").reindex(order[:5]).round(1).to_string())
    print(g.pivot_table(index="setting", columns="K", values="union_f1").reindex(order[:5]).round(2).to_string())

print("\n== R10 for sigma (no response noise): ELBO - n log c")
for prior in ["sss", "ss01"]:
    g = r8[(r8.prior == prior) & r8.setting.isin(order[:5])]
    seeds = set.intersection(*[set(x.seed) for _, x in g.groupby(["setting", "K"])])
    g = g[g.seed.isin(seeds)]
    a = g.loc[g.groupby("seed").elbo_corr.idxmax()]
    b = g.loc[g.groupby(["seed", "K"]).elbo_corr.idxmax()]
    kb = b.groupby("K").union_f1.mean().idxmax(); bk = b[b.K == kb]
    print(f"{prior}: (a) sigma+K by ELBO {ci(a.union_f1)}, picks {a.setting.value_counts().to_dict()}; "
          f"(b) sigma by ELBO, K by truth ({kb}) {ci(bk.union_f1)}, picks {bk.setting.value_counts().to_dict()}")
    print("   mean corrected ELBO:", g.groupby("setting").elbo_corr.mean().reindex(order[:5]).round(1).to_dict())

# ------------------------------------------------------------------ R9
MF = {("one", 0.7): 0.156, ("one", 0.9): 0.345, ("two", 0.7): 0.221, ("two", 0.9): 0.451, ("two", 0.99): 0.740}
MIX = {("one", 0.7): 0.035, ("one", 0.9): 0.183, ("two", 0.7): 0.084, ("two", 0.9): 0.315, ("two", 0.99): 0.646}
fs = glob.glob(os.path.join(root, "r4/R9/*.csv"))
r9 = pd.concat([pd.read_csv(f).assign(tag=re.sub(r"_(one|two)_[0-9.]+\.csv$", "", os.path.basename(f))) for f in fs],
               ignore_index=True)
print("\n== R9 reproduction of Round 3 (same arms, RCI vs RCI)")
for arm in ["sss", "ss", "ss01"]:
    old = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(root, f"r3/E3/{arm}_*.csv"))])
    new = r9[r9.tag == arm]
    m = old.merge(new, on=["id", "K"], suffixes=("_o", "_n"))
    print(f"  {arm}: {len(m)} matched fits; max |d elbo| {np.abs(m.elbo_o - m.elbo_n).max():.2e}, "
          f"max |d tv_topD| {np.abs(m.tv_topD_o - m.tv_topD_n).max():.2e}, "
          f"share identical tv_topD {np.mean(np.isclose(m.tv_topD_o, m.tv_topD_n)):.3f}")

b9 = r9.loc[r9.groupby(["tag", "design", "rho", "replicate"]).elbo.idxmax()]   # K by ELBO
cols = ["tv_topD", "tv_thr", "tv_soft", "tv_tail", "pip_mae_soft", "n_distinct_topD", "size_soft", "mse",
        "tpr_soft", "fdp_soft"]
agg = b9.groupby(["tag", "design", "rho"])[cols].mean()
tags = ["sss", "ss", "ss01", "ss_scalefixed", "ss_kernel", "ss_L0.25", "ss_L0.5", "ss_L2"]
conds = [("one", 0.7), ("one", 0.9), ("two", 0.7), ("two", 0.9), ("two", 0.99)]
print("\n== R9, K by ELBO, mean over 50 datasets")
for c in conds:
    print(f"\n-- {c[0]} triplet(s), rho {c[1]}: their MF TV {MF[c]:.3f}, mixture TV {MIX[c]:.3f}")
    t = agg.xs(c, level=["design", "rho"]).reindex([x for x in tags if x in agg.index.get_level_values(0)])
    print(t.round(3).to_string())

print("\n== readings")
ss = agg.xs("ss", level="tag")
for c in conds:
    s = ss.loc[c]
    print(f"  {c}: ss soft TV {s.tv_soft:.3f} vs MF {MF[c]:.3f} (at/below MF: {s.tv_soft <= MF[c]}), "
          f"vs hard {s.tv_topD:.3f} (within 0.05: {abs(s.tv_soft - s.tv_topD) <= 0.05})")
for tag in ["ss_scalefixed", "ss_kernel"]:
    if tag not in agg.index.get_level_values(0): continue
    a = agg.xs(tag, level="tag")
    ok_h = [(a.loc[c].n_distinct_topD >= 2 * ss.loc[c].n_distinct_topD) and (ss.loc[c].tv_topD - a.loc[c].tv_topD >= 0.1) for c in conds]
    ok_s = [(a.loc[c].n_distinct_topD >= 2 * ss.loc[c].n_distinct_topD) and (ss.loc[c].tv_soft - a.loc[c].tv_soft >= 0.1) for c in conds]
    print(f"  {tag}: criterion met (hard TV) in {sum(ok_h)}/5, (soft TV) in {sum(ok_s)}/5; "
          f"distinct ratio {[round(a.loc[c].n_distinct_topD / ss.loc[c].n_distinct_topD, 2) for c in conds]}")
for c in conds:
    v = {t: round(agg.loc[(t,) + c].n_distinct_topD, 2) for t in ["ss_L0.25", "ss_L0.5", "ss", "ss_L2"] if (t,) + c in agg.index}
    print(f"  tempering {c}: distinct supports {v}")
