"""Analysis of the weight-gradient fixes (plan sec. 14).

Usage: analyze_wfix.py <worktree algorithm_comparison> <results/amvi dir> [<amvi repo root>]
Expects wfix/E3, wfix/R5w (pulled from RCI), r4/R9 (reproduction reference),
r3/R5 + r3/dumps/R5_* (current-GEMSS trajectory baseline).
"""
import glob, os, re, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, sys.argv[1])
from src.hard_data_factory import generate_overlapping_dataset
from src.evaluation import calculate_metrics

root = sys.argv[2]
ARMS = ["current", "frozen", "strat", "score", "ngvi", "fixedpoint", "ngvi_frozen"]
NAME = dict(current="F0 current", frozen="frozen", strat="F1 strat", score="F2 score", ngvi="F3 ngvi",
            fixedpoint="F4 fixedpoint", ngvi_frozen="F3 frozen-w")
# 2x2 decomposition (plan sec. 16): component optimiser {Adam, NGD} x weights {uniform, learned}
DECOMP = (("ngvi", "ngvi_frozen", "learned weights, under NGD"),
          ("ngvi_frozen", "frozen", "NGD vs Adam components, uniform weights"),
          ("ngvi", "current", "F3 vs F0 (both changes)"))
pd.set_option("display.width", 220)


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():+.3f} [{x.mean() - h:+.3f}, {x.mean() + h:+.3f}]"


# ============================================================== E3
fs = glob.glob(os.path.join(root, "wfix/E3/*.csv"))
e3 = pd.concat([pd.read_csv(f) for f in fs if os.path.getsize(f) > 0], ignore_index=True)
print("== E3: fits per arm (expected 750)")
print(e3.groupby("estimator").size().reindex(ARMS).to_string())
bad = e3[~np.isfinite(e3.elbo) | ~np.isfinite(e3.tv_soft)]
print(f"non-finite fits: {bad.groupby('estimator').size().to_dict()}")

old = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(root, "r4/R9/ss_*.csv"))
                 if re.search(r"/ss_(one|two)_", f)], ignore_index=True)
m = old.merge(e3[e3.estimator == "current"], on=["id", "K"], suffixes=("_o", "_n"))
print(f"reproduction of R9 ss by current: {len(m)} matched; identical elbo {np.mean(m.elbo_o == m.elbo_n):.3f}, "
      f"identical tv_soft {np.mean(np.isclose(m.tv_soft_o, m.tv_soft_n)):.3f}")

e3 = e3[np.isfinite(e3.elbo)]
b = e3.loc[e3.groupby(["estimator", "id"]).elbo.idxmax()]               # K by ELBO
cols = ["tv_soft", "tv_soft_unif", "tv_topD", "pip_mae_soft", "mse", "alpha_max", "perplexity", "elbo", "seconds"]
print("\n== E3, K by ELBO, mean over datasets, per condition")
for (d, rho), g in b.groupby(["design", "rho"]):
    print(f"-- {d} {rho}")
    print(g.groupby("estimator")[cols].mean().reindex(ARMS).round(3).to_string())
print("\n== E3 overall (250 datasets)")
print(b.groupby("estimator")[cols].mean().reindex(ARMS).round(3).to_string())
print("chosen K:", b.groupby("estimator").K.value_counts().unstack().reindex(ARMS).to_string())

piv = b.pivot_table(index="id", columns="estimator", values="tv_soft")
print("\n== E3 paired soft-TV differences (row minus column; negative = row better), 250 datasets")
for a in [x for x in ARMS[1:] if x in piv]:
    print(f"  {NAME[a]:14s} - F0: {ci(piv[a] - piv['current'])}")
fixes = ["strat", "score", "ngvi", "fixedpoint"]
best = min(fixes, key=lambda a: piv[a].mean())
print(f"  lowest mean among fixes: {NAME[best]}")
for a in fixes:
    if a != best:
        print(f"  {NAME[a]:14s} - {NAME[best]}: {ci(piv[a] - piv[best])}")
for (d, rho), g in b.groupby(["design", "rho"]):
    pv = g.pivot_table(index="id", columns="estimator", values="tv_soft")
    print(f"  {d} {rho}: " + "; ".join(f"{NAME[a]} - F0 {ci(pv[a] - pv['current'])}" for a in fixes))

print("\n== F3 ngvi minus the frozen control (F3 also replaces Adam on means/variances by NGD/iBLR)")
print(f"  overall: {ci(piv['ngvi'] - piv['frozen'])}")
for (d, rho), g in b.groupby(["design", "rho"]):
    pv = g.pivot_table(index="id", columns="estimator", values="tv_soft")
    print(f"  {d} {rho}: {ci(pv['ngvi'] - pv['frozen'])}")
if "ngvi_frozen" in piv:
    print("\n== 2x2 decomposition, E3 soft TV (row arm minus reference; negative = row arm better)")
    for a, r, lab in DECOMP:
        print(f"  {NAME[a]} - {NAME[r]} ({lab}): overall {ci(piv[a] - piv[r])}")
        for (d, rho), g in b.groupby(["design", "rho"]):
            pv = g.pivot_table(index="id", columns="estimator", values="tv_soft")
            print(f"      {d} {rho}: {ci(pv[a] - pv[r])}")
print("\n== learned vs uniform weights on the same components: tv_soft - tv_soft_unif (negative = weights help)")
for a in [x for x in ARMS if x in piv]:
    if a != "frozen":
        x = b[b.estimator == a]
        print(f"  {NAME[a]:14s} {ci(x.tv_soft - x.tv_soft_unif)}")

if len(sys.argv) > 3:                 # Li et al.'s saved MF / mixture results (their repo, read only)
    import json
    A = sys.argv[3]
    rows = [dict(id=r["id"], who=k, tv=r["methods"][m]["tv"])
            for r in json.load(open(os.path.join(A, "results/one_triplet/records.json"))) if r["p"] == 10
            for m, k in (("MFVI", "MF"), ("Mixture", "MIX"))]
    t2 = pd.read_csv(os.path.join(A, "results/two_triplets/records.csv"))
    t2 = t2[t2.method.isin(["baseline", "original"])]
    rows += [dict(id=x.dataset, who={"baseline": "MF", "original": "MIX"}[x.method], tv=x.group_union_tv)
             for _, x in t2.iterrows()]
    th = pd.DataFrame(rows).pivot_table(index="id", columns="who", values="tv")
    bb = b.merge(th, left_on="id", right_index=True)
    print(f"\n== soft TV minus Li et al.'s mixture (negative = GEMSS arm better), {bb.id.nunique()} datasets matched")
    for a in ("current", "frozen", "ngvi", "strat"):
        x = bb[bb.estimator == a]
        print(f"  {NAME[a]:14s} overall {ci(x.tv_soft - x.MIX)}, win {np.mean(x.tv_soft < x.MIX):.2f}")
        for (d, rho), g in x.groupby(["design", "rho"]):
            print(f"      {d} {rho}: {ci(g.tv_soft - g.MIX)}, win {np.mean(g.tv_soft < g.MIX):.2f}")

# ============================================================== RQ1 trajectory
truth, data = {}, {}
for s in range(1, 26):
    _, _, t, _ = generate_overlapping_dataset(n_samples=100, n_features=200, n_solutions=3, sparsity=10,
        latent_rank=2, overlap=4, noise_std=0.05, nan_ratio=0.0, binarize=False, nonlinear=False,
        noise_dist="gauss", seed=s)
    truth[s] = set(t)
cur = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(root, "r3/R5/*.csv"))], ignore_index=True)
cur["estimator"] = "current"
extra = []
for _, r in cur.iterrows():           # weight-aware columns for current from the R5 dumps
    z = np.load(os.path.join(root, f"r3/dumps/R5_{r.prior}/{r.prior}_r2_sd{r.seed}_K{r.K}.npz"))
    mu, al = z[f"mu_{r.step}"], z[f"alpha_{r.step}"]
    if r.step != 48000:               # the dumps hold the FINAL means under every mu_<step> key (view bug);
        extra.append((float(al.max()), np.nan, np.nan)); continue     # weights are stored correctly
    raw = [tuple(sorted(np.argsort(np.abs(m))[::-1][:10].tolist())) for m in mu]
    share = np.array([len(set(s) & truth[r.seed]) / 10 for s in raw])
    topw = np.argsort(-al)[:3]
    uw = calculate_metrics({f"w{i}": {"support": list(raw[k])} for i, k in enumerate(topw)}, truth[r.seed], 200)["F1_Score"]
    extra.append((float(al.max()), uw, float(al @ share)))
cur["alpha_max"], cur["union_f1_topw"], cur["signal_share_w"] = zip(*extra)
new = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(root, "wfix/R5w/*.csv")) if os.path.getsize(f) > 0],
                ignore_index=True)
tr = pd.concat([cur, new], ignore_index=True)
print("\n\n== RQ1 trajectory: rows per arm/prior (expected 875)")
print(tr.groupby(["prior", "estimator"]).size().unstack().reindex(columns=ARMS).to_string())
tr["perp_rel"] = tr.perplexity / tr.K


def paper_rule(g):           # K by best seed-mean union-F1 at this step
    kb = g.groupby("K").union_f1.mean().idxmax()
    return g[g.K == kb]


for prior in ("sss", "ss"):
    g = tr[tr.prior == prior]
    print(f"\n== RQ1 {prior}: union-F1, K by paper rule (best seed-mean K per step)")
    t = g.groupby(["estimator", "step"])[["K", "union_f1"]].apply(lambda x: paper_rule(x).union_f1.mean()).unstack(0)
    print(t.reindex(columns=ARMS).round(2).to_string())
    for col, lab in (("union_f1_topw", "union-F1 of the 3 highest-weight components (K by its own paper rule)"),):
        t = g.groupby(["estimator", "step"])[["K", col]].apply(lambda x: x.groupby("K")[col].mean().max()).unstack(0)
        print(f"\n   {lab}"); print(t.reindex(columns=ARMS).round(2).to_string())
    for col in ("alpha_max", "perp_rel", "signal_share_topD", "signal_share_w", "elbo", "raw_dissim"):
        t = g[g.step.isin([6000, 24000, 48000])].pivot_table(index=["step"], columns="estimator", values=col)
        print(f"\n   {col} (mean over K, seeds)"); print(t.reindex(columns=ARMS).round(2).to_string())
    g6 = g[g.step == 6000]
    e = g6.loc[g6.groupby(["estimator", "seed"]).elbo.idxmax()]
    print("\n   6k steps, K by ELBO: union-F1 | union-F1 top-weight | K chosen")
    for a in ARMS:
        x = e[e.estimator == a]
        if len(x):
            print(f"   {NAME[a]:14s} {x.union_f1.mean():.2f} ± {stats.sem(x.union_f1) * 2.06:.2f} | "
                  f"{x.union_f1_topw.mean():.2f} | {x.K.value_counts().sort_index().to_dict()}")
    for step in (6000, 48000):        # paired over seeds, K by ELBO within each run
        gs = g[g.step == step]
        e = gs.loc[gs.groupby(["estimator", "seed"]).elbo.idxmax()].pivot_table(index="seed", columns="estimator",
                                                                                values="union_f1")
        pairs = [("ngvi", "current"), ("ngvi", "frozen"), ("frozen", "current")]
        if "ngvi_frozen" in e:
            pairs += [(a, r) for a, r, _ in DECOMP[:2]]
        print(f"   {step} steps, K by ELBO, paired union-F1 over 25 seeds: " + "; ".join(
            f"{NAME[a]} - {NAME[r]} {ci(e[a] - e[r])}" for a, r in pairs))
    if "ngvi_frozen" in set(g.estimator):   # hard readout at fixed K: seed mean over K, paired over seeds
        for step in (6000, 48000):
            e = g[g.step == step].pivot_table(index="seed", columns="estimator", values="union_f1", aggfunc="mean")
            print(f"   {step} steps, fixed K (mean over K per seed), paired union-F1: " + "; ".join(
                f"{NAME[a]} - {NAME[r]} {ci(e[a] - e[r])}" for a, r in (("ngvi_frozen", "frozen"),
                    ("ngvi_frozen", "current"), ("ngvi", "ngvi_frozen"), ("frozen", "current"))))
    print("   seconds per fit at 48k (median):", g[g.step == 48000].groupby("estimator").seconds.median().reindex(ARMS).round(0).to_dict())
