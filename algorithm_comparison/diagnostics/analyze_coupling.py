"""Stage B of the coupling test (special-issue plan sec. 18): joint vs decoupled vs independent.

Usage: analyze_coupling.py <worktree algorithm_comparison> <results/amvi dir>
Reads coup/{B1,B2,B3,B4,E3} + coup/dumps (pulled from RCI) and wfix/E3 (joint E3 arms).
Paired comparisons over seeds; 'equiv' marks a 95 % CI inside the pre-stated band
(+-0.02 union-F1, +-0.2 distinct supports, +-0.02 soft TV).
Dump-derived per-fit metrics on the raw top-D supports of all components:
  sol_jac  per-planted-solution matched Jaccard (Hungarian), as the old repulsion probe
  n_found  planted solutions hit by some component (>= half of that solution's private features)
  F1raw    union-F1 of the raw supports (the old probe's F1; no clustering)
"""
import glob, os, sys
from functools import lru_cache
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, sys.argv[1])
from src.hard_data_factory import generate_overlapping_dataset

root = sys.argv[2]
C = os.path.join(root, "coup")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
BAND = dict(union_f1=0.02, F1raw=0.02, sol_jac=0.02, n_distinct=0.2, n_found=0.2, tv_soft=0.02)


def ci(x, band=None):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return "n/a"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    lo, hi = x.mean() - h, x.mean() + h
    tag = ""
    if band is not None:
        tag = " equiv" if (lo > -band and hi < band) else (" +" if lo > 0 else (" -" if hi < 0 else " ?"))
    return f"{x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]{tag}"


def load(stage):
    fs = [f for f in glob.glob(os.path.join(C, stage, "*.csv")) if os.path.getsize(f) > 0]
    return pd.concat([pd.read_csv(f).assign(task=os.path.basename(f)[:-4]) for f in fs], ignore_index=True)


@lru_cache(maxsize=None)
def planted(seed, ov, D, p, task):
    *_, sup = generate_overlapping_dataset(
        n_samples=100, n_features=p, n_solutions=3, sparsity=D, latent_rank=2, overlap=ov, noise_std=0.05,
        nan_ratio=0.0, binarize=(task == "classification"), nonlinear=False, noise_dist="gauss", seed=seed)
    return [set(s) for s in sup.values()]


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def dump_metrics(df, dumpdir_of):
    out = []
    last = df.step.max()     # dumps written before 2026-11-16 hold the FINAL means under every mu_<step> key
    for _, r in df.iterrows():
        f = os.path.join(C, "dumps", dumpdir_of(r), f"{r.prior}_r2_sd{r.seed}_K{r.K}.npz")
        if r.step != last or not os.path.exists(f):
            out.append((np.nan, np.nan, np.nan)); continue
        mu = np.load(f)[f"mu_{r.step}"]
        sups = [set(np.argsort(-np.abs(m))[:r.D].tolist()) for m in mu]
        pl = planted(int(r.seed), int(r.overlap), int(r.D), int(r.p), r.task_kind)
        sim = np.array([[jac(a, b) for b in sups] for a in pl])
        ri, cj = linear_sum_assignment(-sim)
        sol_jac = float(np.mean([sim[i, j] for i, j in zip(ri, cj)]))
        shared = set.intersection(*pl)
        found = sum(any(len(s & (q - shared)) >= len(q - shared) / 2 for s in sups) for q in pl)
        truth, union = set().union(*pl), set().union(*sups)
        inter = len(truth & union); rec, prec = inter / len(truth), inter / len(union)
        out.append((sol_jac, found, 2 * rec * prec / (rec + prec) if rec + prec else 0.0))
    df = df.copy()
    df["sol_jac"], df["n_found"], df["F1raw"] = zip(*out)
    return df


def fixed_k(g, metric):            # seed mean over K
    return g.groupby(["arm", "seed"])[metric].mean().unstack(0)


def k_by_elbo(g, metric):
    b = g.loc[g.groupby(["arm", "seed"]).elbo.idxmax()]
    return b.pivot_table(index="seed", columns="arm", values=metric)


def paper_rule(g, metric):          # each arm at its K with the best seed-mean union-F1
    cols = {}
    for a, x in g.groupby("arm"):
        kb = x.groupby("K").union_f1.mean().idxmax()
        cols[a] = x[x.K == kb].set_index("seed")[metric]
    return pd.DataFrame(cols)


def pair_table(g, pairs, metric, how):
    piv = how(g, metric)
    return {f"{a} - {b}": ci(piv[a] - piv[b], BAND.get(metric)) for a, b in pairs if a in piv and b in piv}


# ====================================================================== B1
PAIRS = [("F0J", "F0Dec"), ("F0J", "F0IndK"), ("F0J", "F0Ind16"), ("F0Dec", "F0IndK"),
         ("F3J", "F3Dec"), ("F3J", "F3Ind16"), ("F3J", "F0J")]
b1 = load("B1"); b1["task_kind"] = "regression"
b1 = dump_metrics(b1, lambda r: f"B1_{r.arm}_{r.prior}_ov{r.overlap}")
print("== B1 rows per arm (expected 2 priors x 4 overlaps x 25 seeds x 4 K x 2 steps = 1600)")
print(b1.groupby("arm").size().to_string())
for step in (6000, 2000):
    for prior in ("sss", "ss"):
        g0 = b1[(b1.step == step) & (b1.prior == prior)]
        print(f"\n== B1 {prior}, {step} steps: means (fixed K, mean over K and seeds)")
        print(g0.groupby(["arm", "overlap"])[["union_f1", "n_distinct", "n_found", "sol_jac", "dissim"]]
              .mean().unstack("overlap").round(3).to_string())
        for metric, how, lab in (("union_f1", fixed_k, "fixed K"), ("union_f1", k_by_elbo, "K by ELBO"),
                                 ("union_f1", paper_rule, "paper rule"), ("n_distinct", fixed_k, "fixed K"),
                                 ("n_found", fixed_k, "fixed K"), ("sol_jac", fixed_k, "fixed K")):
            rows = {ov: pair_table(g0[g0.overlap == ov], PAIRS, metric, how) for ov in (0, 2, 4, 8)}
            print(f"\n   {metric}, {lab}: paired differences over 25 seeds")
            print(pd.DataFrame(rows).to_string())

# ====================================================================== B2
b2 = load("B2"); b2["task_kind"] = "classification"
b2 = dump_metrics(b2, lambda r: f"B2_{r.arm}_p{r.p}_ov{r.overlap}")
print("\n\n== B2 (old repulsion probe: classification, D 5, K 6, 4k steps, 10 seeds); probe metrics on raw supports")
print(b2.groupby(["arm", "p", "overlap"])[["F1raw", "sol_jac", "n_distinct"]].mean().unstack(["p", "overlap"]).round(3).to_string())
P2 = [("probeInd", "probeJ"), ("F0J", "probeJ"), ("F0Ind16", "F0J"), ("F0IndK", "F0J"), ("F0Dec", "F0J"),
      ("probeInd", "F0Ind16")]
for metric in ("n_distinct", "F1raw", "sol_jac"):
    rows = {}
    for (p, ov), g in b2.groupby(["p", "overlap"]):
        piv = g.pivot_table(index="seed", columns="arm", values=metric)
        rows[f"p{p} ov{ov}"] = {f"{a} - {b}": ci(piv[a] - piv[b], BAND.get(metric)) for a, b in P2}
    print(f"\n   {metric}: paired over 10 seeds"); print(pd.DataFrame(rows).to_string())

# ====================================================================== B3
b3 = load("B3"); b3["task_kind"] = "regression"
b3 = dump_metrics(b3, lambda r: f"B3_{r.init}_{r.arm}_{r.prior}_ov{r.overlap}")
ref = b1[b1.K.isin([6, 12, 24]) & b1.overlap.isin([4, 8]) & b1.arm.isin(["F0J", "F0Dec", "F0IndK", "F3J", "F3Dec"])]
print("\n\n== B3 (common / overlapping starts; K 6/12/24; overlaps 4 and 8), default start from B1 for reference")
allb = pd.concat([b3, ref.assign(init="default")], ignore_index=True)
for step in (6000, 2000):
    for prior in ("sss", "ss"):
        g0 = allb[(allb.step == step) & (allb.prior == prior)]
        print(f"\n-- {prior}, {step} steps: means (fixed K)")
        print(g0.groupby(["init", "arm", "overlap"])[["union_f1", "n_distinct", "n_found"]].mean()
              .unstack("overlap").round(3).to_string())
        for metric in ("union_f1", "n_distinct", "n_found"):
            rows = {}
            for (init, ov), g in g0.groupby(["init", "overlap"]):
                rows[f"{init} ov{ov}"] = pair_table(g, [("F0J", "F0Dec"), ("F0J", "F0IndK"), ("F3J", "F3Dec")],
                                                    metric, fixed_k)
            print(f"\n   {metric}, fixed K: paired over 25 seeds"); print(pd.DataFrame(rows).to_string())

# ====================================================================== B4
b4 = load("B4"); b4["task_kind"] = "regression"
b4 = dump_metrics(b4, lambda r: f"B4_{r.arm.replace('_R', '')}_{r.init}_{r.prior}_ov{r.overlap}")
jdef = b1[b1.arm.isin(["F0J", "F3J"]) & b1.overlap.isin([0, 4, 8]) & b1.K.isin([6, 12, 24])].assign(init="default")
ind_def = b1[b1.arm.isin(["F0Ind16", "F3Ind16"]) & b1.overlap.isin([0, 4, 8])].assign(init="default")
ind_def["arm"] = ind_def.arm + "_R"
allb4 = pd.concat([b4, jdef, ind_def], ignore_index=True)
print("\n\n== B4: multi-start ensembles of the same model (Ind16 with R fits) vs joint (K), by start; 6k steps")
print("   reference (paper RQ1 table, overlap 0/2/4/8): RLE .50/.51/.54/.53, BB-SSL .51/.53/.56/.61, "
      "GEMSS (truth-tuned) .84/.89/.91/.90")
for prior in ("sss", "ss"):
    g = allb4[(allb4.step == 6000) & (allb4.prior == prior)]
    print(f"\n-- {prior}: union-F1 / n_found by arm, start, K or R (mean over 25 seeds)")
    print(g.pivot_table(index=["arm", "init", "K"], columns="overlap", values="union_f1").round(3).to_string())
    print(g.pivot_table(index=["arm", "init", "K"], columns="overlap", values="n_found").round(2).to_string())
    for ov in (0, 4, 8):
        x = g[g.overlap == ov]
        best = {(a, i): x[(x.arm == a) & (x.init == i)].groupby("K").union_f1.mean().max()
                for a, i in x.groupby(["arm", "init"]).size().index}
        print(f"   ov{ov} best-K/R (paper rule) union-F1: " + "; ".join(f"{a}/{i} {v:.3f}" for (a, i), v in sorted(best.items())))

# ====================================================================== E3
e3 = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(C, "E3", "*.csv")) if os.path.getsize(f) > 0] +
               [pd.read_csv(f) for f in glob.glob(os.path.join(root, "wfix", "E3", "*.csv"))
                if os.path.basename(f).split("_")[0] in ("current", "ngvi") and not os.path.basename(f).startswith("ngvi_frozen")],
               ignore_index=True)
e3 = e3[np.isfinite(e3.elbo)]
b = e3.loc[e3.groupby(["estimator", "id"]).elbo.idxmax()]
print("\n\n== E3 soft TV, K by ELBO (250 datasets): joint vs decoupled")
print(b.groupby("estimator")[["tv_soft", "tv_soft_unif", "alpha_max", "elbo"]].mean().round(3).to_string())
piv = b.pivot_table(index="id", columns="estimator", values="tv_soft")
for a, r in (("current", "f0_dec"), ("ngvi", "ngvi_dec")):
    if a in piv and r in piv:
        print(f"  {a} - {r}: overall {ci(piv[a] - piv[r], 0.02)}")
        for (d, rho), g in b.groupby(["design", "rho"]):
            pv = g.pivot_table(index="id", columns="estimator", values="tv_soft")
            print(f"      {d} {rho}: {ci(pv[a] - pv[r], 0.02)}")

# ====================================================================== weight per planted solution at 6k
# Replaces the invalid 6k rows of wfix_alt_mass (plan sec. 15): B1 dumps hold valid means AND weights at
# their final checkpoint (6000). Components assigned by >= half of a solution's private features.
print("\n\n== Weight on best / 2nd / 3rd planted solution and on none, 6k steps (B1 dumps, final checkpoint)")
rows = []
for _, r in b1[(b1.step == 6000) & b1.arm.isin(["F0J", "F3J", "F3Dec"]) & b1.K.isin([6, 12, 24])].iterrows():
    z = np.load(os.path.join(C, "dumps", f"B1_{r.arm}_{r.prior}_ov{r.overlap}", f"{r.prior}_r2_sd{r.seed}_K{r.K}.npz"))
    mu, al = z["mu_6000"], z["alpha_6000"]
    pl = planted(int(r.seed), int(r.overlap), 10, 200, "regression"); sh = set.intersection(*pl)
    priv = [q - sh for q in pl]; w = np.zeros(4)
    for k, m in enumerate(mu):
        top = set(np.argsort(-np.abs(m))[:10].tolist()); ov = [len(top & q) for q in priv]
        j = int(np.argmax(ov)); w[j if ov[j] >= len(priv[j]) / 2 else 3] += al[k]
    ws = np.sort(w[:3])[::-1]
    rows.append(dict(arm=r.arm, prior=r.prior, overlap=r.overlap, K=r.K, w1=ws[0], w2=ws[1], w3=ws[2], none=w[3]))
wt = pd.DataFrame(rows)
print(wt.groupby(["prior", "overlap", "arm", "K"])[["w1", "w2", "w3", "none"]].mean().round(2)
      .loc[(slice(None), [4, 8]), :].to_string())
