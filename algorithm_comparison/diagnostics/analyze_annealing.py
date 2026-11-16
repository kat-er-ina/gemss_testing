"""Plan sec. 19 analysis: A (overlap map vs exact posterior) and F (annealing on RQ1 and E3).

Usage: analyze_annealing.py <worktree algorithm_comparison> <results/amvi dir> [<mechanism csv>]
Paired comparisons; 'equiv' = 95 % CI inside +-0.02 (TV / union-F1), '+'/'-' = CI excludes 0.
Sign convention: differences are row arm minus reference; for TV and errors NEGATIVE = row better.
"""
import glob, os, sys
from functools import lru_cache
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, sys.argv[1])
from src.hard_data_factory import generate_overlapping_dataset

root = sys.argv[2]
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)


def ci(x, band=0.02):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return "n/a"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    lo, hi = x.mean() - h, x.mean() + h
    tag = " equiv" if (band and lo > -band and hi < band) else (" +" if lo > 0 else (" -" if hi < 0 else " ?"))
    return f"{x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]{tag}"


def short(x):        # compact cell: mean and significance mark only
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return "n/a"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    m = x.mean(); mark = "*" if (m - h > 0 or m + h < 0) else " "
    return f"{m:+.3f}{mark}"


def read(pattern):
    fs = [f for f in glob.glob(pattern) if os.path.getsize(f) > 0]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


# ====================================================================== A
A = read(os.path.join(root, "ann", "A", "*.csv"))
if len(A):
    print("== A rows:", len(A), "| per arm/anneal:", A.groupby(["arm", "anneal"]).size().to_dict())
    A["key"] = A.design + "|" + A.rho.astype(str) + "|" + A.n.astype(str) + "|" + A.rep.astype(str)

    def per_rep(df, col, how="meanK"):
        """one value per (dataset, arm, anneal): mean over K, or the K with the highest ELBO."""
        if how == "meanK":
            return df.groupby(["design", "rho", "n", "rep", "arm", "anneal"])[col].mean().reset_index()
        b = df.loc[df.groupby(["design", "rho", "n", "rep", "arm", "anneal"]).elbo.idxmax()]
        return b[["design", "rho", "n", "rep", "arm", "anneal", col]]

    def gap_table(df, col, a, r, anneal="none", how="meanK", fn=short):
        v = per_rep(df[df.anneal == anneal], col, how)
        pv = v.pivot_table(index=["design", "rho", "n", "rep"], columns="arm", values=col)
        d = (pv[a] - pv[r]).reset_index().rename(columns={0: "d"})
        return d.groupby(["design", "rho", "n"])["d"].apply(fn).unstack("n")

    print("\n== A1 exact posterior: mean top-pattern mass | patterns for 90 % mass | |corr(b0,b1)|")
    ex = A.drop_duplicates(["design", "rho", "n", "rep"])
    print(ex.groupby(["design", "rho", "n"])[["ex_maxpat", "ex_n90"]].mean().unstack("n").round(2).to_string())
    print(ex.assign(ac=ex.ex_corr.abs()).groupby(["design", "rho", "n"]).ac.mean().unstack("n").round(2).to_string())

    print("\n== A2 TV (fit weights, mean over K) per arm, no annealing")
    v = per_rep(A[A.anneal == "none"], "tv_fit")
    print(v.groupby(["design", "rho", "n", "arm"]).tv_fit.mean().unstack(["arm"]).unstack("n").round(3).to_string())

    for col in ("tv_fit", "pip_mae_fit", "cov_err_fit"):
        for a, r in (("F3J", "F3Dec"), ("F3J", "F3Ens"), ("F0J", "F0Dec")):
            print(f"\n== A3 {col}: {a} - {r} (mean over K; negative = joint better; * = CI excludes 0), no annealing")
            print(gap_table(A, col, a, r).to_string())
    print("\n== A3b same, K by ELBO: tv_fit F3J - F3Dec")
    print(gap_table(A, "tv_fit", "F3J", "F3Dec", how="elbo").to_string())

    print("\n== A4 TILING: design U, datasets whose exact pattern posterior is near-unimodal (top pattern >= 0.8)")
    U = A[(A.design == "U") & (A.ex_maxpat >= 0.8) & (A.anneal == "none")].copy()
    U["logsd"] = np.abs(np.log(U.sd_ratio_fit))
    print("   datasets per (rho, n):", U.drop_duplicates(["rho", "n", "rep"]).groupby(["rho", "n"]).size().unstack("n").to_string())
    for col in ("corr_err_fit", "logsd", "tv_fit"):
        for a, r in (("F3J", "F3Dec"), ("F0J", "F0Dec")):
            v = U.groupby(["rho", "n", "rep", "arm"])[col].mean().unstack("arm")
            d = (v[a] - v[r]).groupby(["rho", "n"]).apply(short).unstack("n")
            print(f"\n   {col}: {a} - {r} (negative = joint better)"); print(d.to_string())

    print("\n== A5 MASS SHARING: does post-hoc merging bring the decoupled fits to the joint one? (TV, mean over K)")
    v = A[A.anneal == "none"].groupby(["design", "rho", "n", "rep", "arm"])[["tv_fit", "tv_post", "tv_merged"]].mean()
    v = v.unstack("arm")
    for dec in ("F3Dec", "F3Ens"):
        d1 = (v[("tv_merged", dec)] - v[("tv_fit", "F3J")]).groupby(["design", "rho", "n"]).apply(short).unstack("n")
        d2 = (v[("tv_merged", dec)] - v[("tv_post", dec)]).groupby(["design", "rho", "n"]).apply(short).unstack("n")
        print(f"\n   {dec} merged - F3J fit (remaining gap)"); print(d1.to_string())
        print(f"   {dec} merged - {dec} post (what merging buys)"); print(d2.to_string())

    print("\n== A6 ANNEALING on A: schedule - none, per arm (TV fit, mean over K; negative = annealing helps)")
    v = per_rep(A, "tv_fit").pivot_table(index=["design", "rho", "n", "rep", "arm"], columns="anneal", values="tv_fit")
    for s in [c for c in v.columns if c != "none"]:
        d = (v[s] - v["none"]).groupby(["design", "rho", "arm"]).apply(short).unstack("arm")
        print(f"\n   {s} - none (pooled over n)"); print(d.to_string())
    print("\n   joint-minus-decoupled gap under each schedule (F3J - F3Dec, TV fit, pooled over n)")
    g = per_rep(A, "tv_fit").pivot_table(index=["design", "rho", "n", "rep", "anneal"], columns="arm", values="tv_fit")
    g = (g["F3J"] - g["F3Dec"]).groupby(["design", "rho", "anneal"]).apply(short).unstack("anneal")
    print(g.to_string())

# ====================================================================== F on RQ1
F = read(os.path.join(root, "ann", "F", "*.csv"))
if len(F):
    base = read(os.path.join(root, "coup", "B1", "*.csv"))
    base = base[base.arm.isin(["F3J", "F3Dec", "F3Ind16", "F0J", "F0Dec"]) & base.K.isin([6, 12, 24])
                & base.overlap.isin([0, 4, 8])].assign(anneal="none", init="default")
    allF = pd.concat([F, base], ignore_index=True)

    @lru_cache(maxsize=None)
    def planted(seed, ov):
        *_, sup = generate_overlapping_dataset(n_samples=100, n_features=200, n_solutions=3, sparsity=10,
            latent_rank=2, overlap=ov, noise_std=0.05, nan_ratio=0.0, binarize=False, nonlinear=False,
            noise_dist="gauss", seed=seed)
        return [set(s) for s in sup.values()]

    def n_found(r):
        w0, T = (r.anneal.split("/") + ["0"])[:2] if r.anneal != "none" else ("none", "0")
        if r.anneal == "none" and r.init == "default":
            f = os.path.join(root, "coup", "dumps", f"B1_{r.arm}_{r.prior}_ov{r.overlap}", f"{r.prior}_r2_sd{r.seed}_K{r.K}.npz")
        else:
            f = os.path.join(root, "ann", "dumps", f"F_{r.arm}_{w0}_{T}_{r.init}_{r.prior}_ov{r.overlap}",
                             f"{r.prior}_r2_sd{r.seed}_K{r.K}.npz")
        if not os.path.exists(f):
            return np.nan
        mu = np.load(f)["mu_6000"]
        pl = planted(int(r.seed), int(r.overlap)); sh = set.intersection(*pl)
        sups = [set(np.argsort(-np.abs(m))[:10].tolist()) for m in mu]
        return sum(any(len(s & (q - sh)) >= len(q - sh) / 2 for s in sups) for q in pl)

    f6 = allF[allF.step == 6000].copy()
    f6["n_found"] = [n_found(r) for _, r in f6.iterrows()]
    print("\n\n== F rows at 6k per arm/anneal/init:"); print(f6.groupby(["arm", "init", "anneal"]).size().unstack("anneal").to_string())
    for prior in ("sss", "ss"):
        g = f6[f6.prior == prior]
        for col in ("union_f1", "n_distinct", "n_found"):
            print(f"\n== F {prior}: {col} at 6k, mean over seeds and K, by start / schedule / arm / overlap")
            print(g.groupby(["init", "anneal", "arm", "overlap"])[col].mean().unstack(["overlap"]).unstack("arm").round(3).to_string())
        for col in ("union_f1", "n_found"):
            print(f"\n== F {prior}: {col}, schedule minus no-anneal (same start), paired over seeds (mean over K)")
            v = g.groupby(["init", "anneal", "arm", "overlap", "seed"])[col].mean().unstack("anneal")
            for s in [c for c in v.columns if c != "none"]:
                print(f"   {s}:"); print((v[s] - v["none"]).groupby(["init", "arm", "overlap"]).apply(short).unstack("overlap").unstack("init").to_string())
            print(f"   joint-minus-decoupled / -ensemble gap by schedule (default start):")
            w = g[g.init == "default"].groupby(["anneal", "arm", "overlap", "seed"])[col].mean().unstack("arm")
            for a, r in (("F3J", "F3Dec"), ("F3J", "F3Ind16"), ("F0J", "F0Dec")):
                print(f"   {a} - {r}:"); print((w[a] - w[r]).groupby(["anneal", "overlap"]).apply(short).unstack("overlap").to_string())

# ====================================================================== F on E3
E = read(os.path.join(root, "ann", "E3F", "*.csv"))
if len(E):
    b0 = pd.concat([read(os.path.join(root, "wfix", "E3", "current_*.csv")).assign(estimator="f0_joint"),
                    read(os.path.join(root, "wfix", "E3", "ngvi_one_*.csv")), read(os.path.join(root, "wfix", "E3", "ngvi_two_*.csv")),
                    read(os.path.join(root, "coup", "E3", "ngvi_dec_*.csv"))], ignore_index=True).assign(anneal="none")
    e = pd.concat([E, b0], ignore_index=True); e = e[np.isfinite(e.elbo)]
    b = e.loc[e.groupby(["estimator", "anneal", "id"]).elbo.idxmax()]
    print("\n\n== E3 soft TV, K by ELBO (none for f0_joint = the published loop 'current', identical to 5e-6)")
    print(b.groupby(["estimator", "anneal"])[["tv_soft", "tv_soft_unif", "alpha_max"]].mean().round(3).unstack("anneal").to_string())
    pv = b.pivot_table(index="id", columns=["estimator", "anneal"], values="tv_soft")
    for est in ("ngvi", "ngvi_dec", "f0_joint"):
        for s in [c for c in pv[est].columns if c != "none"]:
            print(f"  {est}: {s} - none: {ci(pv[(est, s)] - pv[(est, 'none')])}")
    for s in sorted(set(b.anneal)):
        if ("ngvi", s) in pv and ("ngvi_dec", s) in pv:
            print(f"  ngvi - ngvi_dec under {s}: {ci(pv[('ngvi', s)] - pv[('ngvi_dec', s)])}")

# ====================================================================== mechanism (local probe)
if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
    m = pd.read_csv(sys.argv[3])
    print("\n\n== Mechanism: coupling share of the (tempered) gradient over time, mean over seeds, K, overlap")
    for init in ("default", "overlap"):
        t = m[m.init == init].groupby(["opt", "prior", "anneal", "step"])[["gshare", "leak_unif"]].mean().round(3)
        print(f"\n-- start {init}: gshare"); print(t.gshare.unstack(["opt", "prior", "anneal"]).to_string())
        print(f"-- start {init}: leak (uniform weights)"); print(t.leak_unif.unstack(["opt", "prior", "anneal"]).to_string())
