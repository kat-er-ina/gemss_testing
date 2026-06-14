"""Devil's advocate: can a CHEAP selector run as a compute-matched random-restart
ensemble find GEMSS's overlapping solutions just as well?

GEMSS is ~100-600x slower per run than L1-Lasso. So give Lasso the SAME wall-clock
budget: fit L1-logistic on B bootstrap subsamples (B chosen so the ensemble's time
~= one GEMSS run), each yielding a top-D support. Two honest readouts of "what the
restarts found":
  - union recovery of ALL collected features (stability-selection style);
  - the top-m most-FREQUENT distinct supports as the recovered solution set,
    scored against the planted supports (sol_f1, overlap_struct_err) -- the same
    metrics GEMSS is judged by.

If the ensemble matches GEMSS on recovery AND overlap-structure within the same
compute, the contribution is in trouble (report it). If it recovers the union but
NOT the calibrated overlapping solution set, GEMSS's joint optimisation is the
value.
"""

import argparse
import sys
import os
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.hard_data_factory import generate_overlapping_dataset
from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper
from src.evaluation import calculate_metrics, calculate_structural_metrics

GEMSS_HP = dict(prior="sss", var_slab=32.0, var_spike=0.1, lr=0.01, n_iter=6000,
                batch_size=16, weight_slab=0.9, weight_spike=0.1)


def lasso_support(Xs, y, D, alpha, seed):
    m = LogisticRegression(penalty="l1", solver="liblinear", C=1.0 / max(alpha, 1e-6),
                           max_iter=500, random_state=seed)
    m.fit(Xs, y)
    coef = np.abs(np.asarray(m.coef_).ravel())
    return tuple(sorted(int(i) for i in np.argsort(coef)[::-1][:D]))


def bootstrap_ensemble(X, y, D, m, budget_sec, alpha=0.05, seed=0, randomized=False, wmin=0.2):
    """Fit L1-logistic on bootstrap subsamples until budget_sec is spent; return
    the top-m most-frequent distinct supports + all-feature union + #distinct + B.
    randomized=True -> Meinshausen-Buhlmann randomised lasso (per-restart random
    feature reweighting), the stronger restart-oriented cheap competitor."""
    rng = np.random.default_rng(seed)
    Xs = np.nan_to_num(StandardScaler().fit_transform(np.nan_to_num(X)))
    n, p = Xs.shape
    supports = []
    t0 = time.time()
    b = 0
    while time.time() - t0 < budget_sec:
        idx = rng.choice(n, size=n, replace=True)  # bootstrap
        Xb = Xs[idx]
        if randomized:
            Xb = Xb * rng.uniform(wmin, 1.0, size=p)  # random penalty reweighting
        try:
            supports.append(lasso_support(Xb, y[idx], D, alpha, seed + b))
        except Exception:
            pass
        b += 1
    freq = Counter(supports)
    top = [set(s) for s, _ in freq.most_common(m)]
    union = set().union(*[set(s) for s in supports]) if supports else set()
    # strongest readout: cluster the support cloud into m groups, take each
    # cluster's top-D consensus features.
    clustered = _cluster_consensus(supports, X.shape[1], m, D)
    return top, clustered, union, len(freq), b


def _cluster_consensus(supports, p, m, D):
    from sklearn.cluster import AgglomerativeClustering
    uniq = list(set(supports))
    if len(uniq) < m:
        return [set(s) for s in uniq]
    M = np.zeros((len(uniq), p), dtype=float)
    for i, s in enumerate(uniq):
        M[i, list(s)] = 1.0
    try:
        lab = AgglomerativeClustering(n_clusters=m, metric="jaccard", linkage="average").fit_predict(M)
    except Exception:
        lab = AgglomerativeClustering(n_clusters=m).fit_predict(M)
    out = []
    for c in range(m):
        members = [uniq[i] for i in range(len(uniq)) if lab[i] == c]
        if not members:
            continue
        cnt = Counter(f for s in members for f in s)
        out.append(set(f for f, _ in cnt.most_common(D)))
    return out


def _best_ensemble(X, y, planted, D, K, budget, seed, randomized):
    """Ensemble's BEST-sol_f1 extraction; return (sol_f1, overlap_err, ndist)."""
    top, clustered, union, ndist, B = bootstrap_ensemble(
        X, y, D, K, budget_sec=budget, seed=seed, randomized=randomized)
    best_f1, best_err = 0.0, float("nan")
    for cand in (top, clustered):
        sol = {f"s{i}": {"support": list(s)} for i, s in enumerate(cand)}
        m = calculate_structural_metrics(sol, planted)
        if m["sol_f1"] >= best_f1:
            best_f1, best_err = m["sol_f1"], m["overlap_struct_err"]
    return best_f1, best_err, ndist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--D", type=int, default=5)
    args = ap.parse_args()

    # Regime sweep from EASY (n>>p, high SNR) to HARD (n<<p) at fixed overlap=2.
    # noise_std is the inverse-SNR knob. Each row: (label, n, p, noise, overlap)
    regimes = [
        ("n=p",               150, 150, 0.05, 2),
        ("n<p",               100, 200, 0.05, 2),
        ("n<<p (GEMSS turf)", 60, 300, 0.05, 2),
    ]
    # GEMSS uses its RECOMMENDED high-p config (scale_fixed anti-collapse), and we
    # report BOTH sol_f1 and overlap_struct_err vs the strongest cheap competitor
    # (randomised Lasso). sf_hp drops var_slab (frozen ladder sets variance).
    sf_hp = {k: v for k, v in GEMSS_HP.items() if k != "var_slab"}
    print("FAIR re-run: GEMSS=scale_fixed (recommended high-p config) vs Randomised-Lasso")
    print(f"(compute-matched, best extraction). overlap=2, mean over seeds={args.seeds}.\n")
    print(f"  {'regime':18s} {'n':>4s} {'p':>4s} | {'GEMSS f1':>8s} {'err':>5s} | {'RandL f1':>8s} {'err':>5s} {'ndist':>6s}")
    for label, n, p, noise, ov in regimes:
        gf, ge, rf, re_, nd = [], [], [], [], []
        for sd in args.seeds:
            X, y, truth, planted = generate_overlapping_dataset(
                n_samples=n, n_features=p, n_solutions=3, sparsity=args.D,
                latent_rank=2, overlap=ov, noise_std=noise, binarize=True,
                standardize=True, seed=sd)
            t = time.time()
            gsol = MechanismGEMSSWrapper("classification", "scalefixed", args.K, args.D, **sf_hp).fit(X, y)
            gt = time.time() - t
            gm = calculate_structural_metrics(gsol, planted)
            gf.append(gm["sol_f1"]); ge.append(gm["overlap_struct_err"])
            f1, err, ndist = _best_ensemble(X, y, planted, args.D, args.K, gt, sd, True)
            rf.append(f1); re_.append(err); nd.append(ndist)
        print(f"  {label:18s} {n:>4d} {p:>4d} | {np.mean(gf):8.2f} {np.nanmean(ge):5.2f} | "
              f"{np.mean(rf):8.2f} {np.nanmean(re_):5.2f} {np.mean(nd):6.0f}")


if __name__ == "__main__":
    main()
