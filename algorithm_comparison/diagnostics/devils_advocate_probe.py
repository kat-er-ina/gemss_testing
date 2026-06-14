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


def bootstrap_ensemble(X, y, D, m, budget_sec, alpha=0.05, seed=0):
    """Fit L1-logistic on bootstrap subsamples until budget_sec is spent; return
    the top-m most-frequent distinct supports + all-feature union + #distinct + B."""
    rng = np.random.default_rng(seed)
    Xs = np.nan_to_num(StandardScaler().fit_transform(np.nan_to_num(X)))
    n = Xs.shape[0]
    supports = []
    t0 = time.time()
    b = 0
    while time.time() - t0 < budget_sec:
        idx = rng.choice(n, size=n, replace=True)  # bootstrap
        try:
            supports.append(lasso_support(Xs[idx], y[idx], D, alpha, seed + b))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--overlaps", nargs="+", type=int, default=[0, 2, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--D", type=int, default=5)
    args = ap.parse_args()

    print(f"Compute-matched: GEMSS (1 run) vs Bootstrap-Lasso ensemble (same wall-clock).")
    print(f"p={args.p} n={args.n} | mean over seeds={args.seeds}")
    for ov in args.overlaps:
        gm = {"f1": [], "solf1": [], "err": [], "t": []}
        bl = {"f1": [], "solf1": [], "err": [], "t": [], "B": [], "ndist": []}
        for sd in args.seeds:
            X, y, truth, planted = generate_overlapping_dataset(
                n_samples=args.n, n_features=args.p, n_solutions=3, sparsity=args.D,
                latent_rank=2, overlap=ov, noise_std=0.05, binarize=True,
                standardize=True, seed=sd)
            # GEMSS (default)
            t = time.time()
            gsol = MechanismGEMSSWrapper("classification", "joint", args.K, args.D, **GEMSS_HP).fit(X, y)
            gt = time.time() - t
            gm["t"].append(gt)
            gm["f1"].append(calculate_metrics(gsol, truth, args.p)["F1_Score"])
            gs = calculate_structural_metrics(gsol, planted)
            gm["solf1"].append(gs["sol_f1"]); gm["err"].append(gs["overlap_struct_err"])
            # Bootstrap-Lasso ensemble with the SAME wall-clock budget
            t = time.time()
            top, clustered, union, ndist, B = bootstrap_ensemble(X, y, args.D, args.K, budget_sec=gt, seed=sd)
            bt = time.time() - t
            bl["t"].append(bt); bl["B"].append(B); bl["ndist"].append(ndist)
            esol = {f"s{i}": {"support": list(s)} for i, s in enumerate(top)}
            bl["f1"].append(calculate_metrics(esol, truth, args.p)["F1_Score"])
            es = calculate_structural_metrics(esol, planted)
            bl["solf1"].append(es["sol_f1"]); bl["err"].append(es["overlap_struct_err"])
            csol = {f"c{i}": {"support": list(s)} for i, s in enumerate(clustered)}
            cs = calculate_structural_metrics(csol, planted)
            bl.setdefault("csolf1", []).append(cs["sol_f1"])
            bl.setdefault("cerr", []).append(cs["overlap_struct_err"])
            bl.setdefault("cf1", []).append(calculate_metrics(csol, truth, args.p)["F1_Score"])
        mu = lambda d, k: float(np.mean(d[k]))
        print(f"\n--- overlap={ov} ---")
        print(f"  GEMSS:            unionF1={mu(gm,'f1'):.2f} sol_f1={mu(gm,'solf1'):.2f} overlap_err={mu(gm,'err'):.2f}  time={mu(gm,'t'):.1f}s")
        print(f"  BootLasso top-m:  unionF1={mu(bl,'f1'):.2f} sol_f1={mu(bl,'solf1'):.2f} overlap_err={mu(bl,'err'):.2f}  time={mu(bl,'t'):.1f}s  (B={mu(bl,'B'):.0f}, {mu(bl,'ndist'):.0f} distinct)")
        print(f"  BootLasso clust:  unionF1={mu(bl,'cf1'):.2f} sol_f1={mu(bl,'csolf1'):.2f} overlap_err={mu(bl,'cerr'):.2f}  (m-cluster consensus)")


if __name__ == "__main__":
    main()
