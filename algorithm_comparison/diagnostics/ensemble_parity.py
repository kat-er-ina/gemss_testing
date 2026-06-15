"""Fair-footing parity sweep for the randomised-Lasso ENSEMBLE (instance C),
addressing two confounds in the earlier wall-clock 'compute-matched' comparison:

  (fix 1) implementation-agnostic budget -- report recovery as a function of
          n_restarts (the count of independent fits), NOT wall-clock, so the
          comparison to GEMSS does not depend on torch-vs-sklearn-vs-R speed.
  (fix 2) HP selection -- sweep the Lasso penalty `alpha` (the ensemble's main
          knob), the same tuning courtesy GEMSS and BB-SSL received, instead of
          a single fixed default.

Efficient prefix-curve: collect R_max bootstrap randomised-Lasso supports ONCE
per (alpha, overlap, seed), then cluster the first R of them for each R in the
restart grid -> a nested recovery-vs-budget curve at no extra fitting cost.
Mirrors RandomizedLassoEnsembleWrapper exactly (same _prepare, _make_estimator,
_coef_magnitudes, wmin, average-jaccard clustering, top-D consensus).
"""

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.cluster import AgglomerativeClustering
from src.hard_data_factory import generate_overlapping_dataset
from src.wrappers.sklearn_wrappers import _prepare, _make_estimator, _coef_magnitudes
from src.evaluation import calculate_structural_metrics


def cluster_supports(supports, p, m, D, max_cluster=4000):
    # Keep the most FREQUENT distinct supports (the consensus region) and cap the
    # count so agglomerative clustering stays O(max_cluster^2) at huge restart
    # budgets. The dropped supports are rare singletons that would not seed a
    # cluster anyway; we log nothing here but the cap is documented in the paper.
    nonempty = [s for s in supports if s]
    if not nonempty:
        return {}
    freq = Counter(nonempty)
    uniq = [s for s, _ in freq.most_common(max_cluster)]
    if len(uniq) <= m:
        return {f"s{i}": {"support": list(s)} for i, s in enumerate(uniq)}
    M = np.zeros((len(uniq), p))
    for i, s in enumerate(uniq):
        M[i, list(s)] = 1.0
    try:
        lab = AgglomerativeClustering(n_clusters=m, metric="jaccard",
                                      linkage="average").fit_predict(M)
    except Exception:
        lab = AgglomerativeClustering(n_clusters=m).fit_predict(M)
    out = {}
    for c in range(m):
        mem = [uniq[i] for i in range(len(uniq)) if lab[i] == c]
        if mem:
            cnt = Counter(f for s in mem for f in s)
            out[f"s{c}"] = {"support": [f for f, _ in cnt.most_common(D)]}
    return out


def collect_supports(X, y, alpha, wmin, R_max, seed, D):
    Xs = _prepare(X)
    y = np.asarray(y).ravel().astype(float)
    rng = np.random.default_rng(seed)
    n, p = Xs.shape
    sups = []
    for b in range(R_max):
        idx = rng.choice(n, size=n, replace=True)
        Xb = Xs[idx] * rng.uniform(wmin, 1.0, size=p)
        model = _make_estimator("lasso", "regression", alpha, seed + b)
        try:
            model.fit(Xb, y[idx])
        except Exception:
            sups.append(())
            continue
        mag = _coef_magnitudes(model, p)
        sups.append(tuple(sorted(int(i) for i in np.argsort(mag)[::-1][:D])))
    return sups, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--restarts", nargs="+", type=int, default=[50, 100, 300, 1000, 3000])
    ap.add_argument("--overlaps", nargs="+", type=int, default=[0, 2, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--D", type=int, default=5)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--wmin", type=float, default=0.2)
    ap.add_argument("--out", type=str, default="results/ensemble_parity.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fcsv = open(args.out, "w", newline="")
    w = csv.DictWriter(fcsv, fieldnames=["alpha", "overlap", "seed", "n_restarts",
                                         "sol_f1", "overlap_struct_err", "n_distinct"])
    w.writeheader(); fcsv.flush()
    Rmax = max(args.restarts)
    for alpha in args.alphas:
        for ov in args.overlaps:
            for sd in args.seeds:
                X, y, truth, planted = generate_overlapping_dataset(
                    n_samples=args.n, n_features=args.p, n_solutions=3,
                    sparsity=args.D, latent_rank=2, overlap=ov, noise_std=0.05,
                    binarize=False, seed=sd)
                sups, p = collect_supports(X, y, alpha, args.wmin, Rmax, sd, args.D)
                for R in args.restarts:
                    sols = cluster_supports(sups[:R], p, args.K, args.D)
                    m = calculate_structural_metrics(sols, planted) if sols else None
                    f1 = m["sol_f1"] if m else float("nan")
                    err = m["overlap_struct_err"] if m else float("nan")
                    nd = len({s for s in sups[:R] if s})
                    w.writerow(dict(alpha=alpha, overlap=ov, seed=sd, n_restarts=R,
                                    sol_f1=f1, overlap_struct_err=err, n_distinct=nd))
                    fcsv.flush()
                print(f"  alpha={alpha} ov={ov} sd={sd} done "
                      f"(R={args.restarts}, last sol_f1={f1:.3f})", flush=True)
    fcsv.close()
    print(f"[*] wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
