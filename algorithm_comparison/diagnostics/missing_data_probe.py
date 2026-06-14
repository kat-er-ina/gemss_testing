"""Is GEMSS's NATIVE missing-data handling a demonstrable advantage over the
STRONG cheap baseline (which must impute)? GEMSS marginalises over missing entries
(proper MAR), while imputation distorts the covariance -- predicted to hurt most in
the correlated/overlap regime. We give the ensemble its BEST shot: mean-impute AND
iterative (MICE-like) imputation. If GEMSS's lead over the best-imputed ensemble
GROWS with missingness, the by-product is real.

Data: overlap=2, p=200, n=100; vary nan_ratio. Metrics: per-solution recovery
(sol_f1) + overlap-structure error. GEMSS gets the NaNs natively; the ensemble
gets imputed X.
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from src.hard_data_factory import generate_overlapping_dataset
from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper
from src.wrappers.sklearn_wrappers import RandomizedLassoEnsembleWrapper
from src.evaluation import calculate_structural_metrics

GEMSS_HP = dict(prior="sss", var_slab=32.0, var_spike=0.1, lr=0.01, n_iter=6000,
                batch_size=16, weight_slab=0.9, weight_spike=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nans", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--D", type=int, default=5)
    args = ap.parse_args()

    print("Native-NaN GEMSS vs Randomised-Lasso ensemble (mean-impute & iterative-impute)")
    print(f"overlap=2, p=200, n=100, mean over seeds={args.seeds}. Metric: sol_f1 (overlap_err)\n")
    print(f"  {'nan%':>5s} | {'GEMSS native':>14s} | {'ens mean-imp':>14s} | {'ens iter-imp':>14s}")
    for nan in args.nans:
        g, em, ei = [], [], []
        ge, eme, eie = [], [], []
        for sd in args.seeds:
            X, y, truth, planted = generate_overlapping_dataset(
                n_samples=100, n_features=200, n_solutions=3, sparsity=args.D,
                latent_rank=2, overlap=2, noise_std=0.05, nan_ratio=nan,
                binarize=True, seed=sd)
            # GEMSS: native NaN
            gm = calculate_structural_metrics(
                MechanismGEMSSWrapper("classification", "joint", args.K, args.D, **GEMSS_HP).fit(X, y),
                planted)
            g.append(gm["sol_f1"]); ge.append(gm["overlap_struct_err"])
            # ensemble with mean imputation (wrapper's default _prepare mean-imputes)
            mm = calculate_structural_metrics(
                RandomizedLassoEnsembleWrapper(n_solutions=args.K, sparsity=args.D, n_restarts=250, seed=sd).fit(X, y),
                planted)
            em.append(mm["sol_f1"]); eme.append(mm["overlap_struct_err"])
            # ensemble with iterative (MICE-like) imputation -- its best shot
            if np.isnan(X).any():
                Xi = IterativeImputer(max_iter=10, random_state=sd).fit_transform(X)
            else:
                Xi = X
            im = calculate_structural_metrics(
                RandomizedLassoEnsembleWrapper(n_solutions=args.K, sparsity=args.D, n_restarts=250, seed=sd).fit(Xi, y),
                planted)
            ei.append(im["sol_f1"]); eie.append(im["overlap_struct_err"])
        mu = np.mean
        print(f"  {nan*100:4.0f}% | {mu(g):.2f} ({np.nanmean(ge):.2f})   | "
              f"{mu(em):.2f} ({np.nanmean(eme):.2f})   | {mu(ei):.2f} ({np.nanmean(eie):.2f})")


if __name__ == "__main__":
    main()
