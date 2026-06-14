"""Hunt for PRACTICAL consequences of GEMSS's probabilistic by-products (ELBO,
mixture weights) that a restart+cluster ensemble structurally cannot match.

Deployment reality: you do NOT know how many distinct alternative explanations
exist. A randomised-Lasso ensemble + clustering must be TOLD n_clusters (=m); set
it too high and it manufactures spurious "alternatives". GEMSS instead exposes the
mixture weights alpha and the ELBO, which should signal how many solutions the
DATA actually supports.

Test 1 (alpha as an effective-#-solutions signal): generate data whose true number
of distinct solutions differs (UNIQUE: 1 full-rank support; MULTI: several
equivalent supports). Fit GEMSS with a GENEROUS budget K and read:
  - alpha perplexity exp(H(alpha)) = effective # active components;
  - # distinct recovered supports.
A good signal: small on UNIQUE data, larger on MULTI -> GEMSS "knows".

Test 2 (the ensemble's blind spot): the randomised-Lasso ensemble forced to K
clusters always returns K supports; on UNIQUE data those extra clusters are
SPURIOUS (don't match the one true support). We quantify the false multiplicity.
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.wrappers.logistic_gemss import LogisticBayesianFeatureSelector
from src.wrappers.sklearn_wrappers import RandomizedLassoEnsembleWrapper
from src.hard_data_factory import generate_overlapping_dataset
from src.evaluation import _jaccard

GEMSS_HP = dict(prior="sss", var_slab=32.0, var_spike=0.1, lr=0.01, n_iter=6000,
                batch_size=16, weight_slab=0.9, weight_spike=0.1)


def gen(kind, seed):
    if kind == "UNIQUE":   # 1 support, full-rank -> no equivalent solutions
        return generate_overlapping_dataset(n_samples=120, n_features=200, n_solutions=1,
            sparsity=5, latent_rank=5, overlap=0, noise_std=0.05, binarize=True, seed=seed)
    if kind == "FEW":      # 2 equivalent overlapping solutions
        return generate_overlapping_dataset(n_samples=120, n_features=200, n_solutions=2,
            sparsity=5, latent_rank=2, overlap=2, noise_std=0.05, binarize=True, seed=seed)
    return generate_overlapping_dataset(n_samples=120, n_features=200, n_solutions=4,  # MANY
        sparsity=5, latent_rank=2, overlap=2, noise_std=0.05, binarize=True, seed=seed)


def gemss_signals(X, y, K, D, seed):
    import torch
    torch.manual_seed(seed)
    sel = LogisticBayesianFeatureSelector(n_features=X.shape[1], n_components=K, X=X, y=y,
                                          sss_sparsity=D, **GEMSS_HP)
    sel.optimize(regularize=False, verbose=False)
    alpha = sel.mixture.get_alpha().detach().cpu().numpy()
    alpha = np.clip(alpha, 1e-12, 1)
    perplexity = float(np.exp(-(alpha * np.log(alpha)).sum()))  # effective # components
    mu = sel.mixture.mu.detach().cpu().numpy()
    sups = [frozenset(np.argsort(np.abs(mu[k]))[::-1][:D].tolist()) for k in range(K)]
    # distinct supports, merging near-duplicates (Jaccard > 0.8)
    distinct = []
    for s in sups:
        if not any(_jaccard(set(s), set(d)) > 0.8 for d in distinct):
            distinct.append(s)
    return perplexity, len(distinct)


def ensemble_false_multiplicity(X, y, truth, K, D, seed):
    sols = RandomizedLassoEnsembleWrapper(n_solutions=K, sparsity=D, n_restarts=300, seed=seed).fit(X, y)
    sets = [set(s["support"]) for s in sols.values() if s.get("support")]
    # a returned solution is "real" if it overlaps the true feature union substantially
    real = sum(1 for s in sets if len(s & truth) / max(len(s), 1) >= 0.6)
    return len(sets), real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--D", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    args = ap.parse_args()

    print("How-many-solutions signal: GEMSS alpha-perplexity & #distinct vs true count.")
    print(f"Generous budget K={args.K}. mean over seeds={args.seeds}.\n")
    print(f"  {'data':8s} {'true#':>5s} | {'GEMSS alpha-perplex':>19s} {'GEMSS #distinct':>15s} | {'ens #returned':>13s} {'ens #real':>9s}")
    truecount = {"UNIQUE": 1, "FEW": 2, "MANY": 4}
    for kind in ["UNIQUE", "FEW", "MANY"]:
        pp, gd, er, ereal = [], [], [], []
        for sd in args.seeds:
            X, y, truth, planted = gen(kind, sd)
            truth = set(truth)
            p, d = gemss_signals(X, y, args.K, args.D, sd)
            pp.append(p); gd.append(d)
            nret, nreal = ensemble_false_multiplicity(X, y, truth, args.K, args.D, sd)
            er.append(nret); ereal.append(nreal)
        print(f"  {kind:8s} {truecount[kind]:5d} | {np.mean(pp):19.1f} {np.mean(gd):15.1f} | {np.mean(er):13.1f} {np.mean(ereal):9.1f}")


if __name__ == "__main__":
    main()
