"""Decide the hyperparameter-tuning strategy: is there a ground-truth-FREE signal
(final ELBO, or CV-predictive score) that tracks ground-truth RECOVERY across HP
settings? If yes, we can tune GEMSS in practice without the ground truth; if no,
we must rely on fixed regime-level defaults.

Protocol (see discussion):
- NEVER tune to the recovery metric on test problems.
- Sweep GEMSS's key HPs on a few overlap problems; for each setting record:
    recovery (sol_f1, overlap_struct_err), final ELBO, CV-predictive (pred_best).
- Report Spearman correlation of each INTERNAL signal with recovery. The internal
  signal that best predicts recovery is the legitimate Tier-2 tuning criterion.

Expectation from earlier findings: CV-predictive is near-constant on synthetic
(solutions equally predictive by design) -> weak signal; ELBO is the candidate.
"""

import argparse
import itertools
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hard_data_factory import generate_overlapping_dataset
from gemss.feature_selection.inference import BayesianFeatureSelector
from src.evaluation import calculate_structural_metrics, predictive_quality


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[m]))
    rb = np.argsort(np.argsort(b[m]))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def fit_with_elbo(X, y, planted, task, K, D, var_slab, var_spike, lr, n_iter, lam):
    """Fit a GEMSS selector directly so we can read the final ELBO."""
    from src.wrappers.logistic_gemss import LogisticBayesianFeatureSelector
    cls = LogisticBayesianFeatureSelector if task == "classification" else BayesianFeatureSelector
    sel = cls(n_features=X.shape[1], n_components=K, X=X, y=y, prior="sss",
              sss_sparsity=D, var_slab=var_slab, var_spike=var_spike, lr=lr,
              batch_size=16, n_iter=n_iter)
    hist = sel.optimize(regularize=(lam > 0), lambda_jaccard=lam, verbose=False)
    elbo = float(np.mean(hist["elbo"][-200:]))
    mu = sel.mixture.mu.detach().cpu().numpy()
    sols = {f"c{k}": {"support": np.argsort(np.abs(mu[k]))[::-1][:D].tolist()} for k in range(K)}
    sm = calculate_structural_metrics(sols, planted)
    pq = predictive_quality(X, y, sols, task)
    return elbo, sm, pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--overlaps", nargs="+", type=int, default=[0, 2, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    args = ap.parse_args()

    grid = list(itertools.product(
        [8.0, 32.0, 100.0],       # var_slab
        [0.01, 0.1],              # var_spike
        [0.005, 0.02],            # lr
        [0, 1000],                # lambda_jaccard
    ))
    K, D, n_iter, task = 6, 5, 3000, "classification"

    rows = []  # (elbo, pred_best, sol_f1, overlap_err)
    for ov in args.overlaps:
        for sd in args.seeds:
            X, y, _t, planted = generate_overlapping_dataset(
                n_samples=args.n, n_features=args.p, n_solutions=3, sparsity=D,
                latent_rank=2, overlap=ov, noise_std=0.05, binarize=True,
                standardize=True, seed=sd)
            for vs, vsp, lr, lam in grid:
                torch.manual_seed(sd)
                elbo, sm, pq = fit_with_elbo(X, y, planted, task, K, D, vs, vsp, lr, n_iter, lam)
                rows.append((elbo, pq["pred_best"], sm["sol_f1"], sm["overlap_struct_err"]))

    rows = np.array(rows, float)
    elbo, pred, f1, oerr = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    print(f"HP-criterion probe: {len(rows)} settings ({len(grid)} HPs x overlaps x seeds), p={args.p}.")
    print("Spearman correlation of each GROUND-TRUTH-FREE signal with recovery:")
    print(f"  ELBO        vs sol_f1:            {_spearman(elbo, f1):+.2f}   (want strongly +)")
    print(f"  ELBO        vs -overlap_err:      {_spearman(elbo, -oerr):+.2f}   (want strongly +)")
    print(f"  CV-pred(best) vs sol_f1:          {_spearman(pred, f1):+.2f}")
    print(f"  CV-pred(best) vs -overlap_err:    {_spearman(pred, -oerr):+.2f}")
    print(f"  [pred range: {np.nanmin(pred):.2f}-{np.nanmax(pred):.2f} (flat => weak signal)]")
    print("\nReading: a strong + correlation => that signal is a valid ground-truth-free")
    print("tuning criterion (Tier 2). If both weak => rely on fixed regime defaults (Tier 1).")


if __name__ == "__main__":
    main()
