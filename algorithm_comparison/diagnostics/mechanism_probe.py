"""Test the two prior-art-motivated anti-collapse mechanisms on the overlap toy.

Motivated by anticollapse_analysis.md (see gemss_paper). Two small additions to
the GEMSS objective, both targeting what we measured:

- **kernel repulsion** (SVGD / repulsive-mixtures): replace the support-space
  Jaccard penalty with a parameter-space RBF repulsion on the component means,
  `-gamma * mean_{k<l} exp(-||mu_k - mu_l||^2 / h)`. Predicted to help without
  backfiring under feature overlap (unlike Jaccard).
- **likelihood tempering** (deterministic annealing): scale the likelihood by a
  schedule `beta(t): beta0 -> 1`. Principled form of our frozen variance ladder;
  targets the "likelihood (~n x) out-pulls the entropy repulsion" collapse.

NOTE on scaling: the likelihood is summed over n samples (~ -O(n)), so any
repulsion term must be scaled to compete (the Jaccard penalty already used
lambda up to 1000). kernel_gamma is therefore swept on a large scale, and
tempering (beta<1 early) is partly a way to let O(1) repulsion matter early.

Schemes: joint (baseline) | jaccard (existing penalty) | kernel | anneal |
anneal+kernel | scale_fixed (our current best, for reference).
"""

import argparse
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemss.feature_selection.inference import BayesianFeatureSelector
from src.hard_data_factory import generate_overlapping_dataset
from src.wrappers.logistic_gemss import LogisticBayesianFeatureSelector
from diagnostics.init_probe import _top_supports, _metrics


def _selector(task, X, y, K, D, var_slab, var_spike, lr, n_iter):
    cls = LogisticBayesianFeatureSelector if task == "classification" else BayesianFeatureSelector
    return cls(n_features=X.shape[1], n_components=K, X=X, y=y, prior="sss",
               sss_sparsity=D, var_slab=var_slab, var_spike=var_spike,
               lr=lr, batch_size=16, n_iter=n_iter)


def _beta_schedule(t, n_iter, beta0=0.05, warmup=0.5):
    """Linear anneal of the likelihood weight from beta0 to 1 over warmup*n_iter."""
    if warmup <= 0:
        return 1.0
    return float(min(1.0, beta0 + (1.0 - beta0) * (t / (warmup * n_iter))))


def _kernel_repulsion(mu, bandwidth=None):
    """Mean pairwise RBF kernel on component means (median-heuristic bandwidth)."""
    K = mu.shape[0]
    if K < 2:
        return mu.sum() * 0.0
    d2 = torch.cdist(mu, mu) ** 2
    iu = torch.triu_indices(K, K, offset=1, device=mu.device)
    off = d2[iu[0], iu[1]]
    h = bandwidth if bandwidth is not None else (off.detach().median() + 1e-8)
    return torch.exp(-off / h).mean()


def fit_custom(sel, n_iter, anneal=False, kernel_gamma=0.0, bandwidth=None,
               beta0=0.05, warmup=0.5):
    """GEMSS SGD loop with optional likelihood tempering + mean-space repulsion."""
    opt = sel.opt
    for t in range(n_iter):
        z, _ = sel.mixture.sample(sel.batch_size)
        beta = _beta_schedule(t, n_iter, beta0, warmup) if anneal else 1.0
        logp = sel.prior.log_prob(z) + beta * sel.log_likelihood(z)
        logq = sel.mixture.log_prob(z)
        obj = (logp - logq).mean()
        if kernel_gamma > 0:
            obj = obj - kernel_gamma * _kernel_repulsion(sel.mixture.mu, bandwidth)
        opt.zero_grad()
        (-obj).backward()
        opt.step()


def run_scheme(scheme, X, y, planted, K, D, task, n_iter, lr, var_slab, var_spike,
               jaccard_lam, kernel_gamma, seed):
    torch.manual_seed(seed)
    sel = _selector(task, X, y, K, D, var_slab, var_spike, lr, n_iter)
    if scheme in ("scale_fixed",):
        var_scales = np.geomspace(0.02, 2.0, K).astype(np.float32)
        with torch.no_grad():
            for k in range(K):
                sel.mixture._log_var[k].fill_(float(np.log(np.expm1(var_scales[k]))))
            sel.mixture._log_var.requires_grad_(False)
        sel.optimize(regularize=False, lambda_jaccard=0.0, verbose=False)
    elif scheme == "joint":
        sel.optimize(regularize=False, lambda_jaccard=0.0, verbose=False)
    elif scheme == "jaccard":
        sel.optimize(regularize=True, lambda_jaccard=jaccard_lam, verbose=False)
    elif scheme == "kernel":
        fit_custom(sel, n_iter, anneal=False, kernel_gamma=kernel_gamma)
    elif scheme == "anneal":
        fit_custom(sel, n_iter, anneal=True, kernel_gamma=0.0)
    elif scheme == "anneal+kernel":
        fit_custom(sel, n_iter, anneal=True, kernel_gamma=kernel_gamma)
    else:
        raise ValueError(scheme)
    mu = sel.mixture.mu.detach().cpu().numpy()
    return _metrics(_top_supports(mu, D), [set(s) for s in planted.values()], X.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", nargs="+", type=int, default=[100, 500])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--D", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--var_slab", type=float, default=32.0)
    ap.add_argument("--var_spike", type=float, default=0.1)
    ap.add_argument("--jaccard_lam", type=float, default=1000.0)
    ap.add_argument("--kernel_gamma", type=float, default=1000.0)
    ap.add_argument("--task", default="classification")
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--schemes", nargs="+",
                    default=["joint", "jaccard", "kernel", "anneal", "anneal+kernel", "scale_fixed"])
    args = ap.parse_args()

    print(f"task={args.task} n={args.n} K={args.K} D={args.D} overlap={args.overlap} "
          f"n_iter={args.n_iter} jaccard_lam={args.jaccard_lam} kernel_gamma={args.kernel_gamma} "
          f"| mean over seeds={args.seeds}")
    for p in args.ps:
        print(f"\n=== p={p} (n={args.n}, overlap={args.overlap}) ===")
        print(f"  {'scheme':14s} {'F1':>5s} {'recall':>6s} {'sol_jac':>7s} {'#distinct':>9s}")
        for scheme in args.schemes:
            agg = []
            for sd in args.seeds:
                X, y, _t, planted = generate_overlapping_dataset(
                    n_samples=args.n, n_features=p, n_solutions=3, sparsity=args.D,
                    latent_rank=2, overlap=args.overlap, noise_std=0.05,
                    binarize=(args.task == "classification"), standardize=True, seed=sd)
                agg.append(run_scheme(scheme, X, y, planted, args.K, args.D, args.task,
                                      args.n_iter, args.lr, args.var_slab, args.var_spike,
                                      args.jaccard_lam, args.kernel_gamma, sd))
            m = {k: np.mean([a[k] for a in agg]) for k in agg[0]}
            print(f"  {scheme:14s} {m['F1']:5.2f} {m['recall']:6.2f} "
                  f"{m['sol_jac']:7.2f} {m['n_distinct']:9.2f}")


if __name__ == "__main__":
    main()
