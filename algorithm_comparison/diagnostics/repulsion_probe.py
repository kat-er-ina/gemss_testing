"""Demonstrate (or refute) the repulsion in GEMSS's joint mixture, and test a
variable-variance Gaussian scale mixture as an anti-collapse mechanism.

The whole point of GEMSS's JOINT optimization (vs running m independent VI fits)
is that the mixture-entropy term in the ELBO, -E_q[log q_mixture], repels
overlapping components. If that repulsion works, the joint mixture should recover
MORE DISTINCT solutions than independent fits (which, sharing no entropy term,
all slide into the dominant mode). At high p we observed mode collapse
(n_distinct -> 1.5-3 of K), suggesting the repulsion is being out-pulled by the
likelihood (which is summed over n samples, ~n times stronger per step).

Schemes
-------
- joint            : standard GEMSS (homogeneous, learnable variance).
- independent      : K separate 1-component fits (different seeds), pooled.
                     No shared mixture entropy => NO repulsion. The control.
- scale_fixed      : heterogeneous variance FROZEN per component, log-spaced from
                     var_lo (narrow specialists) to var_hi (broad "covers the
                     rest"). Breaks the collapse symmetry by construction.
- scale_learn      : heterogeneous variance INITIALIZED log-spaced but learnable.

Reported: union F1 / recall, per-solution matched Jaccard, n_distinct supports
(the collapse indicator), and (for scale schemes) whether the broad components
land on DIFFERENT features than the narrow ones.
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


def _inv_softplus(v):
    return float(np.log(np.expm1(v)))


def run_scheme(scheme, X, y, planted, K, D, task, n_iter, lr, var_slab, var_spike,
               lam, seed, var_lo=0.02, var_hi=2.0):
    p = X.shape[1]
    if scheme == "independent":
        # K independent single-component fits, different seeds -> pooled supports.
        supports = []
        for k in range(K):
            torch.manual_seed(1000 * seed + k)
            sel = _selector(task, X, y, 1, D, var_slab, var_spike, lr, n_iter)
            sel.optimize(regularize=(lam > 0), lambda_jaccard=lam, verbose=False)
            mu = sel.mixture.mu.detach().cpu().numpy()
            supports += _top_supports(mu, D)
        return _metrics(supports, [set(s) for s in planted.values()], p)

    torch.manual_seed(seed)
    sel = _selector(task, X, y, K, D, var_slab, var_spike, lr, n_iter)
    if scheme in ("scale_fixed", "scale_learn"):
        # log-spaced per-component variances: narrow specialists -> broad explorer
        var_scales = np.geomspace(var_lo, var_hi, K).astype(np.float32)
        with torch.no_grad():
            for k in range(K):
                sel.mixture._log_var[k].fill_(_inv_softplus(var_scales[k]))
        if scheme == "scale_fixed":
            sel.mixture._log_var.requires_grad_(False)  # freeze the scale ladder
    sel.optimize(regularize=(lam > 0), lambda_jaccard=lam, verbose=False)
    mu = sel.mixture.mu.detach().cpu().numpy()
    supports = _top_supports(mu, D)
    m = _metrics(supports, [set(s) for s in planted.values()], p)
    # diversity of the broad-vs-narrow halves (scale schemes only)
    if scheme.startswith("scale"):
        half = K // 2
        narrow = set().union(*supports[:half]) if half else set()
        broad = set().union(*supports[half:]) if half else set()
        m["broad_extra"] = len(broad - narrow)  # features the broad half adds
    return m


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
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--task", default="classification")
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--var_lo", type=float, default=0.02)
    ap.add_argument("--var_hi", type=float, default=2.0)
    args = ap.parse_args()

    schemes = ["joint", "independent", "scale_fixed", "scale_learn"]
    print(f"task={args.task} n={args.n} K={args.K} D={args.D} overlap={args.overlap} "
          f"lam={args.lam} n_iter={args.n_iter} scale_var=[{args.var_lo},{args.var_hi}] "
          f"| mean over seeds={args.seeds}")
    for p in args.ps:
        print(f"\n=== p={p} (n={args.n}, overlap={args.overlap}) ===")
        print(f"  {'scheme':14s} {'F1':>5s} {'recall':>6s} {'sol_jac':>7s} {'#distinct':>9s} {'broad_extra':>11s}")
        for scheme in schemes:
            agg = []
            for sd in args.seeds:
                X, y, _t, planted = generate_overlapping_dataset(
                    n_samples=args.n, n_features=p, n_solutions=3, sparsity=args.D,
                    latent_rank=2, overlap=args.overlap, noise_std=0.05,
                    binarize=(args.task == "classification"), standardize=True, seed=sd)
                agg.append(run_scheme(scheme, X, y, planted, args.K, args.D, args.task,
                                      args.n_iter, args.lr, args.var_slab, args.var_spike,
                                      args.lam, sd, args.var_lo, args.var_hi))
            mean = {k: np.mean([a.get(k, np.nan) for a in agg]) for k in agg[0]}
            be = mean.get("broad_extra", float("nan"))
            print(f"  {scheme:14s} {mean['F1']:5.2f} {mean['recall']:6.2f} "
                  f"{mean['sol_jac']:7.2f} {mean['n_distinct']:9.2f} "
                  f"{be:11.2f}")


if __name__ == "__main__":
    main()
