"""Initialization diagnostics for GEMSS on a controlled toy problem.

GEMSS initializes every mixture component identically in distribution:
`mu ~ N(0,1)` on ALL p features (models.py), with uniform weights and
sigma^2 ~ 0.69. There is no support-diversity or sparsity bias, so for large p
the optimizer must prune ~p-D features per component AND push components apart
from a symmetric start -- a classic mode-collapse setup for mixture models.

This script isolates the effect of initialization. It runs GEMSS on a small,
fully known problem under several init schemes and reports recovery + a
mode-collapse diagnostic (how many DISTINCT supports the K components end on).

Inits compared
--------------
- default       : N(0,1) on all features (current GEMSS behavior)
- diverse_sparse: each component = D random distinct features set to `mag`
                  (cheap, truth-agnostic heuristic)
- oracle        : component k placed exactly on planted support k (truth-init)
- oracle_pert   : oracle with a fraction of support features swapped for random
                  (probes the basin of attraction around the truth)

If oracle >> default, initialization is the bottleneck and diverse_sparse is the
practical lever. If oracle ~ default, the limit is the objective/optimization.
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


def _top_supports(mu: np.ndarray, D: int):
    """Per-component top-D supports (by |mu|) as sets."""
    return [set(np.argsort(np.abs(mu[k]))[::-1][:D].tolist()) for k in range(mu.shape[0])]


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _metrics(supports, planted, p):
    """Union F1 + per-solution matched mean Jaccard + #distinct + core coverage."""
    from scipy.optimize import linear_sum_assignment

    truth = set().union(*planted)
    union = set().union(*supports) if supports else set()
    inter = len(union & truth)
    rec = inter / len(truth) if truth else 0.0
    prec = inter / len(union) if union else 0.0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0

    sim = np.array([[_jaccard(set(p_), q) for q in supports] for p_ in planted])
    matched = np.zeros(len(planted))
    if supports:
        ri, ci = linear_sum_assignment(-sim)
        for i, j in zip(ri, ci):
            matched[i] = sim[i, j]
    core = set.intersection(*[set(s) for s in planted]) if planted else set()
    preds_core = sum(1 for q in supports if core and core.issubset(q))
    n_distinct = len({frozenset(s) for s in supports})
    return dict(F1=f1, recall=rec, sol_jac=float(matched.mean()),
                preds_with_core=preds_core, n_distinct=n_distinct, core=len(core))


def build_init(scheme, K, p, planted, D, rng, mag=2.0, pert=0.4, X=None, y=None):
    """Construct a [K, p] init-mu matrix for the given scheme."""
    mu = rng.standard_normal((K, p)).astype(np.float32)  # 'default'
    if scheme in ("default", "tight_var"):
        return mu  # random mean; tight_var differs only by the small init variance
    mu = np.zeros((K, p), dtype=np.float32)
    if scheme == "diverse_sparse":
        for k in range(K):
            idx = rng.choice(p, D, replace=False)
            mu[k, idx] = mag
        return mu
    if scheme in ("screen", "screen_conf"):
        # Truth-agnostic, data-driven warm start: rank features by |corr(X_j, y)|
        # then seed each component with a diverse, score-weighted subset of the top pool.
        Xf = np.nan_to_num(X, nan=0.0)
        yc = (y - y.mean())
        denom = (Xf.std(0) * yc.std() + 1e-12)
        score = np.abs((Xf - Xf.mean(0)).T @ yc / len(yc)) / denom
        M = min(p, max(K * D, 4 * D))
        pool = np.argsort(score)[::-1][:M]
        w = score[pool] / score[pool].sum()
        for k in range(K):
            idx = rng.choice(pool, size=D, replace=False, p=w)
            mu[k, idx] = mag
        return mu
    # oracle / oracle_pert: place component k on planted support k (cycle if K>nsol)
    psupports = [list(s) for s in planted.values()]
    for k in range(K):
        supp = list(psupports[k % len(psupports)])
        if scheme == "oracle_pert":
            n_swap = max(1, int(pert * len(supp)))
            keep = supp[: len(supp) - n_swap]
            pool = [j for j in range(p) if j not in supp]
            supp = keep + list(rng.choice(pool, n_swap, replace=False))
        mu[k, supp] = mag
    return mu


def run_once(scheme, X, y, planted, K, D, task, n_iter, lr, var_slab, var_spike,
             lam, regularize, seed, mag, pert, oracle_var=None):
    rng = np.random.default_rng(seed)
    cls = LogisticBayesianFeatureSelector if task == "classification" else BayesianFeatureSelector
    sel = cls(n_features=X.shape[1], n_components=K, X=X, y=y, prior="sss",
              sss_sparsity=D, var_slab=var_slab, var_spike=var_spike,
              lr=lr, batch_size=16, n_iter=n_iter)
    init_mu = build_init(scheme, K, X.shape[1], planted, D, rng, mag=mag, pert=pert, X=X, y=y)
    with torch.no_grad():
        sel.mixture.mu.copy_(torch.tensor(init_mu))
        if oracle_var is not None and scheme in ("oracle", "oracle_pert", "tight_var", "screen_conf"):
            # confident (small-variance) start on the injected support
            inv = torch.log(torch.expm1(torch.tensor(float(oracle_var))))
            sel.mixture._log_var.fill_(float(inv))
    sel.optimize(regularize=regularize, lambda_jaccard=lam, verbose=False)
    mu = sel.mixture.mu.detach().cpu().numpy()
    return _metrics(_top_supports(mu, D), [set(s) for s in planted.values()], X.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--D", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--var_slab", type=float, default=32.0)
    ap.add_argument("--var_spike", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--task", default="classification")
    ap.add_argument("--overlaps", nargs="+", type=int, default=[0, 2, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    ap.add_argument("--mag", type=float, default=2.0)
    ap.add_argument("--oracle_var", type=float, default=0.1)
    args = ap.parse_args()

    schemes = ["default", "tight_var", "screen", "screen_conf", "oracle_noconfvar", "oracle"]
    print(f"toy: p={args.p} n={args.n} K={args.K} D={args.D} task={args.task} "
          f"lam={args.lam} n_iter={args.n_iter} | metrics averaged over seeds={args.seeds}")
    for ov in args.overlaps:
        print(f"\n=== overlap={ov} (core={ov}) ===")
        print(f"  {'scheme':14s} {'F1':>5s} {'recall':>6s} {'sol_jac':>7s} {'core/sol':>8s} {'#distinct':>9s}")
        for scheme in schemes:
            agg = []
            for sd in args.seeds:
                X, y, _truth, planted = generate_overlapping_dataset(
                    n_samples=args.n, n_features=args.p, n_solutions=3, sparsity=args.D,
                    latent_rank=2, overlap=ov, noise_std=0.05,
                    binarize=(args.task == "classification"), standardize=True, seed=sd)
                m = run_once(scheme, X, y, planted, args.K, args.D, args.task,
                             args.n_iter, args.lr, args.var_slab, args.var_spike,
                             args.lam, lam_to_reg(args.lam), sd, args.mag, 0.4,
                             oracle_var=args.oracle_var)
                agg.append(m)
            mean = {k: np.mean([a[k] for a in agg]) for k in agg[0]}
            print(f"  {scheme:14s} {mean['F1']:5.2f} {mean['recall']:6.2f} "
                  f"{mean['sol_jac']:7.2f} {mean['preds_with_core']:8.2f} {mean['n_distinct']:9.2f}")


def lam_to_reg(lam):
    return lam > 0


if __name__ == "__main__":
    main()
