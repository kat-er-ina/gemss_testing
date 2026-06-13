"""Foundational correctness check of GEMSS's variational inference itself.

Everything else rests on GEMSS's VI actually finding the posterior. On a small
(p=2) problem we brute-force the EXACT posterior on a grid -- using GEMSS's OWN
prior.log_prob and likelihood (-1/2 sum residual^2), so we test the OPTIMIZER, not
the model definition -- then check:
  (1) GEMSS's fitted mixture means sit at the true posterior MODES;
  (2) the ELBO respects its bound: ELBO <= log Z (the log-evidence we integrate
      numerically), and the gap is small.

Setup: 2 correlated features both predictive of y; a structured spike-and-slab
prior with sparsity 1 (exactly one active feature) makes the posterior BIMODAL
(mass on support {0} and on {1}). A correct multimodal VI must place components on
BOTH modes.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemss.feature_selection.inference import BayesianFeatureSelector


def main():
    rng = np.random.default_rng(0)
    n = 80
    z0 = rng.standard_normal(n)
    X = np.stack([z0 + 0.1 * rng.standard_normal(n),       # feature 0
                  z0 + 0.1 * rng.standard_normal(n)], 1)   # feature 1 (corr with 0)
    X = (X - X.mean(0)) / X.std(0)
    y = 3.0 * z0
    y = (y - y.mean())

    sel = BayesianFeatureSelector(n_features=2, n_components=4, X=X, y=y, prior="sss",
                                  sss_sparsity=1, var_slab=9.0, var_spike=0.01,
                                  lr=0.02, batch_size=32, n_iter=4000)

    # ---- exact posterior on a grid, using GEMSS's own prior + likelihood ----
    g = np.linspace(-6, 6, 241)
    B0, B1 = np.meshgrid(g, g)
    grid = torch.tensor(np.stack([B0.ravel(), B1.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        loglik = sel.log_likelihood(grid)           # -1/2 sum residual^2 (GEMSS's own)
        logprior = sel.prior.log_prob(grid)
        logpost = (loglik + logprior).numpy()
    dcell = (g[1] - g[0]) ** 2
    logZ = float(torch.logsumexp(torch.tensor(logpost), 0)) + np.log(dcell)

    # true modes: global max, then the best point outside a radius of it
    grid_np = grid.numpy()
    order = np.argsort(logpost)[::-1].copy()
    m1 = grid_np[order[0]]
    far = np.array([p for p in grid_np[order[:500]] if np.linalg.norm(p - m1) > 2.0])
    m2 = far[0]
    true_modes = np.array([m1, m2])

    # ---- fit GEMSS and read component means ----
    hist = sel.optimize(regularize=False, verbose=False)
    mus = sel.mixture.mu.detach().cpu().numpy()       # [4,2]
    elbo = float(np.mean(hist["elbo"][-200:]))

    # each true mode must have a GEMSS component near it
    dists = np.linalg.norm(true_modes[:, None, :] - mus[None, :, :], axis=2)  # [2,4]
    nearest = dists.min(1)
    print("p=2 bimodal spike-and-slab posterior (supports {0} and {1}).")
    print(f"  true modes:        {np.round(true_modes,2).tolist()}")
    print(f"  GEMSS comp means:  {np.round(mus,2).tolist()}")
    print(f"  dist(mode -> nearest comp): {np.round(nearest,2).tolist()}")
    print(f"  ELBO={elbo:.1f}  logZ(grid)={logZ:.1f}  gap={logZ-elbo:.1f}")
    modes_found = bool((nearest < 1.0).all())
    bound_ok = elbo <= logZ + 1.0          # ELBO must not exceed evidence (slack for noise)
    print(f"  modes_found={modes_found}  bound_ok(ELBO<=logZ)={bound_ok}")
    print("PASS" if (modes_found and bound_ok) else "FAIL")


if __name__ == "__main__":
    main()
