"""Independent validation of our OWN baselines (masking, stability selection).

Bug-free-comparison discipline: confirm each baseline reproduces textbook
behavior on a controlled sparse-recovery problem BEFORE trusting it against GEMSS.

Checks
------
1. MaskingWrapper (Lasso/ElasticNet/logistic), 1 solution, on a well-conditioned
   sparse linear problem (n>p, low noise, k true features): must recover the true
   support (recall ~ 1.0). This is the regime where Lasso support recovery is
   theoretically guaranteed.
2. MaskingWrapper, 3 solutions, on the overlap generator at overlap=0 (3 disjoint
   equivalent supports): the union of 3 peeled sets should cover the true union.
3. StabilitySelectionWrapper on the same sparse problem: the stable core should be
   a high-precision subset of the true support (Meinshausen-Buhlmann behavior).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.wrappers.sklearn_wrappers import MaskingWrapper, StabilitySelectionWrapper
from src.hard_data_factory import generate_overlapping_dataset


def _sparse_linear(n=200, p=50, k=5, noise=0.1, seed=0, classify=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    support = sorted(rng.choice(p, k, replace=False).tolist())
    w = np.zeros(p); w[support] = rng.uniform(2, 5, k) * rng.choice([-1, 1], k)
    yc = X @ w + rng.normal(0, noise, n)
    y = (yc > np.median(yc)).astype(float) if classify else yc
    return X, y, set(support)


def _recall(sols, truth):
    union = set().union(*[s["support"] for s in sols.values()]) if sols else set()
    return len(union & truth) / len(truth), (len(union & truth) / len(union) if union else 0.0)


def main():
    ok = True

    print("== 1. Masking (1 solution) on well-conditioned sparse recovery (n=200,p=50,k=5) ==")
    for est, classify in [("lasso", False), ("elasticnet", False), ("logistic", True)]:
        X, y, truth = _sparse_linear(classify=classify, seed=1)
        m = MaskingWrapper(estimator=est, n_solutions=1, sparsity=5,
                           task="classification" if classify else "regression", alpha=0.05, seed=1)
        rec, prec = _recall(m.fit(X, y), truth)
        flag = "" if rec >= 0.8 else "  <-- LOW"
        print(f"  {est:11s} recall={rec:.2f} precision={prec:.2f}{flag}")
        ok &= rec >= 0.8

    print("\n== 2. Masking (3 solutions) on overlap=0 generator (union recovery) ==")
    X, y, truth, _ = generate_overlapping_dataset(n_samples=200, n_features=100, n_solutions=3,
                                                  sparsity=5, latent_rank=2, overlap=0,
                                                  noise_std=0.05, binarize=True, seed=2)
    m = MaskingWrapper(estimator="lasso", n_solutions=3, sparsity=5, task="classification", seed=2)
    rec, prec = _recall(m.fit(X, y), truth)
    print(f"  masking x3  union_recall={rec:.2f} precision={prec:.2f} (true union={len(truth)})")
    ok &= rec >= 0.7

    print("\n== 3. Stability selection: stable core should be high-precision subset of truth ==")
    X, y, truth = _sparse_linear(n=300, p=50, k=5, classify=True, seed=3)
    s = StabilitySelectionWrapper(sparsity=5, task="classification", alpha=0.05,
                                  n_bootstrap=100, threshold=0.6, seed=3)
    rec, prec = _recall(s.fit(X, y), truth)
    print(f"  stability   recall={rec:.2f} precision={prec:.2f} (expect high precision)")
    ok &= prec >= 0.8

    print("\n" + ("PASS" if ok else "FAIL -- a baseline does not reproduce textbook behavior"))


if __name__ == "__main__":
    main()
