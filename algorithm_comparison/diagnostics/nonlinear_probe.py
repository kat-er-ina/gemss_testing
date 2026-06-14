"""The original motivation: nonlinear feature engineering -> massively collinear
engineered space. Does GEMSS beat the strong cheap baseline HERE, where it
couldn't on the linear benchmarks?

Setup: a few correlated base features (driven by r latent factors) generate y
through a SPARSE NONLINEAR mechanism (squares + an interaction). We expand the
base into an engineered bank [x_i] + [x_i^2] + [x_i x_j], which is highly collinear
(squares/products of correlated bases). Ground truth = the engineered terms that
actually appear in y. Methods must recover them in the collinear engineered space.

Hypothesis: under heavy engineered collinearity the irrepresentable condition is
badly violated -> bootstrap-Lasso is extremely unstable (diffuse cloud over a huge
support space); GEMSS's structured prior + joint mixture may recover the true
terms and return a usable m-solution set. Honest test -- could be parity.

Compares GEMSS (logistic default) vs RandomizedLassoEnsembleWrapper (strong cheap
baseline), at moderate and large engineered dimension.
"""

import argparse
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import StandardScaler
from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper
from src.wrappers.sklearn_wrappers import RandomizedLassoEnsembleWrapper
from src.evaluation import calculate_metrics

GEMSS_HP = dict(prior="sss", var_slab=32.0, var_spike=0.1, lr=0.01, n_iter=6000,
                batch_size=16, weight_slab=0.9, weight_spike=0.1)


def make_nonlinear(d_base, n, r, seed):
    """Correlated base features (r latent factors) -> engineered bank; sparse
    nonlinear y. Returns standardized engineered X, binary y, true-term index set."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, r))
    A = rng.standard_normal((r, d_base))
    Xb = Z @ A + 0.1 * rng.standard_normal((n, d_base))
    Xb = StandardScaler().fit_transform(Xb)

    feats, names = [], []
    for i in range(d_base):
        feats.append(Xb[:, i]); names.append(("x", i))
    for i in range(d_base):
        feats.append(Xb[:, i] ** 2); names.append(("x2", i))
    for i in range(d_base):
        for j in range(i + 1, d_base):
            feats.append(Xb[:, i] * Xb[:, j]); names.append(("xx", i, j))
    E = np.stack(feats, 1)

    # sparse nonlinear generating terms (by name) -> ground-truth indices
    gen = [("x2", 0), ("x2", 1), ("xx", 2, 3), ("x", 4)]
    truth = set(names.index(g) for g in gen)
    w = {("x2", 0): 3.0, ("x2", 1): 3.0, ("xx", 2, 3): 4.0, ("x", 4): 2.0}
    ylat = sum(w[g] * E[:, names.index(g)] for g in gen)
    ylat = ylat - ylat.mean()
    y = (1 / (1 + np.exp(-ylat)) > 0.5).astype(float)

    Es = np.nan_to_num(StandardScaler().fit_transform(E))
    return Es, y, truth


def diversity(sols):
    sets = [set(s["support"]) for s in sols.values() if s.get("support")]
    nd = len({frozenset(s) for s in sets})
    return len(sets), nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--D", type=int, default=4)
    ap.add_argument("--K", type=int, default=6)
    args = ap.parse_args()

    configs = [("moderate", 20, 200), ("large", 35, 200)]  # (label, d_base, n)
    print("Nonlinear engineered space (squares + interactions); recover true terms.")
    print(f"D={args.D} K={args.K}, mean over seeds={args.seeds}.\n")
    print(f"  {'config':9s} {'p_eng':>6s} | {'GEMSS rec/F1':>13s} {'gt':>4s} | {'RandL rec/F1':>13s} {'#distinct':>9s} {'gemss_s':>7s} {'randl_s':>7s}")
    for label, d_base, n in configs:
        g_rec, g_f1, r_rec, r_f1, r_nd, gt, gtime, rtime = ([] for _ in range(8))
        for sd in args.seeds:
            X, y, truth = make_nonlinear(d_base, n, r=5, seed=sd)
            p = X.shape[1]; gt.append(p)
            t = time.time()
            gs = MechanismGEMSSWrapper("classification", "joint", args.K, args.D, **GEMSS_HP).fit(X, y)
            gtime.append(time.time() - t)
            gm = calculate_metrics(gs, truth, p)
            g_rec.append(gm["Recall"]); g_f1.append(gm["F1_Score"])
            t = time.time()
            rs = RandomizedLassoEnsembleWrapper(n_solutions=args.K, sparsity=args.D, n_restarts=300, seed=sd).fit(X, y)
            rtime.append(time.time() - t)
            rm = calculate_metrics(rs, truth, p)
            r_rec.append(rm["Recall"]); r_f1.append(rm["F1_Score"])
            r_nd.append(diversity(rs)[1])
        m = np.mean
        print(f"  {label:9s} {m(gt):6.0f} | {m(g_rec):.2f}/{m(g_f1):.2f}    {m(gt):4.0f} | "
              f"{m(r_rec):.2f}/{m(r_f1):.2f}    {m(r_nd):9.1f} {m(gtime):7.1f} {m(rtime):7.1f}")


if __name__ == "__main__":
    main()
