"""Shared-asymptote check: GEMSS (variational instance) vs BB-SSL (Nie & Rockova
2020 -- the randomised-optimization SAMPLING instance for the SAME spike-and-slab
posterior, with matching-rate theory). Run on REGRESSION overlap data (the model
both target). If they tie, two approximations of the same posterior coincide --
confirming the unifying framing on the cleanest possible pair.

BB-SSL is called via the authors' R package through bb_ssl_bridge.R (needs R +
BBSSL on R_LIBS_USER). Its posterior beta samples are clustered into m solutions
(top-D supports -> agglomerative clustering), exactly as for the randomised-Lasso
ensemble, and scored with the same metrics.
"""

import argparse
import subprocess
import sys
import os
import tempfile
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.cluster import AgglomerativeClustering
from src.hard_data_factory import generate_overlapping_dataset
from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper
from src.evaluation import calculate_structural_metrics, calculate_metrics, _jaccard

BRIDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bb_ssl_bridge.R")
GEMSS_HP = dict(prior="sss", var_spike=0.1, lr=0.01, n_iter=6000, batch_size=16,
                weight_slab=0.9, weight_spike=0.1)  # scale_fixed sets variance


def cluster_to_m(supports, p, m, D):
    uniq = list({tuple(s) for s in supports})
    if len(uniq) <= m:
        return [set(s) for s in uniq]
    M = np.zeros((len(uniq), p))
    for i, s in enumerate(uniq):
        M[i, list(s)] = 1.0
    try:
        lab = AgglomerativeClustering(n_clusters=m, metric="jaccard", linkage="average").fit_predict(M)
    except Exception:
        lab = AgglomerativeClustering(n_clusters=m).fit_predict(M)
    out = []
    for c in range(m):
        mem = [uniq[i] for i in range(len(uniq)) if lab[i] == c]
        if mem:
            cnt = Counter(f for s in mem for f in s)
            out.append({f for f, _ in cnt.most_common(D)})
    return out


def run_bbssl(X, y, D, m, lambda0, lambda1, nsample=200, alpha=1.0):
    with tempfile.TemporaryDirectory() as td:
        xf, yf, of = (os.path.join(td, n) for n in ("X.csv", "y.csv", "out.csv"))
        np.savetxt(xf, X, delimiter=","); np.savetxt(yf, y, delimiter=",")
        r = subprocess.run(["Rscript", BRIDGE, xf, yf, of, str(lambda0), str(lambda1),
                            str(nsample), str(alpha)], capture_output=True, text=True)
        if not os.path.exists(of):
            raise RuntimeError("BB-SSL failed: " + r.stderr[-400:])
        beta = np.atleast_2d(np.loadtxt(of, delimiter=","))
        gamma = np.atleast_2d(np.loadtxt(of + ".g", delimiter=",")) if os.path.exists(of + ".g") else None
    supports = []
    for i, bb in enumerate(beta):
        if gamma is not None:
            active = np.where(gamma[i] < 0.5)[0]  # gamma is the SPIKE indicator; slab/included = <0.5
        else:
            active = np.where(np.abs(bb) > 0)[0]
        if active.size == 0:
            continue
        # rank active features by |beta|, keep up to D (the per-solution budget)
        order = active[np.argsort(np.abs(bb[active]))[::-1]][:D]
        supports.append(tuple(sorted(int(j) for j in order)))
    supports = [s for s in supports if s]
    clustered = cluster_to_m(supports, X.shape[1], m, D)
    return {f"s{i}": {"support": list(s)} for i, s in enumerate(clustered)}, len(set(supports))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlaps", nargs="+", type=int, default=[0, 2, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--D", type=int, default=5)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--lambda0", type=float, default=50.0)
    ap.add_argument("--lambda1", type=float, default=0.5)
    ap.add_argument("--nsample", type=int, default=200)
    args = ap.parse_args()
    print(f"GEMSS (variational) vs BB-SSL (sampling) on REGRESSION overlap data. "
          f"p={args.p} n={args.n} lambda=({args.lambda0},{args.lambda1}) NSample={args.nsample}")
    print(f"  {'overlap':>7s} | {'GEMSS f1/err':>13s} | {'BB-SSL f1/err':>13s} {'(#distinct)':>11s}")
    for ov in args.overlaps:
        g, ge, b, be, nd = [], [], [], [], []
        for sd in args.seeds:
            X, y, truth, planted = generate_overlapping_dataset(
                n_samples=args.n, n_features=args.p, n_solutions=3, sparsity=args.D,
                latent_rank=2, overlap=ov, noise_std=0.05, binarize=False, seed=sd)
            gm = calculate_structural_metrics(
                MechanismGEMSSWrapper("regression", "scalefixed", args.K, args.D, **GEMSS_HP).fit(X, y), planted)
            g.append(gm["sol_f1"]); ge.append(gm["overlap_struct_err"])
            try:
                sols, ndist = run_bbssl(X, y, args.D, args.K, args.lambda0, args.lambda1, args.nsample)
                bm = calculate_structural_metrics(sols, planted)
                b.append(bm["sol_f1"]); be.append(bm["overlap_struct_err"]); nd.append(ndist)
            except Exception as e:
                print(f"    ov={ov} seed={sd} BB-SSL ERR: {str(e)[:120]}")
        if b:
            print(f"  {ov:7d} | {np.mean(g):.2f}/{np.nanmean(ge):.2f}    | "
                  f"{np.mean(b):.2f}/{np.nanmean(be):.2f}    ({np.mean(nd):.0f})")


if __name__ == "__main__":
    main()
