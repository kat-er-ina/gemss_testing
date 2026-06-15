"""Three-way set-recovery sweep on REGRESSION overlap data (the cleanest model
where all instances apply). Fills the paper's RQ1 (A vs B shared asymptote) and
RQ2 (A,B,C vs the dedicated multiplicity methods) from ONE internally consistent
run: same generator, same seeds, same metrics for every method.

Instances of the unifying principle (approximate the multimodal spike-and-slab
posterior):
  A = GEMSS            (variational; scale_fixed high-p config)
  B = BB-SSL           (posterior bootstrap; authors' R package via bridge)
  C = RandLasso-ens    (heuristic resampling ensemble + clustering)
Dedicated multiplicity methods (the foils):
  EnumLasso (k-best by objective), StabilitySelection (collapses to a core).

Output: one tidy CSV (method, overlap, seed, sol_f1, overlap_struct_err,
n_distinct, sol_precision, sol_recall) + an aggregated table to stdout.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hard_data_factory import generate_overlapping_dataset
from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper
from src.wrappers.sklearn_wrappers import (
    RandomizedLassoEnsembleWrapper,
    StabilitySelectionWrapper,
)
from src.wrappers.enumlasso_wrapper import EnumLassoWrapper
from src.evaluation import calculate_structural_metrics
from bb_ssl_probe import run_bbssl  # principled posterior-bootstrap bridge

GEMSS_HP = dict(prior="sss", var_spike=0.1, lr=0.01, n_iter=6000, batch_size=16,
                weight_slab=0.9, weight_spike=0.1)  # scale_fixed sets variance


def n_distinct(sols):
    return len({frozenset(s["support"]) for s in sols.values() if s.get("support")})


def run_methods(X, y, planted, K, D, seed, lambda0, lambda1, nsample, methods):
    """Return {method: (sol_f1, overlap_struct_err, n_distinct, prec, rec)} for the
    selected `methods` subset."""
    out = {}

    def score(sols):
        m = calculate_structural_metrics(sols, planted)
        return (m["sol_f1"], m["overlap_struct_err"], n_distinct(sols),
                m["sol_precision"], m["sol_recall"])

    if "A_GEMSS" in methods:
        out["A_GEMSS"] = score(
            MechanismGEMSSWrapper("regression", "scalefixed", K, D, **GEMSS_HP).fit(X, y))

    if "B_BBSSL" in methods:
        try:
            sols, ndist = run_bbssl(X, y, D, K, lambda0, lambda1, nsample)
            f1, err, _, p, r = score(sols)
            out["B_BBSSL"] = (f1, err, ndist, p, r)
        except Exception as e:
            print(f"    [B_BBSSL ERR seed={seed}] {str(e)[:160]}")
            out["B_BBSSL"] = (np.nan, np.nan, np.nan, np.nan, np.nan)

    if "C_RandLasso" in methods:
        out["C_RandLasso"] = score(
            RandomizedLassoEnsembleWrapper(n_solutions=K, sparsity=D, task="regression",
                                           n_restarts=300, seed=seed).fit(X, y))

    if "EnumLasso" in methods:
        out["EnumLasso"] = score(
            EnumLassoWrapper(n_solutions=K, sparsity=D, task="regression", seed=seed).fit(X, y))
    if "StabilitySel" in methods:
        out["StabilitySel"] = score(
            StabilitySelectionWrapper(n_solutions=K, sparsity=D, task="regression",
                                      seed=seed).fit(X, y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlaps", nargs="+", type=int, default=[0, 2, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--D", type=int, default=5)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--lambda0", type=float, default=50.0)
    ap.add_argument("--lambda1", type=float, default=0.5)
    ap.add_argument("--nsample", type=int, default=200)
    ap.add_argument("--methods", nargs="+",
                    default=["A_GEMSS", "B_BBSSL", "C_RandLasso", "EnumLasso", "StabilitySel"],
                    help="subset of methods to run")
    ap.add_argument("--out", type=str, default="results/three_way_sweep.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    methods = set(args.methods)
    print(f"Three-way regression overlap sweep: n={args.n} p={args.p} D={args.D} "
          f"K={args.K} seeds={args.seeds} overlaps={args.overlaps} methods={sorted(methods)}")
    for ov in args.overlaps:
        for sd in args.seeds:
            X, y, truth, planted = generate_overlapping_dataset(
                n_samples=args.n, n_features=args.p, n_solutions=3, sparsity=args.D,
                latent_rank=2, overlap=ov, noise_std=0.05, binarize=False, seed=sd)
            res = run_methods(X, y, planted, args.K, args.D, sd,
                              args.lambda0, args.lambda1, args.nsample, methods)
            for method, (f1, err, nd, prec, rec) in res.items():
                rows.append(dict(method=method, overlap=ov, seed=sd, sol_f1=f1,
                                 overlap_struct_err=err, n_distinct=nd,
                                 sol_precision=prec, sol_recall=rec))
            print(f"  overlap={ov} seed={sd}: " +
                  " ".join(f"{m}={v[0]:.2f}" for m, v in res.items()))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[*] wrote {len(rows)} rows -> {args.out}")

    # aggregated table (mean over seeds), only methods actually present
    present = [m for m in ["A_GEMSS", "B_BBSSL", "C_RandLasso", "EnumLasso", "StabilitySel"]
               if any(r["method"] == m for r in rows)]
    methods = present
    print(f"\n{'method':14s} | " + " | ".join(f"ov={ov} f1/err".rjust(13)
                                              for ov in args.overlaps))
    for m in methods:
        cells = []
        for ov in args.overlaps:
            f1 = [r["sol_f1"] for r in rows if r["method"] == m and r["overlap"] == ov]
            er = [r["overlap_struct_err"] for r in rows if r["method"] == m and r["overlap"] == ov]
            cells.append(f"{np.nanmean(f1):.2f}/{np.nanmean(er):.2f}".rjust(13))
        print(f"{m:14s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
