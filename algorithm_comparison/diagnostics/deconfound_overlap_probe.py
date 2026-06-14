"""De-confound the RECOVERY axis: is the sharing-capable-vs-disjoint method gap
driven by OVERLAP, or by the (coupled) union size / solution budget?

In this generator union = overlap + k*(sparsity-overlap), so overlap cannot be
varied with union, sparsity, and k all fixed. We isolate it with two matched
series at sparsity=5:
  A (overlap effect):   union & sparsity fixed, vary overlap via k:
                        A0=(ov0,k4,union20)  vs  A2=(ov2,k6,union20)
  B (union/budget only): overlap=0, vary union/k:
                        B0=(ov0,k4,union20)  vs  B1=(ov0,k6,union30)
If the disjoint-forcing method's deficit (overlap_struct_err up, sol_f1 down)
appears in A (overlap changes) but NOT in B (only union/budget changes), overlap
is the genuine driver -- not difficulty.
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hard_data_factory import generate_overlapping_dataset
from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper
from src.wrappers.sklearn_wrappers import MaskingWrapper, RandomizedLassoEnsembleWrapper
from src.evaluation import calculate_structural_metrics, calculate_metrics

GEMSS_HP = dict(prior="sss", var_spike=0.1, lr=0.01, n_iter=6000, batch_size=16,
                weight_slab=0.9, weight_spike=0.1)  # scale_fixed sets variance


def run_config(label, n_sol, sparsity, overlap, p, n, seeds):
    union = overlap + n_sol * (sparsity - overlap)
    rows = {m: {"sol_f1": [], "err": [], "rec": []} for m in ["GEMSS", "Masking", "RandLasso"]}
    for sd in seeds:
        X, y, truth, planted = generate_overlapping_dataset(
            n_samples=n, n_features=p, n_solutions=n_sol, sparsity=sparsity,
            latent_rank=2, overlap=overlap, noise_std=0.05, binarize=True, seed=sd)
        methods = {
            "GEMSS": MechanismGEMSSWrapper("classification", "scalefixed", n_sol, sparsity, **GEMSS_HP),
            "Masking": MaskingWrapper("logistic", n_sol, sparsity, "classification", alpha=0.05),
            "RandLasso": RandomizedLassoEnsembleWrapper(n_solutions=n_sol, sparsity=sparsity, n_restarts=250, seed=sd),
        }
        for m, model in methods.items():
            sols = model.fit(X, y)
            sm = calculate_structural_metrics(sols, planted)
            um = calculate_metrics(sols, truth, p)
            rows[m]["sol_f1"].append(sm["sol_f1"]); rows[m]["err"].append(sm["overlap_struct_err"])
            rows[m]["rec"].append(um["Recall"])
    print(f"\n[{label}] n_sol={n_sol} sparsity={sparsity} overlap={overlap} union={union} (true_dissim varies)")
    for m in ["GEMSS", "Masking", "RandLasso"]:
        d = rows[m]
        print(f"    {m:10s} sol_f1={np.mean(d['sol_f1']):.2f}  overlap_err={np.nanmean(d['err']):.2f}  recall={np.mean(d['rec']):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    args = ap.parse_args()
    print("=== Series A: OVERLAP effect (union=20, sparsity=5 held fixed) ===")
    run_config("A0", 4, 5, 0, args.p, args.n, args.seeds)
    run_config("A2", 6, 5, 2, args.p, args.n, args.seeds)
    print("\n=== Series B: UNION/BUDGET effect only (overlap=0) ===")
    run_config("B0", 4, 5, 0, args.p, args.n, args.seeds)
    run_config("B1", 6, 5, 0, args.p, args.n, args.seeds)
    print("\nRead: if Masking overlap_err/sol_f1 moves A0->A2 but is flat B0->B1,")
    print("overlap (not union/budget) drives the disjoint-method deficit.")


if __name__ == "__main__":
    main()
