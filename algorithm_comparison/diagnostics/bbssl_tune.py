"""Fair-tuning sweep for BB-SSL (instance B) on the synthetic overlap generator.
The three-way sweep ran BB-SSL with one untuned penalty pair and it produced a
diffuse posterior (poor set recovery). Before concluding anything about the
shared-asymptote claim we must give BB-SSL a fair hyperparameter selection on
THIS generator -- the same courtesy GEMSS gets with its recommended config.

Grid over (lambda0 spike, lambda1 slab); report mean sol_f1 over seeds per
overlap, so we can pick BB-SSL's best config and compare it honestly to GEMSS.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hard_data_factory import generate_overlapping_dataset
from src.evaluation import calculate_structural_metrics
from bb_ssl_probe import run_bbssl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda0", nargs="+", type=float, default=[5, 10, 30, 100, 300])
    ap.add_argument("--lambda1", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    ap.add_argument("--overlaps", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--D", type=int, default=5)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--nsample", type=int, default=200)
    ap.add_argument("--out", type=str, default="results/bbssl_tune.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    print(f"BB-SSL fair-tuning: lambda0={args.lambda0} lambda1={args.lambda1} "
          f"overlaps={args.overlaps} seeds={args.seeds}")
    for l0 in args.lambda0:
        for l1 in args.lambda1:
            if l1 >= l0:
                continue  # slab must be looser than spike
            for ov in args.overlaps:
                f1s, errs, nds = [], [], []
                for sd in args.seeds:
                    X, y, truth, planted = generate_overlapping_dataset(
                        n_samples=args.n, n_features=args.p, n_solutions=3,
                        sparsity=args.D, latent_rank=2, overlap=ov,
                        noise_std=0.05, binarize=False, seed=sd)
                    try:
                        sols, ndist = run_bbssl(X, y, args.D, args.K, l0, l1, args.nsample)
                        m = calculate_structural_metrics(sols, planted)
                        f1s.append(m["sol_f1"]); errs.append(m["overlap_struct_err"])
                        nds.append(ndist)
                    except Exception as e:
                        print(f"  l0={l0} l1={l1} ov={ov} sd={sd} ERR {str(e)[:100]}")
                if f1s:
                    rows.append(dict(lambda0=l0, lambda1=l1, overlap=ov,
                                     sol_f1=np.mean(f1s),
                                     overlap_struct_err=np.nanmean(errs),
                                     cloud=np.mean(nds)))
                    print(f"  l0={l0:6.1f} l1={l1:4.1f} ov={ov} -> "
                          f"sol_f1={np.mean(f1s):.3f} err={np.nanmean(errs):.3f} "
                          f"cloud={np.mean(nds):.0f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[*] wrote {len(rows)} rows -> {args.out}")
    # best config per overlap
    for ov in args.overlaps:
        sub = [r for r in rows if r["overlap"] == ov]
        if sub:
            b = max(sub, key=lambda r: r["sol_f1"])
            print(f"  BEST ov={ov}: lambda0={b['lambda0']} lambda1={b['lambda1']} "
                  f"sol_f1={b['sol_f1']:.3f}")


if __name__ == "__main__":
    main()
