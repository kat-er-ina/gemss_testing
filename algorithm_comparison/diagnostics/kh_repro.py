"""Reconciliation experiment: reproduce KH's ORIGINAL benchmark
(generate_multi_solution_data) at high p, to test whether GEMSS "shines" there
(as she observed) -- isolating the cause of the high-p collapse seen on our
equal-variance overlap generator.

KH's generator: signal features ~ N(0,1) (std 1.0), noise features ~ N(0,0.01)
(std 0.01) -> signal is 100x louder, so it is variance-separable; extra solutions
are linear combinations of the first (same span), disjoint support indices. We feed
X AS-IS (no extra standardisation) so the amplitude crutch is preserved -- that is
the whole point of the comparison. Same methods, grids, scoring as tuned_compare.

Output one row per (method, n, p, seed, hp); aggregate to mean+-CI / win-rates.
"""
import argparse, csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                       # algorithm_comparison
sys.path.insert(0, HERE)                                        # diagnostics (tuned_compare)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "gemss"))

from gemss.data_handling.generate_artificial_dataset import generate_multi_solution_data
from src.evaluation import calculate_structural_metrics, calculate_metrics, _jaccard
from tuned_compare import grid, fit_sols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--p", type=int, default=5000)
    ap.add_argument("--D", type=int, default=10)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--nrestarts", type=int, default=1000)
    ap.add_argument("--noise", type=float, default=0.01, help="KH default noise_data_std")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def dissim(sols):
        sets = [set(s["support"]) for s in sols.values() if s.get("support")]
        if len(sets) < 2: return 0.0
        return float(np.mean([1 - _jaccard(sets[a], sets[b])
                              for a in range(len(sets)) for b in range(a+1, len(sets))]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["method", "n", "p", "seed", "hp",
                                      "sol_f1", "union_f1", "dissim"])
    w.writeheader(); f.flush()
    for seed in args.seeds:
        df, resp, gen_sols, _ = generate_multi_solution_data(
            n_samples=args.n, n_features=args.p, n_solutions=3, sparsity=args.D,
            noise_data_std=args.noise, binarize=False, random_seed=seed)
        # columns are feature_0..feature_{p-1} in order -> name index == column index
        X = df.to_numpy(dtype=float)
        y = np.asarray(resp, dtype=float)
        planted = {k: [int(s.split("_")[1]) for s in feats] for k, feats in gen_sols.items()}
        truth = set(i for v in planted.values() for i in v)
        for hp in grid(args.method):
            try:
                sols = fit_sols(args.method, hp, X, y, args.D, args.m, args.nrestarts)
                sm = calculate_structural_metrics(sols, planted)
                uf = calculate_metrics(sols, truth, X.shape[1])["F1_Score"]
                sf, di = sm["sol_f1"], dissim(sols)
            except Exception as e:
                print(f"  {args.method} sd={seed} hp={hp} ERR {str(e)[:120]}", flush=True)
                sf, uf, di = float("nan"), float("nan"), float("nan")
            w.writerow(dict(method=args.method, n=args.n, p=args.p, seed=seed,
                            hp=str(hp), sol_f1=sf, union_f1=uf, dissim=di)); f.flush()
            print(f"  {args.method} n={args.n} p={args.p} sd={seed} hp={hp} -> u={uf:.3f} s={sf:.3f}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
