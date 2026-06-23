"""RQ5 (honest negative): do GEMSS's mixture weights reveal the number of
solutions? Vary the TRUE number of planted solutions, fit GEMSS with a generous
fixed budget K, and read (a) the weight perplexity exp(entropy(alpha)) and
(b) the number of DISTINCT recovered supports. If GEMSS "knew" the count, the
perplexity / distinct-count would track the true number; we test that.

FAIR-COUNT BENCHMARK (answers KH's RQ5 objection): the original RQ5 used
``generate_overlapping_dataset`` (rank-2 latent shared across all signal
features), where ANY full-rank subset of the signal union is a valid solution --
so the true count is combinatorial, not ``nsol``, and near-uniform weights are
*correct*, not a failure. Here we use ``generate_distinct_solutions``: each
solution is a DISJOINT full-rank block of a shared response signal + PRIVATE
nuisances, so the number of valid size-D supports is EXACTLY ``nsol`` (planted
blocks reconstruct y at R^2~0.96 vs ~0.66 for cross-block subsets). On this
benchmark the count is well-defined, GEMSS recovers it (union_f1 reported), and
we ask whether the weights track it. We also sweep K to refute "K was too low".

One row per (nsol, K, seed). Aggregate with mean+-95%CI per (nsol, K).
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hard_data_factory import generate_distinct_solutions
from src.evaluation import calculate_metrics
from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper

GEMSS_HP = dict(prior="sss", var_spike=0.1, lr=0.01, n_iter=6000, batch_size=16,
                weight_slab=0.9, weight_spike=0.1)


def perplexity(alpha):
    a = np.asarray(alpha, dtype=float)
    a = a[a > 0]
    if a.size == 0:
        return float("nan")
    ent = -np.sum(a * np.log(a))
    return float(np.exp(ent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsols", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--Ks", type=int, nargs="+", default=[12],
                    help="component budgets to sweep (K-robustness: refute 'K too low')")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--D", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 26)))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["nsol", "K", "seed", "perplexity",
                                      "n_distinct", "n_active", "union_f1"])
    w.writeheader(); f.flush()
    for nsol in args.nsols:
        for K in args.Ks:
            for sd in args.seeds:
                X, y, truth, planted = generate_distinct_solutions(
                    n_samples=args.n, n_features=args.p, n_solutions=nsol,
                    sparsity=args.D, noise_std=0.05, binarize=False, seed=sd)
                try:
                    wr = MechanismGEMSSWrapper("regression", "joint", K, args.D, **GEMSS_HP)
                    sols = wr.fit(X, y)
                    supp = [frozenset(s["support"]) for s in sols.values() if s.get("support")]
                    ndist = len(set(supp))
                    pp = perplexity(wr.alpha_) if wr.alpha_ is not None else float("nan")
                    # "active" components: weight above uniform/10 threshold
                    nact = int(np.sum(np.asarray(wr.alpha_) > (1.0 / K) / 10)) if wr.alpha_ is not None else -1
                    uf1 = float(calculate_metrics(sols, set(truth), args.p)["F1_Score"])
                except Exception as e:
                    print(f"  nsol={nsol} K={K} sd={sd} ERR {str(e)[:120]}", flush=True)
                    pp, ndist, nact, uf1 = float("nan"), -1, -1, float("nan")
                w.writerow(dict(nsol=nsol, K=K, seed=sd, perplexity=pp,
                                n_distinct=ndist, n_active=nact, union_f1=uf1)); f.flush()
                print(f"  nsol={nsol} K={K} sd={sd} -> perplexity={pp:.2f} "
                      f"n_distinct={ndist} n_active={nact} union_f1={uf1:.2f}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
