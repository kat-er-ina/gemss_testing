"""Fully-tuned comparison: run ONE method over its OWN hyperparameter grid on a
synthetic overlap problem and emit sol_f1 for every grid point (one row per
hp-setting). Aggregation later picks each method's best HP by seed-mean -- so
every method, including the ones we expect to lose, is shown at its best.

Per coauthor (KH): tune tau AND the underlying selector for ALFESE (MI expected
best); tune alpha for ensemble/masking/stability; rho for EnumLasso; lambda for
BB-SSL; K (#components) for GEMSS.
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hard_data_factory import generate_overlapping_dataset
from src.evaluation import calculate_structural_metrics, calculate_metrics, _jaccard
from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper
from src.wrappers.sklearn_wrappers import (RandomizedLassoEnsembleWrapper,
                                           MaskingWrapper, StabilitySelectionWrapper)
from src.wrappers.enumlasso_wrapper import EnumLassoWrapper
from src.wrappers.alfese_wrapper import AlfeseWrapper
from bb_ssl_probe import run_bbssl
from ensemble_parity import cluster_supports

GEMSS_HP = dict(prior="sss", var_spike=0.1, lr=0.01, n_iter=6000, batch_size=16,
                weight_slab=0.9, weight_spike=0.1)


def grid(method):
    if method == "GEMSS":  return [{"K": k} for k in (3, 6, 12, 18, 24)]
    if method == "GEMSSjoint":  return [{"K": k} for k in (3, 6, 12, 18, 24)]  # learned variances (KH's default)
    if method == "GEMSSJ": return [{"K": 6, "lj": lj} for lj in (0, 0.3, 1, 3, 10, 100, 1000)]
    if method == "GEMSSJHI": return [{"K": 6, "lj": lj} for lj in (1000, 3000, 10000, 30000, 100000, 1000000)]
    if method == "GEMSSJjoint": return [{"K": 6, "lj": lj} for lj in (0, 0.3, 1, 3, 10, 100, 1000, 3000, 10000, 30000, 100000, 1000000)]  # joint + jaccard, full bracket
    if method == "ENS":    return [{"alpha": a} for a in (0.005, 0.01, 0.02, 0.05)]
    if method == "BBSSL":  return [{"l0": l0, "l1": l1} for l0 in (5, 10, 30) for l1 in (0.1, 0.5)]
    if method == "ENUM":   return [{"rho": r} for r in (0.01, 0.02, 0.05, 0.1)]
    if method == "STAB":   return [{"alpha": a} for a in (0.01, 0.05, 0.1)]
    if method == "ALFESE": return [{"sel": "mi", "tau": t}
                                   for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    if method == "MASK":   return [{"alpha": a} for a in (0.01, 0.05, 0.1)]
    raise ValueError(method)


def fit_sols(method, hp, X, y, D, m, nrestarts):
    """Every method returns exactly m final solutions. GEMSS fits K (>= m)
    components as internal capacity, then CLUSTERS them down to m -- the same
    agglomerative-Jaccard consensus the ensemble applies to its restart cloud,
    so the reduction to m is identical across methods (no 'best-m-of-K' credit;
    GEMSS's near-equal component weights make weight-ranking meaningless)."""
    p = X.shape[1]
    if method == "GEMSS":
        sols = MechanismGEMSSWrapper("regression", "scalefixed", hp["K"], D, **GEMSS_HP).fit(X, y)
        supp = [tuple(sorted(s["support"])) for s in sols.values() if s.get("support")]
        return cluster_supports(supp, p, m, D)
    if method == "GEMSSjoint":                           # KH's default: LEARNED variances
        sols = MechanismGEMSSWrapper("regression", "joint", hp["K"], D, **GEMSS_HP).fit(X, y)
        supp = [tuple(sorted(s["support"])) for s in sols.values() if s.get("support")]
        return cluster_supports(supp, p, m, D)          # K components -> m consensus
    if method in ("GEMSSJ", "GEMSSJHI", "GEMSSJjoint"):  # GEMSS + Jaccard diversity penalty
        mech = "joint" if method == "GEMSSJjoint" else "scalefixed"
        sols = MechanismGEMSSWrapper("regression", mech, hp["K"], D,
                                     lambda_jaccard=hp["lj"], **GEMSS_HP).fit(X, y)
        supp = [tuple(sorted(s["support"])) for s in sols.values() if s.get("support")]
        return cluster_supports(supp, p, m, D)
    if method == "ENS":
        return RandomizedLassoEnsembleWrapper(n_solutions=m, sparsity=D, task="regression",
                                              alpha=hp["alpha"], n_restarts=nrestarts).fit(X, y)
    if method == "BBSSL":
        return run_bbssl(X, y, D, m, hp["l0"], hp["l1"], 200)[0]
    if method == "ENUM":
        return EnumLassoWrapper(n_solutions=m, sparsity=D, task="regression", rho=hp["rho"]).fit(X, y)
    if method == "STAB":
        return StabilitySelectionWrapper(n_solutions=m, sparsity=D, task="regression",
                                         alpha=hp["alpha"]).fit(X, y)
    if method == "ALFESE":
        return AlfeseWrapper(n_solutions=m, task="regression", selector_type=hp["sel"],
                             tau=hp["tau"], k=D).fit(X, y)
    if method == "MASK":
        return MaskingWrapper("lasso", m, D, "regression", alpha=hp["alpha"]).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--D", type=int, default=10)
    ap.add_argument("--m", type=int, default=3, help="# final solutions every method returns")
    ap.add_argument("--nrestarts", type=int, default=3000)
    ap.add_argument("--noise", type=float, default=0.05, help="generator noise_std (RQ6 noise sweep)")
    ap.add_argument("--nan", type=float, default=0.0, help="generator nan_ratio / missing fraction (RQ6)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="run all these seeds (paired across methods); one row per seed/hp")
    ap.add_argument("--torchseed", action="store_true",
                    help="also seed torch per fit (reduces GEMSS run-to-run variance)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def _dissim(sols):
        sets = [set(s["support"]) for s in sols.values() if s.get("support")]
        if len(sets) < 2:
            return 0.0
        ds = [1 - _jaccard(sets[a], sets[b]) for a in range(len(sets)) for b in range(a + 1, len(sets))]
        return float(np.mean(ds))

    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["method", "n", "p", "overlap", "noise", "nan", "seed", "hp",
                                      "sol_f1", "union_f1", "dissim", "overlap_struct_err"])
    w.writeheader(); f.flush()
    for seed in args.seeds:
        X, y, truth, planted = generate_overlapping_dataset(
            n_samples=args.n, n_features=args.p, n_solutions=3, sparsity=args.D,
            latent_rank=2, overlap=args.overlap, noise_std=args.noise, nan_ratio=args.nan,
            binarize=False, seed=seed)
        for hp in grid(args.method):
            if args.torchseed:
                import torch; torch.manual_seed(seed)
            try:
                sols = fit_sols(args.method, hp, X, y, args.D, args.m, args.nrestarts)
                m = calculate_structural_metrics(sols, planted)
                uf = calculate_metrics(sols, set(truth), X.shape[1])["F1_Score"]
                f1, err, dis = m["sol_f1"], m["overlap_struct_err"], _dissim(sols)
            except Exception as e:
                print(f"  {args.method} sd={seed} hp={hp} ERR {str(e)[:120]}", flush=True)
                f1, err, uf, dis = float("nan"), float("nan"), float("nan"), float("nan")
            w.writerow({"method": args.method, "n": args.n, "p": args.p, "overlap": args.overlap,
                        "noise": args.noise, "nan": args.nan,
                        "seed": seed, "hp": str(hp), "sol_f1": f1, "union_f1": uf,
                        "dissim": dis, "overlap_struct_err": err})
            f.flush()
            print(f"  {args.method} ov={args.overlap} sd={seed} hp={hp} -> "
                  f"u={uf:.3f} s={f1:.3f}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
