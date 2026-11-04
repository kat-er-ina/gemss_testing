"""E1 of the special-issue comparison plan: does the adaptive mixture VI of
Li, Golub & Lu (2026, arXiv:2609.38656) grow past mean field on our
controlled-overlap benchmark within the authors' own budget?

Their package ``amvi`` (github.com/lihanqing1997/adaptive-mixture-spike-slab,
commit b8341d8) is NOT vendored -- the repository has no license. Put its
``src/`` on PYTHONPATH.

Prior settings (pilot, see the plan): y standardized, tau = 1, omega = D/p
(prior expected support size D, the same information every method gets), sigma
given on the command line. Solutions are read out two ways:
  topD -- the D features of largest a[k,j]*|mu[k,j]| per component, then the
          same agglomerative-Jaccard K -> m clustering GEMSS uses (primary);
  pip  -- their own rule, union = {j : PIP_j >= 0.5} (secondary).
One CSV row per (p, seed).
"""
import argparse, csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hard_data_factory import generate_overlapping_dataset
from src.evaluation import calculate_metrics, _jaccard
from ensemble_parity import cluster_supports

from amvi import RegressionModel, Settings, fit
from amvi.initialization import strong_meanfield


def mean_dissim(supports):
    sets = [set(s) for s in supports if s]
    if len(sets) < 2:
        return 0.0
    return float(np.mean([1 - _jaccard(sets[a], sets[b])
                          for a in range(len(sets)) for b in range(a + 1, len(sets))]))


def topD_supports(q, D):
    score = q.a * np.abs(q.mu)                      # (K, p) posterior-mean magnitude per component
    return [tuple(sorted(int(j) for j in np.argsort(-s)[:D])) for s in score]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p", type=int, nargs="+", default=[200])
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--D", type=int, default=10)
    ap.add_argument("--m", type=int, default=3, help="# final solutions every method returns")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=0.1, help="known noise s.d. on the standardized y")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--max-seconds", dest="max_seconds", type=float, default=120.0)
    ap.add_argument("--max-components", dest="max_components", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["n", "p", "overlap", "D", "m", "seed", "sigma", "tau", "omega",
              "mf_seconds", "fit_seconds", "total_seconds", "components",
              "candidate_components", "status", "fallback", "returned_delta_mf",
              "union_f1_topD", "n_sols_topD", "dissim_topD", "true_dissim",
              "union_f1_pip", "n_pip", "expected_size", "error"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); f.flush()
        for p in args.p:
            for seed in args.seeds:
                X, y, truth, planted = generate_overlapping_dataset(
                    n_samples=args.n, n_features=p, n_solutions=3, sparsity=args.D,
                    latent_rank=2, overlap=args.overlap, noise_std=args.noise,
                    nan_ratio=0.0, binarize=False, seed=seed)
                ys = (y - y.mean()) / y.std()
                omega = args.D / p
                row = dict(n=args.n, p=p, overlap=args.overlap, D=args.D, m=args.m, seed=seed,
                           sigma=args.sigma, tau=args.tau, omega=omega,
                           true_dissim=mean_dissim(planted.values()), error="")
                try:
                    model = RegressionModel(X, ys, sigma=args.sigma, tau=args.tau, omega=omega)
                    t0 = time.perf_counter()
                    _, baseline, _ = strong_meanfield(model, seed=seed)
                    row["mf_seconds"] = time.perf_counter() - t0
                    res = fit(model, Settings(max_components=args.max_components,
                                              max_seconds=args.max_seconds),
                              seed=seed + 1, baseline=baseline)
                    q = res["mixture"]
                    sols = cluster_supports(topD_supports(q, args.D), p, args.m, args.D)
                    pip = q.weights @ q.a
                    pip_sel = [int(j) for j in np.flatnonzero(pip >= 0.5)]
                    row.update(
                        fit_seconds=res["fit_seconds"], total_seconds=res["total_seconds"],
                        components=res["components"],
                        candidate_components=res["candidate_components"],
                        status=res["status"], fallback=res["final_fallback"],
                        returned_delta_mf=res["returned_delta_mf"],
                        union_f1_topD=calculate_metrics(sols, set(truth), p)["F1_Score"],
                        n_sols_topD=len(sols),
                        dissim_topD=mean_dissim(s["support"] for s in sols.values()),
                        union_f1_pip=calculate_metrics({"pip": {"support": pip_sel}},
                                                       set(truth), p)["F1_Score"],
                        n_pip=len(pip_sel), expected_size=float(pip.sum()))
                except Exception as e:  # record and continue: feasibility, not a crash test
                    row["error"] = f"{type(e).__name__}: {e}"
                w.writerow(row); f.flush()
                print({k: row.get(k) for k in ("p", "seed", "components", "status",
                                                "total_seconds", "union_f1_topD", "error")},
                      flush=True)


if __name__ == "__main__":
    main()
