"""Plan sec. 50, stage 1b: the prior as a design choice on the RQ1 benchmark generator.

Data: generate_overlapping_dataset as in E1c and coupling_probe.py (n 100, p 200, 3 planted solutions of D = 10,
latent rank 2, overlap 4, noise s.d. 0.05, Gaussian), seeds 1..25; X and y as generated.
Fit: d_select.fit_arm (F3J = ngvi, K 12, RQ3 hyperparameters incl. slab 32, 1/L step rule), the prior from
--prior/--var-spike/--t-df/--t-scale exactly as in stage 1a (d_select.prior_hp). The K raw top-D supports are
clustered to m = 3 solutions (cluster_supports, as in RQ1).
Per seed: union-F1 of the 3 solutions against the union of the planted features (calculate_metrics, as in the
paper), planted_recovery = mean over the 3 planted solutions of the best Jaccard among the returned solutions,
n_planted_found = planted solutions matched at Jaccard >= 0.8, n_distinct_raw.
Usage: prior_rq1.py --prior mc|exact|ss|student [--var-spike s] [--t-df v --t-scale s] --seeds 1 .. 25 --out x.csv

Stage 1c (sec. 50.6) adds other designs; the defaults reproduce stage 1b exactly:
  --generator latent|equicorr|distinct  generate_overlapping_dataset (default), its block-equicorrelation variant, or
                                        generate_distinct_solutions (3 disjoint blocks of D, each a full solution in
                                        which every feature is needed)
  --rank r (latent), --binarize (classification: Bernoulli likelihood, as in the paper's classification RQs),
  --n, --p (p >> n). Extra metrics: n_planted_found50 (Jaccard >= 0.5), share_signal (mean share of signal
  features in the returned solutions).

Sec. 50.8 (diagnosing why planted solutions are not recovered) adds
  --noise unit|eb|oracle   GEMSS's fixed noise variance 1 (default); empirical Bayes inside the loop (started at 1);
                           or fixed at the oracle = mean OLS residual variance of the planted solutions (regression)
  --init default|planted   GEMSS's random start, or the K components started at the planted solutions (round robin,
                           OLS coefficients on the solution's features, 0 elsewhere)
  and records the raw components' match (planted_recovery_raw, n_planted_found_raw), the final noise variance and
  the run's objective under the fitted model (elbo_model, comparable across starts and noise variances).
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for q in (ROOT, HERE):
    sys.path.insert(0, q)
import d_select as ds                                   # noqa: E402
from src.hard_data_factory import (generate_overlapping_dataset, generate_overlapping_dataset_equicorr,  # noqa: E402
                                   generate_distinct_solutions)
from src.evaluation import calculate_metrics            # noqa: E402
from ensemble_parity import cluster_supports            # noqa: E402
from sklearn.linear_model import LinearRegression       # noqa: E402

D, M, K = 10, 3, 12


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def make_data(a, seed):
    if a.generator == "latent":
        return generate_overlapping_dataset(
            n_samples=a.n, n_features=a.p, n_solutions=3, sparsity=D, latent_rank=a.rank, overlap=a.overlap,
            noise_std=0.05, nan_ratio=0.0, binarize=a.binarize, nonlinear=False, noise_dist="gauss", seed=seed)
    if a.generator == "equicorr":
        return generate_overlapping_dataset_equicorr(
            n_samples=a.n, n_features=a.p, n_solutions=3, sparsity=D, overlap=a.overlap, noise_std=0.05,
            nan_ratio=0.0, binarize=a.binarize, nonlinear=False, noise_dist="gauss", seed=seed)
    return generate_distinct_solutions(n_samples=a.n, n_features=a.p, n_solutions=3, sparsity=D, noise_std=0.05,
                                       binarize=a.binarize, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["mc", "exact", "ss", "student"], default="mc")
    ap.add_argument("--var-spike", dest="var_spike", type=float, default=None)
    ap.add_argument("--t-df", dest="t_df", type=float, default=3.0)
    ap.add_argument("--t-scale", dest="t_scale", type=float, default=1.0)
    ap.add_argument("--t-eb", dest="t_eb", choices=["none", "scale", "both"], default="none")
    ap.add_argument("--t-eb-lr", dest="t_eb_lr", type=float, default=0.005)
    ap.add_argument("--t-df-max", dest="t_df_max", type=float, default=10.0)
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--generator", choices=["latent", "equicorr", "distinct"], default="latent")
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--binarize", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--noise", choices=["unit", "eb", "oracle"], default="unit")
    ap.add_argument("--init", choices=["default", "planted"], default="default")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 26)))
    ap.add_argument("--n-iter", dest="n_iter", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.var_slab = None
    torch.set_num_threads(1)
    rows = []
    for seed in a.seeds:
        X, y, truth, planted = make_data(a, seed)
        task = "classification" if a.binarize else "regression"
        P = X.shape[1]
        pl = [list(v) for v in (planted.values() if isinstance(planted, dict) else planted)]
        ols = [LinearRegression(fit_intercept=False).fit(X[:, b], y) for b in pl]
        oracle = float(np.mean([np.mean((y - m.predict(X[:, b])) ** 2) for m, b in zip(ols, pl)]))
        a.noise_opts = (None if a.noise == "unit" else dict(mode="eb", start=1.0) if a.noise == "eb"
                        else dict(mode="fixed", var=oracle))
        a.init_mu = None
        if a.init == "planted":
            mu = np.zeros((K, P))
            for k in range(K):
                mu[k, pl[k % len(pl)]] = ols[k % len(pl)].coef_
            a.init_mu = mu
        raw, _, elbo, secs = ds.fit_arm(X, y, task, "F3J", D, seed, K, M, a.n_iter, a)
        sols = cluster_supports(raw, P, M, D)
        uf = calculate_metrics(sols, set(truth), P)["F1_Score"]
        ret = [v["support"] for v in sols.values() if v.get("support")]
        best = [max((jac(s, r) for r in ret), default=0.0) for s in pl]
        best_raw = [max((jac(s, r) for r in raw), default=0.0) for s in pl]
        W = ds.LAST_WRAPPER
        nh = getattr(W, "noise_hist_", [])
        rows.append(dict(prior=a.prior, var_spike=a.var_spike or 0.1,
                         t_df=a.t_df if a.prior == "student" else None,
                         t_scale=a.t_scale if a.prior == "student" else None,
                         t_eb=a.t_eb if a.prior == "student" else None, **ds.eb_final(), overlap=a.overlap, seed=seed, K=K, m=M,
                         generator=a.generator, rank=a.rank, task=task, n=X.shape[0], p=P,
                         union_f1=uf, planted_recovery=float(np.mean(best)), n_planted_found=int(sum(b >= 0.8 for b in best)),
                         n_planted_found50=int(sum(b >= 0.5 for b in best)),
                         noise=a.noise, init=a.init, oracle_var=oracle,
                         noise_var_final=(nh[-1][1] if nh else (oracle if a.noise == "oracle" else 1.0)),
                         elbo_model=getattr(W, "elbo_model_", np.nan),
                         planted_recovery_raw=float(np.mean(best_raw)),
                         n_planted_found_raw=int(sum(b >= 0.8 for b in best_raw)),
                         share_signal=float(np.mean([len(set(r) & set(truth)) / len(r) for r in ret])) if ret else np.nan,
                         n_distinct_raw=len(set(raw)), elbo=elbo, fit_seconds=secs,
                         solutions=";".join(",".join(map(str, sorted(s))) for s in ret)))
        print(f"  {a.prior} spike {a.var_spike} t({a.t_df},{a.t_scale}) seed {seed}: union-F1 {uf:.3f} "
              f"planted {np.mean(best):.2f} found {rows[-1]['n_planted_found']} ({secs:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
