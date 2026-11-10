"""E3 of the special-issue comparison plan: GEMSS on the p = 10 benchmark of
Li, Golub & Lu (2026, arXiv:2609.38656), scored with their yardstick against the
exact posterior.

Their package ``amvi`` (commit b8341d8, no license -> NOT vendored) supplies the
data generators, the dataset seeds (results/datasets.json) and the exact
enumeration reference; put its ``src/`` on PYTHONPATH and pass --amvi-root.

Metrics follow their definitions (validated against their saved mean-field
records to 5 decimals): grouped-support total variation over the inclusion
patterns of the correlated block(s) (8 patterns for one triplet, 64 for two),
PIP mean absolute error, true-positive rate and false-discovery proportion at
PIP >= 0.5, and noiseless test prediction MSE of the posterior mean.

GEMSS's distribution over supports is sum_k alpha_k * delta(S_k), with S_k read
out two ways: topD (the D largest |mu_k|, GEMSS's own readout; D = their true
sparsity 5) and thr (coordinates with |mu_k| > 3 * spike s.d.). PIP_j =
sum_k alpha_k [j in S_k]; posterior mean = sum_k alpha_k mu_k.

Arms (all GEMSS 'joint', lr 0.01, 6000 steps, 16 draws, as in RQ1):
  sss      published prior: structured spike-and-slab, D = 5, slab s.d. 100, spike s.d. 0.1
  ss       their prior as close as GEMSS allows: slab weight omega, slab s.d. tau, spike s.d. 0.1
  ss01     as ss, spike s.d. 0.01
GEMSS's data term is -1/2 sum (x b - y)^2, i.e. unit noise; their sigma = 1, so the
likelihoods coincide (y is divided by sigma anyway).
"""
import argparse, csv, json, os, sys, time
from itertools import product
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemss.feature_selection.inference import BayesianFeatureSelector

from amvi import RegressionModel
from amvi.data import make_one_triplet, make_two_triplets
from amvi.exact import exact_reference


def patterns(g):
    return np.array(list(product([0, 1], repeat=g)))


def dist_from_supports(alpha, supports, cols):
    g = len(cols); out = np.zeros(2 ** g)
    for a, S in zip(alpha, supports):
        bits = [1 if c in S else 0 for c in cols]
        out[int("".join(map(str, bits)), 2)] += a
    return out


def dist_exact(ref, cols):
    sub = ref["masks"][:, cols]
    idx = sub @ (2 ** np.arange(len(cols))[::-1])
    return np.bincount(idx, weights=ref["support"], minlength=2 ** len(cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amvi-root", dest="root", required=True)
    ap.add_argument("--design", choices=["one", "two"], required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--arm", choices=["sss", "ss", "ss01"], required=True)
    ap.add_argument("--K", type=int, nargs="+", default=[3, 6, 12])
    ap.add_argument("--D", type=int, default=5)
    ap.add_argument("--n-iter", dest="n_iter", type=int, default=6000)
    ap.add_argument("--replicates", type=int, nargs="*", default=None, help="default: all 50")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfgs = json.load(open(os.path.join(args.root, "results/datasets.json")))
    sel_cfgs = [c for c in cfgs if c["p"] == 10 and abs(c["rho"] - args.rho) < 1e-9
                and ((c.get("groups") == 2) == (args.design == "two"))
                and (args.replicates is None or c["replicate"] in args.replicates)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["id", "design", "rho", "replicate", "arm", "K", "elbo", "seconds",
              "tv_topD", "pip_mae_topD", "tpr_topD", "fdp_topD",
              "tv_thr", "pip_mae_thr", "tpr_thr", "fdp_thr", "mse",
              "perplexity", "n_distinct_topD", "exact_mass_topD_supports"]
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); f.flush()
    for c in sel_cfgs:
        if args.design == "two":
            d = make_two_triplets(c)
            x, y, beta, test, cols = d["x"], d["y"], d["truth"], d["test"], sorted(int(i) for i in d["groups"].ravel())
        else:
            x, y, beta, test, block = make_one_triplet(c)
            cols = sorted(int(i) for i in block)
        ref = exact_reference(RegressionModel(x, y, c["sigma"], c["tau"], c["omega"]))
        pe = dist_exact(ref, cols)
        masks_idx = {tuple(np.flatnonzero(m)): i for i, m in enumerate(ref["masks"])}
        active = beta != 0
        if args.arm == "sss":
            kw = dict(prior="sss", var_spike=0.1)
        else:
            kw = dict(prior="ss", var_spike=0.1 if args.arm == "ss" else 0.01,
                      weight_slab=c["omega"], weight_spike=1 - c["omega"], var_slab=c["tau"])
        spike = kw["var_spike"]
        for K in args.K:
            torch.manual_seed(c["replicate"] * 1000 + K)
            t0 = time.perf_counter()
            sel = BayesianFeatureSelector(n_features=x.shape[1], n_components=K, X=x, y=y / c["sigma"],
                                          sss_sparsity=args.D, lr=0.01, batch_size=16,
                                          n_iter=args.n_iter, **kw)
            sel.optimize(regularize=False, verbose=False)
            with torch.no_grad():
                z_eval, _ = sel.mixture.sample(256)
                elbo = float(sel.elbo(z_eval).item())
            mu = sel.mixture.mu.detach().cpu().numpy()
            alpha = sel.mixture.get_alpha().detach().cpu().numpy()
            row = dict(id=c["id"], design=args.design, rho=args.rho, replicate=c["replicate"],
                       arm=args.arm, K=K, elbo=elbo, seconds=time.perf_counter() - t0,
                       mse=float(np.mean((test @ (alpha @ mu * c["sigma"] - beta)) ** 2)))
            # GEMSS fits y / sigma, so its coefficients are on the y / sigma scale (sigma = 1 here)
            for tag, sups in (("topD", [set(np.argsort(-np.abs(m))[:args.D].tolist()) for m in mu]),
                              ("thr", [set(np.flatnonzero(np.abs(m) > 3 * spike).tolist()) for m in mu])):
                pq = dist_from_supports(alpha, sups, cols)
                pip = np.array([sum(a for a, S in zip(alpha, sups) if j in S) for j in range(x.shape[1])])
                sel_ = pip >= 0.5
                row.update({f"tv_{tag}": 0.5 * np.abs(pq - pe).sum(),
                            f"pip_mae_{tag}": float(np.abs(pip - ref["pip"]).mean()),
                            f"tpr_{tag}": float(sel_[active].mean()),
                            f"fdp_{tag}": float(np.sum(sel_ & ~active) / max(1, sel_.sum()))})
                if tag == "topD":
                    distinct = {tuple(sorted(S)) for S in sups}
                    row["n_distinct_topD"] = len(distinct)
                    row["exact_mass_topD_supports"] = float(sum(ref["support"][masks_idx[s]] for s in distinct))
            a = alpha[alpha > 0]
            row["perplexity"] = float(np.exp(-(a * np.log(a)).sum()))
            w.writerow(row); f.flush()
            print(f"  {c['id']} {args.arm} K={K} tv={row['tv_topD']:.3f}/{row['tv_thr']:.3f} "
                  f"pip={row['pip_mae_topD']:.3f} mse={row['mse']:.4f}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
