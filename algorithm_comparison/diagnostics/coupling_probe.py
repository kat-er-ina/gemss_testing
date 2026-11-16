"""Stage A of the coupling test (plan sec. 18): does the entropy coupling act, and when?

For each (optimiser, prior, seed, K, overlap, start) the joint fit J and the decoupled fit Dec are run
from the same seed (same starting means, same random numbers); Dec differs only in log q (own component).
At fine checkpoints, on J's state:
  leak        mean over components of 1 - own responsibility for own draws (actual weights)
  leak_frac   share of components with leak > 1 %
  gshare      ||d/dmu_c of the coupling term sum_k a_k E_qk[log r_k]|| / ||d/dmu_c of J's ELBO||,
              mean and max over components (64 reparameterised draws per component)
  maha_min    smallest pairwise Mahalanobis distance between component means
and J vs Dec: max relative |mu| difference, share of components with the same top-D set,
plus union-F1 (top-D -> cluster K -> m) and #distinct top-D sets of each.
Optimisers: f0 = (f0_joint, f0_dec), f3 = (ngvi, ngvi_dec).
Usage: coupling_probe.py --out a.csv [--opts f0 f3] [--priors sss ss] [--seeds 1 2 3 4 5]
       [--K 6 12 24] [--overlaps 0 8] [--inits default] [--workers 8]
"""
import argparse, csv, itertools, os, sys, time
import numpy as np
import torch
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.hard_data_factory import generate_overlapping_dataset
from src.evaluation import calculate_metrics
from ensemble_parity import cluster_supports
from gemss.feature_selection.inference import BayesianFeatureSelector
from src.wrappers.weightfix_gemss import make_runner, apply_init, comp_logpdf, RegressionTarget

CPS = [0, 10, 25, 50, 100, 200, 400, 700, 1000, 1500, 2000, 4000, 6000]
ARMS = dict(f0=("f0_joint", "f0_dec"), f3=("ngvi", "ngvi_dec"))
D, P, N, M = 10, 200, 100, 3


def selector(prior, X, y, K):
    kw = dict(prior="sss", var_spike=0.1) if prior == "sss" else dict(
        prior="ss", var_spike=0.1, weight_slab=D / P, weight_spike=1 - D / P)
    return BayesianFeatureSelector(n_features=P, n_components=K, X=X, y=y, sss_sparsity=D,
                                   lr=0.01, batch_size=16, n_iter=1, **kw)


def run_arm(est, prior, X, y, K, seed, init):
    torch.manual_seed(seed)
    sel = selector(prior, X, y, K)
    apply_init(sel.mixture, init, X, y, D, seed)
    r = make_runner(sel, est)
    snaps, done = {}, 0
    for cp in CPS:
        if cp > done:
            r.run(cp - done); done = cp
        mix = sel.mixture
        snaps[cp] = (mix.mu.detach().clone(), mix.get_variance().detach().clone(), mix.get_alpha().detach().clone())
    return sel, snaps


def coupling_stats(mu, var, alpha, target, n=64, seed=0):
    K, p = mu.shape
    g = torch.Generator().manual_seed(seed)
    mu_t = mu.clone().requires_grad_(True)
    z = (mu_t[:, None, :] + torch.randn(K, n, p, generator=g) * var.sqrt()[:, None, :]).reshape(K * n, p)
    lpk = comp_logpdf(z, mu_t, var) + torch.log(alpha)[None]
    lq = torch.logsumexp(lpk, -1)
    c = torch.arange(K).repeat_interleave(n)
    logr = lpk[torch.arange(K * n), c] - lq
    wts = alpha[c] / n
    coup = (wts * logr).sum()
    elbo = (wts * (target.logp(z) - lq)).sum()
    gc = torch.autograd.grad(coup, mu_t, retain_graph=True)[0]
    ge = torch.autograd.grad(elbo, mu_t)[0]
    share = (gc.norm(dim=1) / ge.norm(dim=1).clamp_min(1e-30)).numpy()
    leak = (1 - logr.detach().exp()).view(K, n).mean(1).numpy()
    d2 = (((mu[:, None] - mu[None]) ** 2) / ((var[:, None] + var[None]) / 2)).sum(-1)
    iu = torch.triu_indices(K, K, 1)
    return dict(leak=float(leak.mean()), leak_frac=float((leak > 0.01).mean()), gshare=float(share.mean()),
                gshare_max=float(share.max()), maha_min=float(d2[iu[0], iu[1]].min().sqrt()))


def readout(mu, truth, planted):
    raw = [tuple(sorted(np.argsort(np.abs(m))[::-1][:D].tolist())) for m in mu.numpy()]
    sols = cluster_supports(raw, P, M, D)
    return calculate_metrics(sols, truth, P)["F1_Score"], len(set(raw))


def task(args):
    opt, prior, seed, K, ov, init = args
    torch.set_num_threads(1)
    X, y, truth, planted = generate_overlapping_dataset(
        n_samples=N, n_features=P, n_solutions=3, sparsity=D, latent_rank=2, overlap=ov, noise_std=0.05,
        nan_ratio=0.0, binarize=False, nonlinear=False, noise_dist="gauss", seed=seed)
    truth = set(truth)
    t0 = time.perf_counter()
    selJ, sJ = run_arm(ARMS[opt][0], prior, X, y, K, seed, init)
    _, sD = run_arm(ARMS[opt][1], prior, X, y, K, seed, init)
    target = RegressionTarget(selJ)
    rows = []
    for cp in CPS:
        muJ, varJ, aJ = sJ[cp]; muD = sD[cp][0]
        st = coupling_stats(muJ, varJ, aJ, target, seed=cp)
        same = np.mean([set(np.argsort(-np.abs(a))[:D].tolist()) == set(np.argsort(-np.abs(b))[:D].tolist())
                        for a, b in zip(muJ.numpy(), muD.numpy())])
        ufJ, ndJ = readout(muJ, truth, planted); ufD, ndD = readout(muD, truth, planted)
        rows.append(dict(opt=opt, prior=prior, seed=seed, K=K, overlap=ov, init=init, step=cp, **st,
                         rel_diff=float((muJ - muD).abs().max() / muJ.abs().max().clamp_min(1e-12)),
                         same_topD=float(same), uf_J=ufJ, uf_Dec=ufD, nd_J=ndJ, nd_Dec=ndD,
                         var_med=float(varJ.median()), alpha_max=float(aJ.max())))
    print(f"  {opt} {prior} sd{seed} K{K} ov{ov} {init}: leak@100 {rows[4]['leak']:.3f} @6k {rows[-1]['leak']:.3f} "
          f"rel_diff@6k {rows[-1]['rel_diff']:.1e} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--opts", nargs="+", default=["f0", "f3"])
    ap.add_argument("--priors", nargs="+", default=["sss", "ss"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--K", type=int, nargs="+", default=[6, 12, 24])
    ap.add_argument("--overlaps", type=int, nargs="+", default=[0, 8])
    ap.add_argument("--inits", nargs="+", default=["default"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = list(itertools.product(a.opts, a.priors, a.seeds, a.K, a.overlaps, a.inits))
    jobs.sort(key=lambda j: (-j[3], j[0]))                  # big K first
    with Pool(a.workers) as pool, open(a.out, "w", newline="") as f:
        w = None
        for rows in pool.imap_unordered(task, jobs):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
            w.writerows(rows); f.flush()


if __name__ == "__main__":
    main()
