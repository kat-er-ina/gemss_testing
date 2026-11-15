"""Stage-1 checks for the weight-gradient fixes (plan sec. 14), all local and label-free.

C1  closed-form log-density / gradient / Hessian diagonal used by ngvi equal autograd
    (ss and sss priors, Gaussian likelihood, mixture log q).
C2  on dumped GEMSS fits: mean of each estimator's weight update versus the exact
    ELBO gradient (current, strat, score: d ELBO / d logits; ngvi: d ELBO / d pi_c,
    its expectation-parameter form E_qc[f] - E_qK[f]).
C3  known weights: target = 0.6 / 0.3 / 0.1 mixture of N(c_m, 0.5^2 I) in d = 20.
    (a) K = 3 started at the centres, (b) K = 6 started two per centre; report the
    learned weight per centre and its TV distance to (0.6, 0.3, 0.1).
Usage: weightfix_checks.py <worktree algorithm_comparison> <results/amvi dir> [C1 C2 C3]
"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from src.hard_data_factory import generate_overlapping_dataset
from src.wrappers.weightfix_gemss import (WeightFixRunner, RegressionTarget, GaussMixTarget, logq_terms,
                                          comp_logpdf, log_alpha, make_runner)
from gemss.feature_selection.inference import BayesianFeatureSelector
from gemss.feature_selection.models import GaussianMixture

root = sys.argv[2]
which = set(sys.argv[3:]) or {"C1", "C2", "C3"}


def data(seed):
    X, y, _, _ = generate_overlapping_dataset(n_samples=100, n_features=200, n_solutions=3, sparsity=10,
        latent_rank=2, overlap=4, noise_std=0.05, nan_ratio=0.0, binarize=False, nonlinear=False,
        noise_dist="gauss", seed=seed)
    return X, y


def selector(prior, X, y, K):
    kw = dict(prior="sss", var_spike=0.1) if prior == "sss" else dict(prior="ss", var_spike=0.1,
              weight_slab=10 / 200, weight_spike=1 - 10 / 200)
    return BayesianFeatureSelector(n_features=200, n_components=K, X=X, y=y, sss_sparsity=10,
                                   lr=0.01, batch_size=16, n_iter=1, **kw)


def load_fit(sel, f):
    z = np.load(f)
    with torch.no_grad():
        sel.mixture.mu.copy_(torch.tensor(z["mu"]))
        sel.mixture._log_var.copy_(torch.log(torch.expm1(torch.tensor(z["var"]))))
        a = torch.tensor(z["alpha"]); sel.mixture._log_alpha.copy_(torch.log(a[:-1] / a[-1]))


if "C1" in which:
    print("== C1: closed form vs autograd (max relative error)")
    X, y = data(1)
    for prior in ("ss", "sss"):
        sel = selector(prior, X, y, 4); T = RegressionTarget(sel)
        torch.manual_seed(5); z = (torch.randn(3, 200) * 0.5).requires_grad_(True)
        torch.manual_seed(7); lp_ref = T.logp(z)
        g_ref = torch.autograd.grad(lp_ref.sum(), z)[0]
        torch.manual_seed(7); lp, g, H = T.terms(z.detach())
        Hd = []
        for b in range(3):
            def f(v):
                torch.manual_seed(7); return T.logp(v[None])[0]
            Hd.append(torch.diagonal(torch.autograd.functional.hessian(f, z[b].detach())))
        Hd = torch.stack(Hd)
        rel = lambda a, b_: float(((a - b_).abs().max() / b_.abs().max()).item())
        print(f"  {prior}: logp {rel(lp, lp_ref.detach()):.1e}  grad {rel(g, g_ref):.1e}  hess-diag {rel(H, Hd):.1e}")
    mix = GaussianMixture(4, 200)
    with torch.no_grad():
        mix._log_var.copy_(torch.randn(4, 200) * 0.3); mix._log_alpha.copy_(torch.randn(3))
    z = (mix.mu[0] + 0.3 * torch.randn(3, 200)).detach().requires_grad_(True)
    lq_ref = mix.log_prob(z); g_ref = torch.autograd.grad(lq_ref.sum(), z)[0]
    lq, g, H, _ = logq_terms(z.detach(), mix.mu.detach(), mix.get_variance().detach(), torch.log(mix.get_alpha()).detach())
    Hd = torch.stack([torch.diagonal(torch.autograd.functional.hessian(lambda v: mix.log_prob(v[None])[0], z[b].detach())) for b in range(3)])
    print(f"  mixture log q: logp {float((lq - lq_ref).abs().max()):.1e} (abs)  grad {float((g - g_ref).abs().max() / g_ref.abs().max()):.1e}"
          f"  hess-diag {float((H - Hd).abs().max() / Hd.abs().max()):.1e}")

if "C2" in which:
    print("\n== C2: weight updates on dumped fits, mean over N batches vs exact")
    N = 2000
    for prior, sub, seed, K in (("ss", "r2/dumps/R2_GEMSSjointSS", 1, 6), ("sss", "r2/dumps/R2_GEMSSjoint", 1, 6),
                                ("sss", "r2/dumps/R4_GEMSSjoint_r2", 2, 12)):
        X, y = data(seed); meth = "GEMSSjoint" if prior == "sss" else "GEMSSjointSS"
        f = os.path.join(root, sub, f"{meth}_r2_sd{seed}_K{K}.npz")
        print(f"  -- {prior} {sub.split('/')[-1]} seed {seed} K {K}")
        torch.manual_seed(0)
        sel = selector(prior, X, y, K); load_fit(sel, f); T = RegressionTarget(sel); mix = sel.mixture
        with torch.no_grad():                                   # exact: per-component E_qk[f]
            al = mix.get_alpha(); var = mix.get_variance(); E = torch.zeros(K)
            for k in range(K):          # 80 calls x 100 draws: the sss prior redraws its supports per call
                E[k] = torch.stack([(lambda zz: (T.logp(zz) - mix.log_prob(zz)).mean())(
                    mix.mu[k] + torch.randn(100, 200) * var[k].sqrt()) for _ in range(80)]).mean()
        exact_logit = (al * (E - (al * E).sum()))[:-1]
        exact_pi = E[:-1] - E[-1]
        for est in ("current", "strat", "score", "ngvi"):
            g = np.zeros((N, K - 1)); t0 = time.perf_counter()
            for t in range(N):
                if est == "current":
                    mix.zero_grad(); z, _ = mix.sample(16); (T.logp(z) - mix.log_prob(z)).mean().backward()
                    g[t] = mix._log_alpha.grad.numpy()
                elif est in ("strat", "score"):
                    r = WeightFixRunner(mix, T, est, sel=sel); mix.zero_grad()
                    if est == "strat":
                        n = r.strat_n; v = mix.get_variance()
                        zz = mix.mu[:, None] + torch.randn(K, n, 200) * v.sqrt()[:, None]
                        obj = (log_alpha(mix).exp() * r._h_own(zz.reshape(-1, 200)).view(K, n).mean(1)).sum()
                    else:
                        z, _ = mix.sample(16); v = mix.get_variance(); lpk = comp_logpdf(z, mix.mu, v); la = log_alpha(mix)
                        h = T.logp(z) - torch.logsumexp(lpk + la.detach()[None], -1); hd = h.detach()
                        obj = h.mean() + ((hd - (hd.sum() - hd) / 15) * torch.logsumexp(lpk.detach() + la[None], -1)).mean()
                    obj.backward(); g[t] = mix._log_alpha.grad.numpy()
                else:                                            # ngvi: E[(d_c - d_K)(f - baseline)]
                    with torch.no_grad():
                        la = torch.log(mix.get_alpha()); v = mix.get_variance()
                        z, _ = mix.sample(16); lq, _, _, rr = logq_terms(z, mix.mu, v, la)
                        fz = T.logp(z) - lq; fb = fz - (fz.sum() - fz) / 15
                        d = rr / la.exp()[None]
                        g[t] = ((d[:, :-1] - d[:, -1:]) * fb[:, None]).mean(0).numpy()
            ex = exact_pi if est == "ngvi" else exact_logit
            m, se = g.mean(0), g.std(0) / np.sqrt(N)
            z_ = (m - ex.numpy()) / np.maximum(se, 1e-12)
            print(f"    {est:8s} exact {np.round(ex.numpy(), 2)}\n             mean  {np.round(m, 2)}  "
                  f"max|z|={np.abs(z_).max():.1f}  mean SE {se.mean():.3f}  ({time.perf_counter() - t0:.0f}s)")

if "C3" in which:
    print("\n== C3: known weights 0.6/0.3/0.1, d = 20, target sd 0.5 (truth var 0.25)")
    d, sd, w = 20, 0.5, torch.tensor([0.6, 0.3, 0.1])
    cen = torch.zeros(3, d); cen[0, 0], cen[1, 1], cen[2, 2] = 4.0, 4.0, 4.0
    T = GaussMixTarget(cen, w, sd)
    for case, K, start in (("a", 3, [0, 1, 2]), ("b", 6, [0, 0, 1, 1, 2, 2])):
        for est in ("current", "frozen", "strat", "score", "ngvi", "fixedpoint"):
            torch.manual_seed(1)
            mix = GaussianMixture(K, d)
            with torch.no_grad():
                mix.mu.copy_(cen[start] + 0.3 * torch.randn(K, d))
            r = WeightFixRunner(mix, T, est, batch_size=16, lr=0.01)
            t0 = time.perf_counter(); r.run(3000)
            with torch.no_grad():
                a = mix.get_alpha(); assign = torch.cdist(mix.mu, cen).argmin(1)
                per = torch.zeros(3).index_add_(0, assign, a)
                v = mix.get_variance().mean()
            print(f"  ({case}) K={K} {est:10s} weight per centre {np.round(per.numpy(), 3)}  TV {0.5 * float((per - w).abs().sum()):.3f}"
                  f"  mean var {float(v):.3f}  ({time.perf_counter() - t0:.0f}s)")
