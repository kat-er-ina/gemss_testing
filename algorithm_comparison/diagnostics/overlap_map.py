"""Experiment A of plan sec. 19: where does joint optimisation pay off? Overlap map against the EXACT
posterior of GEMSS's own model (standard spike-and-slab with a Gaussian spike, unit-noise likelihood),
enumerated over all 2^p inclusion patterns (small p).

Designs (p features, x_j ~ N(0, 1), y = X b + N(0, 1)):
  M  alternatives: features 0,1,2 pairwise correlated rho, b_0 = b; independent true feature b_3 = b.
     The posterior has alternative supports (which of 0/1/2) -> room for REPULSION.
  U  one correlated mode: features 0,1 correlated rho, b_0 = b_1 = b; b_3 = b. Both included (mostly) with
     strongly dependent coefficients -> room for TILING (within-mode dependence) with few alternatives.
Arms (paired: same seed, same starting means):
  F3J  ngvi (joint)          F3Dec  ngvi_dec (independent, ceil(16/K) draws each)   F3Ens  ngvi_dec, 16 draws
  F0J  f0_joint              F0Dec  f0_dec
Weightings for the soft readout of each fit: fit (the arm's own weights), unif, post (post-hoc
a_k ~ exp(L_k), L_k = per-component ELBO by Monte Carlo), merged (post-hoc after merging near-duplicate
components: a group shares exp(mean L) equally) -> separates MASS SHARING (fixable post hoc) from coupling.
Metrics vs exact: tv (full pattern distribution), pip_mae, cov_err (pairwise inclusion covariance MAE),
corr_err (|corr(b_i, b_j)| error on the correlated pair(s)), sd_ratio (q / exact posterior s.d., correlated
features), plus leak (uniform-weight overlap between components) and exact-posterior summaries.
Annealing (F of plan sec. 19): --anneals none 0.03/4000 ... (likelihood weight omega0 -> 1).
Usage: overlap_map.py --design M --rho 0.9 --n 25 100 --reps 30 --out a.csv [--anneals none 0.03/4000]
       overlap_map.py --selftest
"""
import argparse, csv, itertools, os, sys, time
import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemss.feature_selection.inference import BayesianFeatureSelector
from src.wrappers.weightfix_gemss import make_runner, RegressionTarget, comp_logpdf

ARMS = dict(F3J=("ngvi", {}), F3Dec=("ngvi_dec", {}), F3Ens=("ngvi_dec", dict(strat_draws=16)),
            F0J=("f0_joint", {}), F0Dec=("f0_dec", {}))
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(64)
GH_W = GH_W / GH_W.sum()


def make_data(design, rho, n, rep, p=10, b=0.5):
    rng = np.random.default_rng([ord(design), int(round(rho * 1000)), n, rep])
    g = 3 if design == "M" else 2
    S = (1 - rho) * np.eye(g) + rho * np.ones((g, g))
    X = np.empty((n, p))
    X[:, :g] = rng.standard_normal((n, g)) @ np.linalg.cholesky(S).T
    X[:, g:] = rng.standard_normal((n, p - g))
    beta = np.zeros(p)
    if p > 3:
        beta[3] = b
    if design == "M":
        beta[0] = b
    else:
        beta[0] = beta[1] = b
    y = X @ beta + rng.standard_normal(n)
    return X, y, list(range(g))


def selector(X, y, K, slab, spike, w):
    return BayesianFeatureSelector(n_features=X.shape[1], n_components=K, X=X, y=y, prior="ss",
                                   var_slab=slab, var_spike=spike, weight_slab=w, weight_spike=1 - w,
                                   lr=0.01, batch_size=16, n_iter=1)


def exact_posterior(sel):
    """Enumerate all inclusion patterns of GEMSS's ss model (prior parameters are s.d., as in
    RegressionTarget._prior_terms); likelihood -1/2 ||X b - y||^2."""
    P = sel.prior
    s1, s2, w1 = float(P.var_slab), float(P.var_spike), float(P.w_slab)
    X = sel.X.double().numpy(); y = sel.y.double().numpy().ravel()
    p = X.shape[1]
    G = np.array(list(itertools.product([0, 1], repeat=p)), dtype=float)
    v = np.where(G == 1, s1 ** 2, s2 ** 2)                                     # (2^p, p)
    XtX, Xty, yty = X.T @ X, X.T @ y, y @ y
    A = XtX[None] + np.einsum("gi,ij->gij", 1 / v, np.eye(p))
    Sg = np.linalg.inv(A)
    m = Sg @ Xty
    logml = -0.5 * (yty - m @ Xty) - 0.5 * (np.log(v).sum(1) + np.linalg.slogdet(A)[1])
    lp = logml + (G * np.log(w1) + (1 - G) * np.log(1 - w1)).sum(1)
    pi = np.exp(lp - lp.max()); pi /= pi.sum()
    mean = pi @ m
    second = np.einsum("g,gij->ij", pi, Sg + np.einsum("gi,gj->gij", m, m))
    return dict(G=G, pi=pi, pip=pi @ G, mean=mean, cov=second - np.outer(mean, mean), s1=s1, s2=s2, w=w1,
                m=m, sd=np.sqrt(np.einsum("gii->gi", Sg)))


def set_start(mix, mode, ex, seed):
    """Diverse starts for the ensemble-fairness check (same start for every arm of a dataset/K).
    default  GEMSS: mu ~ N(0, 1), var = softplus(0)   (unchanged)
    oracle   component k at the conditional posterior mean / s.d. of the k-th most probable EXACT pattern
             (an upper bound on how diverse an ensemble's starts can be)
    patdiv   K distinct random supports of size 1..4 (uniform over features, no use of the posterior mass),
             each at its conditional posterior mean / s.d."""
    if mode == "default":
        return
    K, p = mix.mu.shape
    if mode == "oracle":
        idx = np.argsort(-ex["pi"])[:K]
    elif mode == "patdiv":
        rng = np.random.default_rng(20_000 + seed); code = 2 ** np.arange(p)[::-1]; seen = set()
        while len(seen) < K:
            g = np.zeros(p, int); g[rng.choice(p, size=rng.integers(1, 5), replace=False)] = 1
            seen.add(int(g @ code))
        idx = np.array(sorted(seen))          # row index in the itertools.product enumeration
    else:
        raise ValueError(mode)
    mix.mu.copy_(torch.tensor(ex["m"][idx], dtype=mix.mu.dtype))
    mix._log_var.copy_(torch.log(torch.expm1(torch.tensor(ex["sd"][idx] ** 2, dtype=mix.mu.dtype))))


def slab_resp(mu, var, w, s1, s2):
    bb = mu[..., None] + np.sqrt(var)[..., None] * GH_X
    ls = np.log(w) + norm.logpdf(bb, scale=s1); lp = np.log1p(-w) + norm.logpdf(bb, scale=s2)
    return (np.exp(ls - np.logaddexp(ls, lp)) * GH_W).sum(-1)


def pattern_dist(alpha, a, G):
    return np.einsum("k,gk->g", alpha, np.exp(G @ np.log(a.T + 1e-300) + (1 - G) @ np.log(1 - a.T + 1e-300)))


def pattern_moments(q, G):
    pip = q @ G
    co = np.einsum("g,gi,gj->ij", q, G, G) - np.outer(pip, pip)
    return pip, co


def beta_moments(alpha, mu, var):
    mean = alpha @ mu
    return mean, np.einsum("k,ki,kj->ij", alpha, mu, mu) + np.diag(alpha @ var) - np.outer(mean, mean)


def per_component_elbo(mix_mu, mix_var, target, n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    K, p = mix_mu.shape
    z = mix_mu[:, None] + torch.randn(K, n, p, generator=g, dtype=mix_mu.dtype) * mix_var.sqrt()[:, None]
    with torch.no_grad():
        lq = -0.5 * (torch.log(2 * torch.pi * mix_var)[:, None] + (z - mix_mu[:, None]) ** 2 / mix_var[:, None]).sum(-1)
        lpz = target.logp(z.reshape(K * n, p)).view(K, n)
    return (lpz - lq).mean(1).numpy()


def merged_weights(L, mu, var, thr=1.0):
    K = len(L)
    d = np.sqrt((((mu[:, None] - mu[None]) ** 2) / ((var[:, None] + var[None]) / 2)).sum(-1))
    lab = np.arange(K)
    for i in range(K):                                   # single linkage
        for j in range(i + 1, K):
            if d[i, j] < thr:
                lab[lab == lab[j]] = lab[i]
    out = np.zeros(K)
    groups = {l: np.where(lab == l)[0] for l in np.unique(lab)}
    gL = np.array([L[idx].mean() for idx in groups.values()])
    gw = np.exp(gL - gL.max()); gw /= gw.sum()
    for wgt, idx in zip(gw, groups.values()):
        out[idx] = wgt / len(idx)
    return out, len(groups)


def metrics(alpha, mu, var, ex, cols):
    a = slab_resp(mu, var, ex["w"], ex["s1"], ex["s2"])
    q = pattern_dist(alpha, a, ex["G"])
    pip, co = pattern_moments(q, ex["G"])
    _, coe = pattern_moments(ex["pi"], ex["G"])
    iu = np.triu_indices(len(pip), 1)
    bm, bc = beta_moments(alpha, mu, var)
    sd_q, sd_e = np.sqrt(np.diag(bc)), np.sqrt(np.diag(ex["cov"]))
    pairs = list(itertools.combinations(cols, 2))
    corr = lambda C, i, j: C[i, j] / np.sqrt(C[i, i] * C[j, j])
    return dict(tv=0.5 * np.abs(q - ex["pi"]).sum(), pip_mae=float(np.abs(pip - ex["pip"]).mean()),
                cov_err=float(np.abs(co[iu] - coe[iu]).mean()),
                corr_err=float(np.mean([abs(corr(bc, i, j) - corr(ex["cov"], i, j)) for i, j in pairs])),
                sd_ratio=float(np.mean(sd_q[cols] / sd_e[cols])))


def selftest():
    X, y, cols = make_data("U", 0.9, 50, 0, p=4)
    sel = selector(X, y, 3, 2.0, 0.1, 0.2)
    ex = exact_posterior(sel)
    T = RegressionTarget(sel)
    z = torch.randn(5, 4, dtype=sel.X.dtype)
    P = sel.prior; s1, s2, w = float(P.var_slab), float(P.var_spike), float(P.w_slab)
    mine = np.log(w * norm.pdf(z.numpy(), scale=s1) + (1 - w) * norm.pdf(z.numpy(), scale=s2)).sum(1)
    d = P.log_prob(z).detach().numpy() - mine
    print("prior: GEMSS log_prob minus exact mixture log density (should be constant):", np.round(d, 6))
    res = (z @ sel.X.T - sel.y.ravel()[None]).detach().numpy()
    d2 = sel.log_likelihood(z).detach().numpy() + 0.5 * (res ** 2).sum(1)
    print("likelihood: GEMSS minus -1/2||Xz-y||^2 (should be constant):", np.round(d2, 6))
    # brute-force check of the exact PIP for p = 2 by grid integration
    X2, y2, _ = make_data("U", 0.9, 30, 1, p=2)
    s2sel = selector(X2, y2, 2, 2.0, 0.1, 0.2); e2 = exact_posterior(s2sel)
    g = np.linspace(-3, 3, 1201); B1, B2 = np.meshgrid(g, g, indexing="ij")
    Bt = torch.tensor(np.stack([B1.ravel(), B2.ravel()], 1), dtype=s2sel.X.dtype)
    lp = (s2sel.prior.log_prob(Bt) + s2sel.log_likelihood(Bt)).detach().numpy()
    wts = np.exp(lp - lp.max()); wts /= wts.sum()
    r = lambda bb: w * norm.pdf(bb, scale=s1) / (w * norm.pdf(bb, scale=s1) + (1 - w) * norm.pdf(bb, scale=s2))
    pip_grid = [float((wts * r(Bt[:, j].numpy())).sum()) for j in range(2)]
    mean_grid = [float((wts * Bt[:, j].numpy()).sum()) for j in range(2)]
    print("p=2: exact PIP", np.round(e2["pip"], 4), "grid", np.round(pip_grid, 4),
          "| exact mean", np.round(e2["mean"], 4), "grid", np.round(mean_grid, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--design", choices=["M", "U"])
    ap.add_argument("--rho", type=float)
    ap.add_argument("--n", type=int, nargs="+", default=[25, 100, 400])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--rep0", type=int, default=0)
    ap.add_argument("--K", type=int, nargs="+", default=[3, 6, 12])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--anneals", nargs="+", default=["none"])
    ap.add_argument("--inits", nargs="+", default=["default"], choices=["default", "oracle", "patdiv"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--b", type=float, default=0.5)
    ap.add_argument("--slab", type=float, default=2.0)
    ap.add_argument("--spike", type=float, default=0.1)
    ap.add_argument("--w", type=float, default=0.2)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    f = open(a.out, "w", newline=""); w = None
    for n, rep, ann in itertools.product(a.n, range(a.rep0, a.rep0 + a.reps), a.anneals):
        anneal = None if ann == "none" else (float(ann.split("/")[0]), int(ann.split("/")[1]))
        X, y, cols = make_data(a.design, a.rho, n, rep, a.p, a.b)
        ex = None
        for init, arm, K in itertools.product(a.inits, a.arms, a.K):
            est, opts = ARMS[arm]
            seed = 1000 * rep + K
            torch.manual_seed(seed)
            sel = selector(X, y, K, a.slab, a.spike, a.w)
            if ex is None:
                ex = exact_posterior(sel)
                top = np.sort(ex["pi"])[::-1]
                ex_info = dict(ex_maxpat=float(top[0]), ex_n90=int(np.searchsorted(np.cumsum(top), 0.9) + 1),
                               ex_pip_group=";".join(f"{ex['pip'][c]:.3f}" for c in cols),
                               ex_corr=float(ex["cov"][0, 1] / np.sqrt(ex["cov"][0, 0] * ex["cov"][1, 1])))
            with torch.no_grad():
                set_start(sel.mixture, init, ex, seed)
            r = make_runner(sel, est, anneal=anneal, **opts)
            t0 = time.perf_counter(); r.run(a.steps); secs = time.perf_counter() - t0
            mix = sel.mixture
            mu_t, var_t = mix.mu.detach().clone(), mix.get_variance().detach().clone()
            with torch.no_grad():
                elbo = float(sel.elbo(mix.sample(256)[0]).item())
            mu, var = mu_t.double().numpy(), var_t.double().numpy()
            al_fit = mix.get_alpha().detach().double().numpy(); al_fit = al_fit / al_fit.sum()
            L = per_component_elbo(mu_t, var_t, RegressionTarget(sel), seed=rep)
            al_post = np.exp(L - L.max()); al_post /= al_post.sum()
            al_merged, ngroups = merged_weights(L, mu, var)
            lpu = comp_logpdf(torch.tensor(mu[:, None] + np.sqrt(var)[:, None] * np.random.default_rng(rep).standard_normal((K, 64, a.p))).reshape(K * 64, a.p),
                              torch.tensor(mu), torch.tensor(var)).numpy()
            own = lpu[np.arange(K * 64), np.repeat(np.arange(K), 64)]
            leak = float((1 - np.exp(own - np.logaddexp.reduce(lpu, axis=1))).mean())
            row = dict(design=a.design, rho=a.rho, n=n, rep=rep, arm=arm, K=K, anneal=ann, init=init, elbo=elbo, seconds=secs,
                       leak=leak, n_groups=ngroups, alpha_max_fit=float(al_fit.max()), **ex_info)
            for tag, al in (("fit", al_fit), ("unif", np.full(K, 1.0 / K)), ("post", al_post), ("merged", al_merged)):
                row.update({f"{k}_{tag}": v for k, v in metrics(al, mu, var, ex, cols).items()})
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader()
            w.writerow(row); f.flush()
        print(f"  {a.design} rho={a.rho} n={n} rep={rep} {ann}: done", flush=True)
    f.close()


if __name__ == "__main__":
    main()
