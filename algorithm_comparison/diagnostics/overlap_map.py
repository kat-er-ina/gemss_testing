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
from src.hard_data_factory import generate_overlapping_dataset

ARMS = dict(F3J=("ngvi", {}), F3Dec=("ngvi_dec", {}), F3Ens=("ngvi_dec", dict(strat_draws=16)),
            F0J=("f0_joint", {}), F0Dec=("f0_dec", {}))
# localised coupling (plan sec. 27): the joint fit with components free only on a kept set of coordinates, the rest
# shared. Kept set: the block itself ("block"; an implementation check, exact here since block and nuisance are
# orthogonal) or the top-m features by marginal score |x_j^T y| / ||x_j|| (a data-driven screen).
LOC = dict(F3Jloc_or="block", F3Jloc20=20, F3Jloc50=50)
ARMS.update({k: ("ngvi", {}) for k in LOC})


def loc_keep(X, y, g, spec):
    """Coordinates the localised joint fit keeps free (all others are shared across components)."""
    if spec == "block":
        return np.arange(g)
    score = np.abs(X.T @ y) / np.linalg.norm(X, axis=0)
    return np.sort(np.argsort(-score, kind="stable")[:min(int(spec), X.shape[1])])
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(64)
GH_W = GH_W / GH_W.sum()


def make_data(design, rho, n, rep, p=10, b=0.5, r_overlap=0, yscale=1.0):
    if design == "R":
        # mini-RQ1: the RQ1 generator at small p (3 planted supports of 3 features over a rank-2 latent,
        # feature noise 0.05, raw unscaled y without response noise), y divided by yscale (distance knob).
        # The data do not depend on yscale except through y / yscale (paired across yscale).
        X, y, _, sup = generate_overlapping_dataset(n_samples=n, n_features=p, n_solutions=3, sparsity=3,
            latent_rank=2, overlap=r_overlap, noise_std=0.05, binarize=False, standardize=True,
            seed=int(np.random.default_rng([ord("R"), r_overlap, n, rep]).integers(2 ** 31)))
        return X, y / yscale, sorted({j for s in sup.values() for j in s})
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


P_MAX = 1000


def embed(Xb, P, key):
    """Append P - g nuisance columns that are exactly orthogonal to the block columns Xb (n x g).
    GEMSS's likelihood is -1/2 ||X z - y||^2 and its ss prior is independent across coordinates, so with
    Xb^T X_nuis = 0 the posterior factorises: the block marginal is the exact posterior of the block-only
    problem for every P (checked in --selftest). Nuisance columns: N(0, 1) draws projected onto the
    orthogonal complement of span(Xb) and scaled to ||x||^2 = n (the block's per-feature scale); they are
    correlated among themselves once P - g > n - g, as in p >> n data. Nested in P (same draws, first P - g)."""
    n, g = Xb.shape
    if P <= g:
        return Xb
    U, s, _ = np.linalg.svd(Xb, full_matrices=False)
    U = U[:, s > 1e-10 * s[0]]
    N = np.random.default_rng(key).standard_normal((n, P_MAX))[:, :P - g]
    N -= U @ (U.T @ N)
    N *= np.sqrt(n) / np.linalg.norm(N, axis=0)
    return np.column_stack([Xb, N])


def embed_diag(Xb, y, P, key):
    """Pure-dimension embedding: P - g nuisance features orthogonal to the block AND to each other, each with
    ||x_j||^2 = n and x_j^T y ~ N(0, n) (as for a unit-noise response unrelated to x_j). Built from sufficient
    statistics: the (P+1) x (P+1) Gram matrix of [X, y] keeps the block's X_b^T X_b and X_b^T y exactly and is
    realised by its Cholesky factor (P + 1 rows). GEMSS's likelihood depends on (X, y) only through this Gram
    matrix (up to a constant), so the block posterior is exactly that of (Xb, y), and each nuisance coordinate
    has its own closed-form posterior (exact_nuisance_pip). No null space, no collinearity. Nested in P."""
    n, g = Xb.shape
    if P <= g:
        return Xb, y
    cN = np.random.default_rng(key).standard_normal(P_MAX)[:P - g] * np.sqrt(n)
    G = np.zeros((P + 1, P + 1))
    G[:g, :g] = Xb.T @ Xb
    G[g:P, g:P] = n * np.eye(P - g)
    c = np.concatenate([Xb.T @ y, cN])
    G[:P, P] = G[P, :P] = c
    G[P, P] = y @ y + cN @ cN / n + 1.0                    # any value making G positive definite
    M = np.linalg.cholesky(G).T                             # M^T M = G
    return M[:, :P], M[:, P]


def exact_nuisance_pip(X, y, g, w, s1, s2):
    """Exact inclusion probabilities of mutually orthogonal nuisance columns (embed_diag): coordinate j has
    likelihood exp(-n_j z^2 / 2 + c_j z), so its marginal likelihood under N(0, s^2) is
    (1 + n_j s^2)^(-1/2) exp(c_j^2 s^2 / (2 (1 + n_j s^2)))."""
    nj = (X[:, g:] ** 2).sum(0); cj = X[:, g:].T @ y
    lml = lambda s: -0.5 * np.log1p(nj * s ** 2) + 0.5 * cj ** 2 * s ** 2 / (1 + nj * s ** 2)
    a, b = np.log(w) + lml(s1), np.log1p(-w) + lml(s2)
    return np.exp(a - np.logaddexp(a, b))


def selector(X, y, K, slab, spike, w):
    return BayesianFeatureSelector(n_features=X.shape[1], n_components=K, X=X, y=y, prior="ss",
                                   var_slab=slab, var_spike=spike, weight_slab=w, weight_spike=1 - w,
                                   lr=0.01, batch_size=16, n_iter=1)


def exact_posterior(sel, g=None):
    """Enumerate all inclusion patterns of GEMSS's ss model (prior parameters are s.d., as in
    RegressionTarget._prior_terms); likelihood -1/2 ||X b - y||^2. With g, only the first g (block)
    columns: the exact block marginal when the remaining columns are orthogonal to them (embed)."""
    P = sel.prior
    s1, s2, w1 = float(P.var_slab), float(P.var_spike), float(P.w_slab)
    X = sel.X.double().numpy(); y = sel.y.double().numpy().ravel()
    if g is not None:
        X = X[:, :g]
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


def mode_geometry(ex, min_mass=0.01, top=10):
    """How far apart are the exact posterior's modes? Patterns with mass >= min_mass (at most top), each
    summarised by its conditional posterior mean and diagonal s.d.; returns the mass-weighted mean
    Mahalanobis distance to the nearest other retained pattern (the merge metric of merged_weights), the
    number retained, and their total mass. Large = separated sharp modes; small = connected posterior."""
    idx = np.argsort(-ex["pi"])[:top]
    idx = idx[ex["pi"][idx] >= min_mass]
    if len(idx) < 2:
        return dict(ex_dnn=np.nan, ex_nmodes=len(idx), ex_mmass=float(ex["pi"][idx].sum()))
    m, v = ex["m"][idx], ex["sd"][idx] ** 2
    d = np.sqrt((((m[:, None] - m[None]) ** 2) / ((v[:, None] + v[None]) / 2)).sum(-1))
    np.fill_diagonal(d, np.inf)
    pi = ex["pi"][idx]
    return dict(ex_dnn=float((pi * d.min(1)).sum() / pi.sum()), ex_nmodes=len(idx), ex_mmass=float(pi.sum()))


def dimension_stats(alpha, mu, var, ex, g):
    """Mechanism of a possible dimension effect. d2_block / d2_nuis: mean over component pairs of the squared
    Mahalanobis distance contributed by block vs nuisance coordinates (do components separate along the many
    nuisance directions?); n_groups_block: distinct components judged on block coordinates only;
    nuis_pip: q's mean inclusion probability on nuisance coordinates (mass spent on nuisance)."""
    K = mu.shape[0]
    iu = np.triu_indices(K, 1)
    dd = ((mu[:, None] - mu[None]) ** 2) / ((var[:, None] + var[None]) / 2)       # K x K x P
    out = dict(d2_block=float(dd[..., :g].sum(-1)[iu].mean()) if K > 1 else 0.0,
               d2_nuis=float(dd[..., g:].sum(-1)[iu].mean()) if (K > 1 and mu.shape[1] > g) else 0.0)
    _, out["n_groups_block"] = merged_weights(np.zeros(K), mu[:, :g], var[:, :g])
    out["nuis_pip"] = out["nuis_pip_mae"] = np.nan
    if mu.shape[1] > g and not ex.get("nuis_gauss"):
        a = slab_resp(mu[:, g:], var[:, g:], ex["w"], ex["s1"], ex["s2"])
        out["nuis_pip"] = float((alpha @ a).mean())
        if ex.get("nuis_pip") is not None:
            out["nuis_pip_mae"] = float(np.abs(alpha @ a - ex["nuis_pip"]).mean())
    return out


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
    g = ex["m"].shape[1]                  # block size; nuisance coordinates (if embedded) start identical
    if mode == "oracle":
        idx = np.argsort(-ex["pi"])[:K]
    elif mode == "patdiv":
        rng = np.random.default_rng(20_000 + seed); code = 2 ** np.arange(g)[::-1]; seen = set()
        while len(seen) < K:
            gam = np.zeros(g, int); gam[rng.choice(g, size=rng.integers(1, 5), replace=False)] = 1
            seen.add(int(gam @ code))
        idx = np.array(sorted(seen))          # row index in the itertools.product enumeration
    else:
        raise ValueError(mode)
    mu = np.zeros((K, p)); sd = np.full((K, p), ex["s2"])     # at 0 with spike s.d. in every component
    mu[:, :g], sd[:, :g] = ex["m"][idx], ex["sd"][idx]
    mix.mu.copy_(torch.tensor(mu, dtype=mix.mu.dtype))
    mix._log_var.copy_(torch.log(torch.expm1(torch.tensor(sd ** 2, dtype=mix.mu.dtype))))


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
    # embedding: full enumeration over block + orthogonal nuisance, marginalised to the block, must equal the
    # block-only posterior (factorisation), for fixed w
    for design, kw in (("U", {}), ("R", dict(r_overlap=0, yscale=10.0))):
        pb = 4 if design == "U" else 9
        Xb, yb, _ = make_data(design, 0.9, 30, 2, p=pb, **kw)
        Xe = embed(Xb, pb + 3, [1, 2, 3])
        slab = 2.0 if design == "U" else 100.0
        full = exact_posterior(selector(Xe, yb, 2, slab, 0.1, 0.2))
        blk = exact_posterior(selector(Xb, yb, 2, slab, 0.1, 0.2))
        blk_e = exact_posterior(selector(Xe, yb, 2, slab, 0.1, 0.2), g=pb)
        code = full["G"][:, :pb] @ (2 ** np.arange(pb)[::-1])           # block pattern of every full pattern
        marg = np.bincount(code.astype(int), weights=full["pi"], minlength=2 ** pb)
        print(f"embedding ({design}, block {pb} + 3 nuisance): max|X_b^T X_n|/n = "
              f"{np.abs(Xe[:, :pb].T @ Xe[:, pb:]).max() / 30:.1e}; max |full marginal - block-only| = "
              f"{np.abs(marg - blk['pi']).max():.1e}; max |block(g) - block-only| = {np.abs(blk_e['pi'] - blk['pi']).max():.1e}")
        Xd, yd = embed_diag(Xb, yb, pb + 3, [4, 5, 6])
        full = exact_posterior(selector(Xd, yd, 2, slab, 0.1, 0.2))
        marg = np.bincount((full["G"][:, :pb] @ (2 ** np.arange(pb)[::-1])).astype(int), weights=full["pi"], minlength=2 ** pb)
        npip = exact_nuisance_pip(Xd, yd, pb, 0.2, slab, 0.1)
        print(f"embed_diag ({design}): Gram of block / X^T y preserved: {np.abs(Xd[:, :pb].T @ Xd[:, :pb] - Xb.T @ Xb).max():.1e} / "
              f"{np.abs(Xd[:, :pb].T @ yd - Xb.T @ yb).max():.1e}; max |full marginal - block-only| = {np.abs(marg - blk['pi']).max():.1e}; "
              f"nuisance PIP full enumeration vs closed form: {np.abs(full['pip'][pb:] - npip).max():.1e}")
    # per-coordinate prior weights (plan sec. 24, --nuis-prior gauss): a weight vector equal to the scalar must
    # reproduce the scalar path exactly (log density and the ngvi closed-form terms); slab weight 0 on a
    # coordinate must give GEMSS's pure-spike Gaussian there
    Xd, yd = embed_diag(*make_data("M", 0.9, 25, 0, 10)[:2], 14, [7, 8, 9])
    sa, sb = selector(Xd, yd, 3, 2.0, 0.1, 0.2), selector(Xd, yd, 3, 2.0, 0.1, 0.2)
    wv = torch.full((14,), 0.2, dtype=torch.float32); sb.prior.w_slab, sb.prior.w_spike = wv, 1.0 - wv
    z = torch.randn(7, 14, dtype=sa.X.dtype)
    ta, tb = RegressionTarget(sa).terms(z), RegressionTarget(sb).terms(z)
    e_same = max(float((u - v).abs().max()) for u, v in zip(ta, tb))
    e_lp = float((sa.prior.log_prob(z) - sb.prior.log_prob(z)).abs().max())
    wv = torch.tensor([0.2] * 10 + [0.0] * 4, dtype=torch.float32); sb.prior.w_slab, sb.prior.w_spike = wv, 1.0 - wv
    s2v = float(sb.prior.var_spike)
    ref = -0.5 * (z[:, 10:] / s2v) ** 2 - np.log(s2v)                         # GEMSS normalisation (no 2 pi)
    blk = torch.log(0.2 * torch.exp(-0.5 * (z[:, :10] / 2.0) ** 2) / 2.0 + 0.8 * torch.exp(-0.5 * (z[:, :10] / s2v) ** 2) / s2v)
    want = blk.sum(-1) + ref.sum(-1)
    Tb = RegressionTarget(sb)
    lp_t, g_t, H_t = Tb._prior_terms(z)
    e_g = float((Tb.logp(z) - sb.log_likelihood(z) - want).abs().max() / want.abs().max())   # stable form, any z
    zs = z.clone(); zs[:, 10:] *= 0.1                     # GEMSS's raw form agrees where it does not underflow
    want_s = blk.sum(-1) + (-0.5 * (zs[:, 10:] / s2v) ** 2 - np.log(s2v)).sum(-1)
    e_raw = float((sb.prior.log_prob(zs) - want_s).abs().max())
    e_t = float((lp_t - want).abs().max() / want.abs().max())
    print(f"   raw GEMSS log_prob at realistic nuisance values (|z| ~ 0.1): {e_raw:.1e}")
    assert e_raw < 1e-4
    e_grad = float((g_t[:, 10:] + z[:, 10:] / s2v ** 2).abs().max()); e_H = float((H_t[:, 10:] + 1 / s2v ** 2).abs().max())
    print(f"per-coordinate weights: vector == scalar (terms / log_prob) {e_same:.1e} / {e_lp:.1e}; nuisance slab weight 0 "
          f"-> Gaussian log density (target, rel.) {e_g:.1e}, ngvi terms rel. {e_t:.1e}, gradient {e_grad:.1e}, curvature {e_H:.1e}")
    assert e_same == 0.0 and e_lp == 0.0 and e_g < 1e-6 and e_t < 1e-6 and e_grad < 1e-3 and e_H < 1e-3
    # localised coupling (plan sec. 27)
    Xd, yd = embed_diag(*make_data("M", 0.9, 25, 0, 10)[:2], 40, [7, 8, 9])
    tm = torch.zeros(40, dtype=torch.bool); tm[10:] = True
    runs = []
    for tie in (None, torch.zeros(40, dtype=torch.bool), tm):            # no tie, empty tie, nuisance tied
        torch.manual_seed(5); sl = selector(Xd, yd, 4, 2.0, 0.1, 0.2)
        r = make_runner(sl, "ngvi", tie=tie); r.run(300); runs.append(r)
    e_empty = float(max((runs[0].mu - runs[1].mu).abs().max(), (runs[0].prec - runs[1].prec).abs().max()))
    e_tied = float(max((runs[2].mu[:, 10:] - runs[2].mu[:1, 10:]).abs().max(),
                       (runs[2].prec[:, 10:] - runs[2].prec[:1, 10:]).abs().max()))
    d_blk = float((runs[2].mu[:, :10] - runs[2].mu[:1, :10]).abs().max())
    # the shared factor's update direction is the gradient of the stratified ELBO estimate w.r.t. the shared mean
    T = RegressionTarget(selector(Xd, yd, 3, 2.0, 0.1, 0.2)); dt = T.sel.X.dtype
    K, n = 3, 6
    gen = torch.Generator().manual_seed(11)
    mu_b = torch.randn(K, 40, generator=gen, dtype=dt); mu0 = torch.randn(40, generator=gen, dtype=dt).requires_grad_(True)
    sd = torch.rand(K, 40, generator=gen, dtype=dt) + 0.3; sd[:, tm] = sd[0, tm]
    eps = torch.randn(K, n, 40, generator=gen, dtype=dt); eps[:, :, tm] = eps[:1, :, tm]
    la = torch.log_softmax(torch.randn(K, generator=gen, dtype=dt), 0)
    mu = torch.where(tm, mu0.expand(K, 40), mu_b)
    zf = (mu[:, None] + eps * sd[:, None]).reshape(K * n, 40)
    lq = torch.logsumexp(comp_logpdf(zf, mu.detach(), sd ** 2) + la[None], -1)      # pathwise part only
    ((la.exp() * (T.logp(zf) - lq).view(K, n).mean(1)).sum()).backward()
    from src.wrappers.weightfix_gemss import logq_terms
    with torch.no_grad():
        _, gq, _, _ = logq_terms(zf, mu, sd ** 2, la); _, gp, _ = T.terms(zf)
        gw = (la.exp()[:, None] * (gq - gp).view(K, n, 40).mean(1)).sum(0)
    e_dir = float((mu0.grad[tm] + gw[tm]).abs().max() / gw[tm].abs().max())
    keep = loc_keep(Xd, yd, 10, 20)
    print(f"localised coupling: empty tie == no tie {e_empty:.1e}; tied coordinates identical across components "
          f"{e_tied:.1e} (block spread {d_blk:.2f}); shared-factor direction vs autograd (rel.) {e_dir:.1e}; "
          f"screen m=20 keeps {len(keep)} incl. {int((keep < 10).sum())} block")
    assert e_empty == 0.0 and e_tied == 0.0 and d_blk > 0.05 and e_dir < 1e-4 and len(keep) == 20
    # spike-width path (plan sec. 28): schedule end points, prior restored to the target spike, and the path is
    # the only difference (no path == plain run, bit for bit)
    outs = []
    for sp in (None, (None, 0), (0.1, 200)):
        torch.manual_seed(3); sl = selector(Xd, yd, 3, 2.0, 0.01, 0.2)
        r = make_runner(sl, "ngvi", spike_path=sp)
        s_first = r.spike_at(0) if r.spike_path else None
        r.run(300); outs.append((r, s_first, float(sl.prior.var_spike)))
    e_none = float((outs[0][0].mu - outs[1][0].mu).abs().max())
    r2, s_first, s_after = outs[2]
    print(f"spike path: no path == plain {e_none:.1e}; s(0) = {s_first:.3f}, s(T/2) = {r2.spike_at(100):.4f}, "
          f"after run {s_after:.3f}; path changes the fit: {float((outs[0][0].mu - r2.mu).abs().max()):.2f}")
    # Bernoulli target (plan sec. 30): closed-form gradient and Hessian diagonal of log prior + Bernoulli
    # log-likelihood vs autograd of the selector's own log density; make_runner picks it for the logistic selector
    from src.wrappers.logistic_gemss import LogisticBayesianFeatureSelector
    from src.wrappers.weightfix_gemss import LogisticTarget
    rng = np.random.default_rng(4); Xc = rng.standard_normal((40, 6)); yc = (Xc[:, 0] - Xc[:, 1] + rng.standard_normal(40) > 0) * 1.0
    for pr in ("ss", "sss"):
        kw = dict(prior=pr, var_slab=32.0, var_spike=0.1, weight_slab=0.3 if pr == "ss" else 0.9,
                  weight_spike=0.7 if pr == "ss" else 0.1, lr=0.01, batch_size=16, n_iter=1)
        torch.manual_seed(0); lsel = LogisticBayesianFeatureSelector(n_features=6, n_components=2, X=Xc, y=yc, sss_sparsity=2, **kw)
        Tl = make_runner(lsel, "ngvi").target
        zt = (0.5 * torch.randn(3, 6, dtype=lsel.X.dtype)).requires_grad_(True)
        lp, g, H = Tl.terms(zt.detach())
        if pr == "sss":                                   # the sss prior term is a fresh Monte Carlo draw per call:
            P_ = lsel.prior; lsel.prior = None            # compare the likelihood part only
            ll = lsel.log_likelihood(zt)
            lsel.prior = P_
            lp0, g0, H0 = Tl._prior_terms(zt.detach()); lp, g, H = lp - lp0, g - g0, H - H0
        else:
            ll = Tl.logp(zt)
        ga = torch.autograd.grad(ll.sum(), zt, create_graph=True)[0]
        Ha = torch.stack([torch.autograd.grad(ga[:, j].sum(), zt, retain_graph=True)[0][:, j] for j in range(6)], 1)
        e = max(float((lp - ll).abs().max() / ll.abs().max()), float((g - ga).abs().max() / ga.abs().max()),
                float((H - Ha).abs().max() / Ha.abs().max()))
        print(f"Bernoulli target ({pr}): {type(Tl).__name__}; log density / gradient / Hessian diagonal vs autograd, max rel. {e:.1e}")
        assert isinstance(Tl, LogisticTarget) and e < 1e-5
    assert e_none == 0.0 and abs(s_first - 0.1) < 1e-12 and abs(r2.spike_at(100) - 0.1 ** 0.5 * 0.01 ** 0.5) < 1e-12 \
        and s_after == 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--design", choices=["M", "U", "R"])
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--n", type=int, nargs="+", default=[25, 100, 400])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--rep0", type=int, default=0)
    ap.add_argument("--K", type=int, nargs="+", default=[3, 6, 12])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--anneals", nargs="+", default=["none"])
    ap.add_argument("--spike-paths", dest="spike_paths", nargs="+", default=["none"],
                    help="plan sec. 28: 'none' or s0/T (spike s.d. from s0 to --spike, geometric over T steps)")
    ap.add_argument("--inits", nargs="+", default=["default"], choices=["default", "oracle", "patdiv"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--p", type=int, default=10, help="block size (features of the exact problem)")
    ap.add_argument("--P", type=int, nargs="+", default=[0],
                    help="total dimensions after orthogonal nuisance embedding (0 or <= p: no embedding)")
    ap.add_argument("--nuis", default="diag", choices=["diag", "coll"],
                    help="diag: mutually orthogonal nuisance (pure dimension, exact nuisance PIP); "
                         "coll: random columns in the complement of the block (collinear once P > n)")
    ap.add_argument("--wmodes", nargs="+", default=["fixed"], choices=["fixed", "scaled"],
                    help="prior slab weight: fixed = --w; scaled = --w * p / P (as D/p in RQ1, 5/p in Li et al.)")
    ap.add_argument("--nuis-prior", dest="nuis_prior", default="ss", choices=["ss", "gauss"],
                    help="ss: nuisance coordinates get the model's spike-and-slab prior (as the block); gauss "
                         "(plan sec. 24 control): slab weight 0 on nuisance coordinates, i.e. the spike alone, a "
                         "Gaussian prior, so their exact posterior is Gaussian; the block's prior is unchanged")
    ap.add_argument("--yscales", type=float, nargs="+", default=[1.0], help="design R: y divided by this")
    ap.add_argument("--r-overlap", dest="r_overlap", type=int, default=0, help="design R: shared core size")
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
    for n, rep, ann, ys, sp in itertools.product(a.n, range(a.rep0, a.rep0 + a.reps), a.anneals, a.yscales,
                                                 a.spike_paths):
        anneal = None if ann == "none" else (float(ann.split("/")[0]), int(ann.split("/")[1]))
        spath = None if sp == "none" else (float(sp.split("/")[0]), int(sp.split("/")[1]))
        Xb, y, cols = make_data(a.design, a.rho, n, rep, a.p, a.b, a.r_overlap, ys)
        g = Xb.shape[1]
        for P in a.P:
            key = [ord(a.design), int(round(a.rho * 1000)), a.r_overlap, n, rep, 77]
            if a.nuis == "diag":
                X, yy = embed_diag(Xb, y, P, key)
            else:
                X, yy = embed(Xb, P, key), y
            Pe = X.shape[1]
            for wmode in a.wmodes:
                if wmode == "scaled" and Pe == g and "fixed" in a.wmodes:
                    continue                                   # identical to fixed at P = p
                wt = a.w if wmode == "fixed" else a.w * g / Pe
                ex = None
                for init, arm, K in itertools.product(a.inits, a.arms, a.K):
                    est, opts = ARMS[arm]
                    seed = 1000 * rep + K
                    torch.manual_seed(seed)
                    sel = selector(X, yy, K, a.slab, a.spike, wt)
                    if ex is None:
                        ex = exact_posterior(sel, g)
                        ex["nuis_pip"] = (exact_nuisance_pip(sel.X.double().numpy(), sel.y.double().numpy().ravel(), g,
                                          wt, ex["s1"], ex["s2"]) if (a.nuis == "diag" and Pe > g) else None)
                        top = np.sort(ex["pi"])[::-1]
                        Xs = sel.X.double().numpy()
                        ex_info = dict(ex_maxpat=float(top[0]), ex_n90=int(np.searchsorted(np.cumsum(top), 0.9) + 1),
                                       ex_pip_group=";".join(f"{ex['pip'][c]:.3f}" for c in cols),
                                       ex_corr=float(ex["cov"][cols[0], cols[1]] / np.sqrt(ex["cov"][cols[0], cols[0]] * ex["cov"][cols[1], cols[1]])),
                                       xcross=float(np.abs(Xs[:, :g].T @ Xs[:, g:]).max() / n) if Pe > g else 0.0,
                                       **mode_geometry(ex))
                        q = ex["nuis_pip"]                 # exact nuisance inclusion entropy (nats) and mean PIP
                        ex_info.update(ex_nuis_ent=float(-(q * np.log(q + 1e-300) + (1 - q) * np.log1p(-q + 1e-300)).sum())
                                       if q is not None else np.nan, ex_nuis_pip=float(q.mean()) if q is not None else np.nan)
                    if a.nuis_prior == "gauss" and Pe > g:
                        if not ex.get("nuis_gauss"):            # nuisance posterior is Gaussian: no inclusion
                            ex.update(nuis_gauss=True, nuis_pip=None)
                            ex_info.update(ex_nuis_ent=0.0, ex_nuis_pip=0.0)
                        wv = torch.tensor([wt] * g + [0.0] * (Pe - g), dtype=torch.float32)
                        sel.prior.w_slab, sel.prior.w_spike = wv, 1.0 - wv
                    with torch.no_grad():
                        set_start(sel.mixture, init, ex, seed)
                    tie, scr = None, dict(scr_m=np.nan, scr_block=np.nan, scr_cols=np.nan, scr_rel=np.nan, scr_pip10=np.nan)
                    if arm in LOC:
                        keep = loc_keep(sel.X.double().numpy(), sel.y.double().numpy().ravel(), g, LOC[arm])
                        tm = np.ones(Pe, bool); tm[keep] = False; tie = torch.tensor(tm)
                        big = np.flatnonzero(ex["pip"][:g] > 0.1)      # block features with exact PIP > 0.1
                        rel = sorted(set(cols) | ({3} if a.design in ("M", "U") and g > 3 else set()))   # signal features
                        scr = dict(scr_m=len(keep), scr_block=int((keep < g).sum()), scr_cols=int(np.isin(cols, keep).sum()),
                                   scr_rel=float(np.isin(rel, keep).mean()),
                                   scr_pip10=float(np.isin(big, keep).mean()) if len(big) else np.nan)
                    r = make_runner(sel, est, anneal=anneal, tie=tie, spike_path=spath, **opts)
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
                    lpu = comp_logpdf(torch.tensor(mu[:, None] + np.sqrt(var)[:, None] * np.random.default_rng(rep).standard_normal((K, 64, Pe))).reshape(K * 64, Pe),
                                      torch.tensor(mu), torch.tensor(var)).numpy()
                    own = lpu[np.arange(K * 64), np.repeat(np.arange(K), 64)]
                    leak = float((1 - np.exp(own - np.logaddexp.reduce(lpu, axis=1))).mean())
                    row = dict(design=a.design, rho=a.rho, n=n, rep=rep, arm=arm, K=K, anneal=ann, init=init,
                               spike=a.spike, spike_path=sp,
                               yscale=ys, r_overlap=a.r_overlap, P=Pe, g=g, nuis=a.nuis, wmode=wmode, w=wt,
                               nuis_prior=a.nuis_prior,
                               slab=a.slab, n_rows=X.shape[0],
                               elbo=elbo, seconds=secs, leak=leak, n_groups=ngroups, alpha_max_fit=float(al_fit.max()),
                               **scr, **ex_info, **dimension_stats(al_post, mu, var, ex, g))
                    for tag, al in (("fit", al_fit), ("unif", np.full(K, 1.0 / K)), ("post", al_post), ("merged", al_merged)):
                        row.update({f"{k}_{tag}": v for k, v in metrics(al, mu[:, :g], var[:, :g], ex, cols).items()})
                    if w is None:
                        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader()
                    w.writerow(row); f.flush()
        print(f"  {a.design} rho={a.rho} n={n} rep={rep} {ann} yscale={ys}: done", flush=True)
    f.close()


if __name__ == "__main__":
    main()
