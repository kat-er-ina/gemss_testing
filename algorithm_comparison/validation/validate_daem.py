"""Validate the tempering principle the way it is actually published
(Ueda & Nakano 1998): deterministic-annealing EM reaches good optima MORE
RELIABLY than plain EM, averaged over random initializations, on a hard
(overlapping) mixture where plain EM frequently gets stuck in local optima.

Bug-free discipline for mechanism B (likelihood tempering / β-schedule). The
single-bad-init test is the wrong one — plain EM is robust on well-separated
data. The published benefit is init-robustness on overlapping clusters.

Setup: K=4 overlapping 1-D Gaussians at means [-3,-1,1,3] (std 1). 40 random
inits. DA-EM = staged cooling β: 0.1→1 (inner EM iters per temperature) with a
small temperature-scaled centroid perturbation. Metric: fraction of inits whose
final log-likelihood is within tol of the best found (= "success rate"), and mean
final log-likelihood.

Pass: DA-EM success rate > plain EM success rate, and DA-EM mean loglik >= EM.
"""

import numpy as np

TRUE = np.array([-3.0, -1.0, 1.0, 3.0])


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(m, 1.0, n // 4) for m in TRUE])


def _loglik(x, mu, var, w):
    comp = w[None, :] * np.exp(-0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :]) / np.sqrt(2 * np.pi * var[None, :])
    return float(np.log(comp.sum(1) + 1e-300).sum())


def _em_iters(x, mu, var, w, beta, iters, perturb=0.0, rng=None):
    for _ in range(iters):
        logN = -0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :] - 0.5 * np.log(2 * np.pi * var[None, :])
        logr = beta * (np.log(w[None, :] + 1e-300) + logN)
        logr -= logr.max(1, keepdims=True)
        r = np.exp(logr); r /= r.sum(1, keepdims=True)
        Nk = r.sum(0) + 1e-300
        w = Nk / Nk.sum()
        mu = (r * x[:, None]).sum(0) / Nk
        if perturb > 0 and rng is not None:
            mu = mu + rng.normal(0, perturb, size=mu.shape)
        var = np.maximum((r * (x[:, None] - mu[None, :]) ** 2).sum(0) / Nk, 1e-2)
    return mu, var, w


def plain_em(x, K, seed):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(x.min(), x.max(), K)
    mu, var, w = _em_iters(x, mu, np.ones(K), np.ones(K) / K, 1.0, 300)
    return _loglik(x, mu, var, w)


def da_em(x, K, seed):
    rng = np.random.default_rng(seed)
    mu, var, w = rng.uniform(x.min(), x.max(), K), np.ones(K), np.ones(K) / K
    for beta in np.geomspace(0.1, 1.0, 25):
        mu, var, w = _em_iters(x, mu, var, w, beta, 15, perturb=0.2 * (1.0 - beta), rng=rng)
    mu, var, w = _em_iters(x, mu, var, w, 1.0, 100)  # final convergence, no perturbation
    return _loglik(x, mu, var, w)


def main():
    x = _data()
    K = 4
    seeds = range(40)
    ll_em = np.array([plain_em(x, K, s) for s in seeds])
    ll_da = np.array([da_em(x, K, s + 1000) for s in seeds])
    best = max(ll_em.max(), ll_da.max())
    tol = 5.0
    succ_em = float((ll_em > best - tol).mean())
    succ_da = float((ll_da > best - tol).mean())
    print("Overlapping K=4 GMM (means -3,-1,1,3); 40 random inits each.")
    print(f"  plain EM: success_rate={succ_em:.2f}  mean_loglik={ll_em.mean():.0f}  best={ll_em.max():.0f}")
    print(f"  DA-EM:    success_rate={succ_da:.2f}  mean_loglik={ll_da.mean():.0f}  best={ll_da.max():.0f}")
    print()
    print("VERDICT: INCONCLUSIVE / negative. Two honest observations:")
    print(" 1. On every simple GMM tried, plain EM is already init-robust (success~1.0),")
    print("    so there is no local optimum for DA to escape -- the easy setups can't")
    print("    demonstrate DA's published benefit.")
    print(" 2. A faithful DA-EM needs explicit centroid-SPLITTING at phase transitions")
    print("    (Rose 1998); this tempered-EM-without-splitting collapses to the symmetric")
    print("    solution and underperforms. So the tempering benefit is NOT validated here.")
    print("=> Treat the GEMSS likelihood-tempering lever as UNVALIDATED / low-priority")
    print("   (consistent with its weak showing in mechanism_probe). The repulsion lever")
    print("   IS validated (validate_svgd.py). Validating DA would require implementing")
    print("   Rose-style annealing with splitting -- deferred unless we commit to it.")


if __name__ == "__main__":
    main()
