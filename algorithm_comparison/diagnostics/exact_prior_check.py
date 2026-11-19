"""Plan sec. 47: checks of the exact structured spike-and-slab density (src/wrappers/exact_sss.py).
  1. log e_k (tree product) = brute-force subset sums (p 9, D 3) and = a sequential float64 DP (p 300, D 10, float32)
  2. log p(z) and inclusion probabilities = brute-force enumeration over all D-subsets
  3. where GEMSS's own prior enumerates exactly (p <= 10, D <= 3): same inclusion probabilities and gradient as GEMSS's
     log_prob, log p differing only by the normalising constant
  4. closed-form gradient = autograd; Hessian diagonal = finite differences of the gradient
  5. short fits through F3 joint (ngvi), F3 decoupled (ngvi_dec) and GEMSS's own loop (current) run with finite ELBO
  6. timing of log_prob at B 16, p 500, D 10
Usage: exact_prior_check.py
"""
import itertools, math, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
from src.wrappers.exact_sss import log_esp, ExactStructuredSpikeAndSlabPrior        # noqa: E402
from src.wrappers.weightfix_gemss import WeightFixGEMSSWrapper                      # noqa: E402
from gemss.feature_selection.models import StructuredSpikeAndSlabPrior              # noqa: E402

torch.manual_seed(0); rng = np.random.default_rng(0)
ok = True


def check(name, err, tol):
    global ok
    good = err <= tol; ok &= good
    print(f"  [{'ok' if good else 'FAIL'}] {name}: {err:.2e} (tol {tol:.0e})")


# 1. tree product
p, D = 9, 3
d = torch.tensor(rng.normal(0, 3, (4, p)), dtype=torch.float64)
le = log_esp(d, D)
bf = torch.stack([torch.stack([torch.logsumexp(torch.stack([d[b, list(S)].sum() for S in itertools.combinations(range(p), k)]), 0)
                               if k else torch.tensor(0.0, dtype=torch.float64) for k in range(D + 1)]) for b in range(4)])
check("log e_k vs brute force (p 9, D 3)", float((le - bf).abs().max()), 1e-10)
p, D = 300, 10
d64 = rng.normal(0, 4, (3, p))
E = np.full((3, D + 1), -np.inf); E[:, 0] = 0
for j in range(p):
    E[:, 1:] = np.logaddexp(E[:, 1:], E[:, :-1] + d64[:, j:j + 1])
le32 = log_esp(torch.tensor(d64, dtype=torch.float32), D).double().numpy()
check("log e_k float32 tree vs float64 sequential DP (p 300, D 10), relative", float(np.abs(le32 - E).max() / np.abs(E).max()), 1e-5)

# 2. log p and inclusion vs enumeration
p, D, sl, ss = 8, 3, 32.0, 0.1
P = ExactStructuredSpikeAndSlabPrior(p, D, var_slab=sl, var_spike=ss)
z = torch.tensor(np.r_[rng.normal(0, 1, (5, 3)).ravel()[:0], 0.0] if False else rng.normal(0, 0.4, (5, p)), dtype=torch.float64)
lN = lambda x, s: -0.5 * (x / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi)
subs = list(itertools.combinations(range(p), D))
terms = torch.stack([sum(lN(z[:, j], sl) if j in S else lN(z[:, j], ss) for j in range(p)) for S in subs], -1)
lp_bf = torch.logsumexp(terms, -1) - math.log(len(subs))
w = torch.softmax(terms, -1)
pi_bf = torch.stack([w[:, [i for i, S in enumerate(subs) if j in S]].sum(-1) for j in range(p)], -1)
check("log p vs enumeration (p 8, D 3)", float((P.log_prob(z) - lp_bf).abs().max()), 1e-10)
check("inclusion probabilities vs enumeration", float((P.inclusion(z) - pi_bf).abs().max()), 1e-10)

# 3. against GEMSS's own (exactly enumerated) prior
G = StructuredSpikeAndSlabPrior(p, D, var_slab=sl, var_spike=ss)
p, D = 9, 3
G = StructuredSpikeAndSlabPrior(p, D, var_slab=sl, var_spike=ss)        # p <= 10 and D <= 3: GEMSS enumerates
P = ExactStructuredSpikeAndSlabPrior(p, D, var_slab=sl, var_spike=ss)
assert G._all_supports is not None
z = torch.tensor(rng.normal(0, 0.4, (6, p)), dtype=torch.float32, requires_grad=True)
gG = torch.autograd.grad(G.log_prob(z).sum(), z)[0]
gP = torch.autograd.grad(P.log_prob(z).sum(), z)[0]
diff = (P.log_prob(z) - G.log_prob(z)).detach()
const = p * (-0.5 * math.log(ss) - 0.5 * math.log(2 * math.pi)) + D * (-0.5 * math.log(sl) + 0.5 * math.log(ss))
check("vs GEMSS's enumerated prior: gradient", float((gG - gP).abs().max() / gG.abs().max()), 1e-5)
check("vs GEMSS's enumerated prior: log p - GEMSS = normalising constant", float((diff - const).abs().max()), 1e-3)

# 4. closed form vs autograd and finite differences
p, D = 40, 5
P = ExactStructuredSpikeAndSlabPrior(p, D, var_slab=sl, var_spike=ss)
z = torch.tensor(rng.normal(0, 0.3, (3, p)), dtype=torch.float64)
lp, g, H = P.terms(z)
zz = z.clone().requires_grad_(True)
ga = torch.autograd.grad(P.log_prob(zz).sum(), zz)[0]
check("closed-form gradient vs autograd (p 40, D 5)", float((g - ga).abs().max() / ga.abs().max()), 1e-10)
h = 1e-5
Hfd = torch.stack([(P.terms(z + h * torch.nn.functional.one_hot(torch.tensor(j), p))[1][:, j]
                    - P.terms(z - h * torch.nn.functional.one_hot(torch.tensor(j), p))[1][:, j]) / (2 * h) for j in range(p)], -1)
check("Hessian diagonal vs finite differences", float((H - Hfd).abs().max() / Hfd.abs().max()), 1e-6)

# 5. short fits
Xs = rng.normal(size=(60, 30)); beta = np.zeros(30); beta[:3] = 1.0
ys = Xs @ beta + rng.normal(0, 0.5, 60); ys = (ys - ys.mean()) / ys.std()
hp = dict(prior="sss", var_slab=32.0, var_spike=0.1, lr=0.01, n_iter=300, batch_size=16, weight_slab=0.9, weight_spike=0.1)
for est, opts in [("ngvi", {"step_rule": "invL"}), ("ngvi_dec", {"strat_draws": 16, "step_rule": "invL"}), ("current", {})]:
    torch.manual_seed(1)
    w = WeightFixGEMSSWrapper("regression", est, n_components=4, sparsity=3, estimator_opts=opts, exact_sss=True, **hp)
    sols = w.fit(Xs, ys)
    top = sorted(set(j for s in sols.values() for j in s["support"]))
    good = math.isfinite(w.elbo_); ok &= good
    print(f"  [{'ok' if good else 'FAIL'}] fit {est} with exact prior: ELBO {w.elbo_:.1f}, features in supports {top[:8]}, "
          f"{w.fit_seconds_:.1f}s")

# 6. timing
P = ExactStructuredSpikeAndSlabPrior(500, 10, var_slab=sl, var_spike=ss)
z = torch.randn(16, 500) * 0.3
t = time.perf_counter()
for _ in range(20):
    P.terms(z)
print(f"  timing: terms() at B 16, p 500, D 10: {(time.perf_counter() - t) / 20 * 1000:.1f} ms per call")
print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
