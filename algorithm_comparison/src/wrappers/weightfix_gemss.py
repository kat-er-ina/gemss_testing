"""Weight-gradient fixes for GEMSS (special-issue comparison plan, sec. 14).

GEMSS (gemss 1.1.0) draws the mixture component with ``torch.multinomial``, so
autograd only sees the mixture weights through -log q(z) at fixed z, whose
expectation is zero: the weights random-walk (plan sec. 13). This module keeps
GEMSS's model -- its prior, its unit-noise Gaussian likelihood and its
mixture of diagonal Gaussians, as built by ``BayesianFeatureSelector`` -- and
swaps only the optimisation loop:

  current     GEMSS's own loop (``sel.optimize``), unchanged                    [F0]
  frozen      as current, weights held at their initial value 1/K           [control]
  strat       stratified ELBO: n draws from EVERY component and
              ELBO = sum_k a_k mean_b h(z_kb); autograd then gives the exact
              weight gradient and no component goes without draws. Adam.        [F1]
  score       GEMSS's sampler; weight gradient by the score function
              mean_b (h_b - leave-one-out mean) * d log q(z_b) / d logits
              (Euclidean form of Lin, Khan & Schmidt 2019, App. B.3). Adam.     [F2]
  ngvi        natural-gradient VI for a diagonal mixture of Gaussians, the updates of
              Lin, Khan & Schmidt (2019, ICML, arXiv 1906.02914) eq. 16:
                prec_c  <- prec_c + b E_qc[H] (+ b^2 E_qc[H]^2 / (2 prec_c))
                mu_c    <- mu_c - b E_qc[g] / prec_c
                logit_c <- logit_c - b_w (E_qc[h] - E_qK[h])
              with g / H the gradient / Hessian diagonal of h = log q - log p(y, z) in
              closed form. Expectations are taken with draws from EACH component
              (their derivation's E_{q(z|w=c)} form). The b^2 term is the positive-
              definite correction of the improved Bayesian learning rule, which
              keeps precisions positive.                                          [F3]
  ngvi_mix    the same updates estimated as in their paper, from mixture draws
              weighted by d_c = q_c(z)/q(z). Diverges on our problems: a rarely
              drawn low-weight component gets d_c ~ 1/a_c, an effective step > 1,
              and the b^2 term then explodes. Kept to document that.
  fixedpoint  GEMSS's sampler with the weights held fixed between updates; every
              ``fp_every`` steps a_k <- a_k exp(eta (E_qk[f] - E_q[f])) from
              ``fp_draws`` draws per component, f = log p(y, z) - log q(z). With
              eta = 1 and separated components this is the closed form
              a_k ∝ exp(component ELBO).                                         [F4]

Every estimator except ``current`` / ``frozen`` uses a numerically stable softmax
for the weights (GEMSS's ``get_alpha`` exponentiates raw logits and silently falls
back to uniform weights on overflow).
"""

import math
import time
import types
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from gemss.feature_selection.inference import BayesianFeatureSelector
from gemss.feature_selection.models import SpikeAndSlabPrior, StructuredSpikeAndSlabPrior

from .mechanism_gemss import MechanismGEMSSWrapper, _elbo_plateau_iter

ESTIMATORS = ("current", "frozen", "strat", "score", "ngvi", "ngvi_mix", "fixedpoint")
LOG2PI = math.log(2 * math.pi)


# ----------------------------------------------------------------------------- mixture helpers
def log_alpha(mix) -> torch.Tensor:
    """log weights, stable: log_softmax([logits, 0])."""
    la = mix._log_alpha
    return F.log_softmax(torch.cat([la, la.new_zeros(1)]), dim=0)


def _stable_get_alpha(self) -> torch.Tensor:
    la = self._log_alpha
    return F.softmax(torch.cat([la, la.new_zeros(1)]), dim=0)


def comp_logpdf(z: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """log N(z | mu_k, diag var_k) for every draw and component: (B, K). Same
    expression as GaussianMixture.log_prob."""
    return -0.5 * torch.sum(torch.log(2 * torch.pi * var)[None] + (z[:, None, :] - mu[None]) ** 2 / var[None], dim=-1)


def logq_terms(z, mu, var, la):
    """log q(z), its gradient and Hessian diagonal for a diagonal Gaussian mixture,
    plus responsibilities r (B, K). Closed form, no autograd."""
    lpk = comp_logpdf(z, mu, var) + la[None]
    lq = torch.logsumexp(lpk, -1)
    r = torch.softmax(lpk, -1)
    sk = -(z[:, None, :] - mu[None]) / var[None]                    # d log q_k / dz
    g = (r[..., None] * sk).sum(1)
    H = (r[..., None] * (-1.0 / var[None] + sk ** 2)).sum(1) - g ** 2
    return lq, g, H, r


# ----------------------------------------------------------------------------- targets
class Target:
    """log p(y, z) up to a constant. ``logp`` is autograd-able; ``terms`` returns
    (logp, grad, hess_diag) in closed form (needed by ngvi only)."""

    def logp(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def terms(self, z: torch.Tensor):  # pragma: no cover - interface
        raise NotImplementedError


class RegressionTarget(Target):
    """GEMSS's model: its prior object plus -1/2 ||X z - y||^2."""

    def __init__(self, sel: BayesianFeatureSelector):
        if sel._has_missing_X:
            raise ValueError("weightfix estimators do not handle missing X")
        self.sel, self.prior, self.X, self.y = sel, sel.prior, sel.X, sel.y
        self.colsq = (self.X ** 2).sum(0)

    def logp(self, z):
        return self.sel.prior.log_prob(z) + self.sel.log_likelihood(z)

    def _prior_terms(self, z):
        P = self.prior
        if isinstance(P, SpikeAndSlabPrior):          # p_j = w1 N(z; 0, s1) + w2 N(z; 0, s2), s = s.d.
            s1, s2, w1, w2 = P.var_slab, P.var_spike, P.w_slab, P.w_spike
            l1 = math.log(w1) - 0.5 * (z / s1) ** 2 - math.log(s1)
            l2 = math.log(w2) - 0.5 * (z / s2) ** 2 - math.log(s2)
            lp = torch.logaddexp(l1, l2).sum(-1)
            r = torch.sigmoid(l1 - l2)
            a1, a2 = 1.0 / s1 ** 2, 1.0 / s2 ** 2
            prec = r * a1 + (1 - r) * a2
            return lp, -z * prec, -prec + z ** 2 * r * (1 - r) * (a1 - a2) ** 2
        if isinstance(P, StructuredSpikeAndSlabPrior):
            # Same estimator as P.log_prob (log-mean-exp over supports drawn afresh),
            # same RNG calls, same (non-standard) normalisation -0.5 (z/v)^2 - 0.5 log v.
            p, D = z.shape[-1], P.sparsity
            if P._all_supports is not None:
                sup = torch.tensor(P._all_supports, dtype=torch.long, device=z.device)
            else:
                S = int(np.round(100 * P.sample_more_priors_coeff)) if P.sample_more_priors_coeff else 100
                sup = torch.topk(torch.rand(S, p, device=z.device), k=D, dim=1).indices
            m = torch.zeros(sup.shape[0], p, device=z.device, dtype=z.dtype)
            m.scatter_(1, sup, 1.0)
            vl, vs = P.var_slab, P.var_spike
            spike = -0.5 * ((z / vs) ** 2 + math.log(vs))
            slab = -0.5 * ((z / vl) ** 2 + math.log(vl))
            delta = slab - spike
            a = spike.sum(-1, keepdim=True) + delta @ m.T              # (B, S)
            lp = torch.logsumexp(a, -1) - math.log(sup.shape[0])
            pj = torch.softmax(a, -1) @ m                               # P_w(j in S)
            d1 = -z / vl ** 2 + z / vs ** 2
            d2 = -1.0 / vl ** 2 + 1.0 / vs ** 2
            g = -z / vs ** 2 + pj * d1
            H = -1.0 / vs ** 2 + pj * d2 + d1 ** 2 * pj * (1 - pj)
            return lp, g, H
        raise ValueError(f"ngvi: no closed form for prior {type(P).__name__}")

    def terms(self, z):
        lp, gp, Hp = self._prior_terms(z)
        res = z @ self.X.T - self.y[None]
        return lp - 0.5 * (res ** 2).sum(-1), gp - res @ self.X, Hp - self.colsq[None]


class GaussMixTarget(Target):
    """Unnormalised target sum_m w_m N(z | c_m, s^2 I) (for the known-weight check)."""

    def __init__(self, centers: torch.Tensor, weights, sd: float):
        self.c = centers
        self.var = torch.full_like(centers, sd ** 2)
        self.la = torch.log(torch.as_tensor(weights, dtype=centers.dtype))

    def logp(self, z):
        return torch.logsumexp(comp_logpdf(z, self.c, self.var) + self.la[None], -1)

    def terms(self, z):
        lq, g, H, _ = logq_terms(z, self.c, self.var, self.la)
        return lq, g, H


# ----------------------------------------------------------------------------- runner
class WeightFixRunner:
    """Runs one of ESTIMATORS on a GaussianMixture ``mix`` against ``target``.

    For the regression model, ``current`` and ``frozen`` call ``sel.optimize`` so
    that ``current`` is GEMSS's code path bit for bit. ``run(n)`` can be called
    repeatedly (trajectory checkpoints); optimiser state persists."""

    def __init__(self, mix, target: Target, estimator: str, batch_size: int = 16, lr: float = 0.01,
                 sel: Optional[BayesianFeatureSelector] = None, strat_draws: Optional[int] = None,
                 ngvi_beta: float = 0.02, ngvi_beta_w: Optional[float] = None,
                 fp_every: int = 100, fp_draws: int = 64, fp_eta: float = 1.0, logit_clip: float = 80.0):
        if estimator not in ESTIMATORS:
            raise ValueError(f"unknown estimator {estimator}")
        self.mix, self.target, self.est, self.sel = mix, target, estimator, sel
        self.B, self.K, self.p = batch_size, mix.mu.shape[0], mix.mu.shape[1]
        self.strat_n = strat_draws or max(1, math.ceil(batch_size / self.K))
        self.beta, self.beta_w = ngvi_beta, (ngvi_beta if ngvi_beta_w is None else ngvi_beta_w)
        self.fp_every, self.fp_draws, self.fp_eta, self.clip = fp_every, fp_draws, fp_eta, logit_clip
        self.t = 0
        if estimator not in ("current", "frozen"):
            mix.get_alpha = types.MethodType(_stable_get_alpha, mix)
        if estimator in ("frozen", "fixedpoint"):
            mix._log_alpha.requires_grad_(False)
        if estimator in ("current", "frozen", "strat", "score", "fixedpoint"):
            self.opt = sel.opt if sel is not None else torch.optim.Adam(
                [q for q in mix.parameters()], lr=lr)
        if estimator in ("ngvi", "ngvi_mix"):
            with torch.no_grad():
                self.mu = mix.mu.detach().clone()
                self.prec = 1.0 / mix.get_variance().detach().clone()
                self.logit = mix._log_alpha.detach().clone()

    # -- helpers
    def _h_own(self, z):
        """h = log p(y, z) - log q(z), with a stable log q (autograd w.r.t. all mixture params)."""
        var = self.mix.get_variance()
        lq = torch.logsumexp(comp_logpdf(z, self.mix.mu, var) + log_alpha(self.mix)[None], -1)
        return self.target.logp(z) - lq

    def _clip_logits(self):
        with torch.no_grad():
            self.mix._log_alpha.clamp_(-self.clip, self.clip)

    def sync(self):
        """Write ngvi state back into the mixture (readouts use the mixture)."""
        if self.est not in ("ngvi", "ngvi_mix"):
            return
        with torch.no_grad():
            self.mix.mu.copy_(self.mu)
            v = 1.0 / self.prec
            self.mix._log_var.copy_(torch.where(v > 20, v, torch.log(torch.expm1(v))))
            self.mix._log_alpha.copy_(self.logit)

    # -- steps
    def _step_generic_current(self):
        z, _ = self.mix.sample(self.B)
        lq = self.mix.log_prob(z)
        h = self.target.logp(z) - lq
        self.opt.zero_grad(); (-h.mean()).backward(); self.opt.step()
        return float(h.mean())

    def _step_strat(self):
        K, n, p = self.K, self.strat_n, self.p
        var = self.mix.get_variance()
        z = self.mix.mu[:, None, :] + torch.randn(K, n, p, device=var.device) * var.sqrt()[:, None, :]
        h = self._h_own(z.reshape(K * n, p)).view(K, n).mean(1)
        obj = (log_alpha(self.mix).exp() * h).sum()
        self.opt.zero_grad(); (-obj).backward(); self.opt.step(); self._clip_logits()
        return float(obj)

    def _step_score(self):
        z, _ = self.mix.sample(self.B)
        var = self.mix.get_variance()
        lpk = comp_logpdf(z, self.mix.mu, var)
        la = log_alpha(self.mix)
        h = self.target.logp(z) - torch.logsumexp(lpk + la.detach()[None], -1)     # mu/var paths
        lq_alpha = torch.logsumexp(lpk.detach() + la[None], -1)                     # weight path
        hd = h.detach()
        loo = (hd.sum() - hd) / (self.B - 1)
        obj = h.mean() + ((hd - loo) * lq_alpha).mean()
        self.opt.zero_grad(); (-obj).backward(); self.opt.step(); self._clip_logits()
        return float(hd.mean())

    def _ngvi_update(self, Hc, gc, dlogit):
        b = self.beta
        prec = self.prec + b * Hc + b ** 2 * Hc ** 2 / (2 * self.prec)
        self.mu = self.mu - b * gc / prec
        self.prec = prec
        self.logit = (self.logit - self.beta_w * dlogit).clamp(-self.clip, self.clip)

    @torch.no_grad()
    def _step_ngvi(self):
        K, n, p = self.K, self.strat_n, self.p
        la = F.log_softmax(torch.cat([self.logit, self.logit.new_zeros(1)]), 0)
        var = 1.0 / self.prec
        z = (self.mu[:, None, :] + torch.randn(K, n, p, device=var.device) * var.sqrt()[:, None, :]).reshape(K * n, p)
        lq, gq, Hq, _ = logq_terms(z, self.mu, var, la)
        lp, gp, Hp = self.target.terms(z)
        h = (lq - lp).view(K, n).mean(1)
        gc = (gq - gp).view(K, n, p).mean(1)
        Hc = (Hq - Hp).view(K, n, p).mean(1)
        self._ngvi_update(Hc, gc, h[:-1] - h[-1])
        return float(-(la.exp() * h).sum())

    @torch.no_grad()
    def _step_ngvi_mix(self):
        la = F.log_softmax(torch.cat([self.logit, self.logit.new_zeros(1)]), 0)
        var = 1.0 / self.prec
        idx = torch.multinomial(la.exp(), self.B, replacement=True)
        z = self.mu[idx] + torch.randn(self.B, self.p, device=var.device) * var[idx].sqrt()
        lq, gq, Hq, r = logq_terms(z, self.mu, var, la)
        lp, gp, Hp = self.target.terms(z)
        h, g, H = lq - lp, gq - gp, Hq - Hp                  # h = -(ELBO integrand)
        d = r / la.exp()[None]                                # q_c(z) / q(z), (B, K)
        Hc = (d[..., None] * H[:, None, :]).mean(0)
        gc = (d[..., None] * g[:, None, :]).mean(0)
        hb = h - (h.sum() - h) / (self.B - 1)
        self._ngvi_update(Hc, gc, ((d[:, :-1] - d[:, -1:]) * hb[:, None]).mean(0))
        return float(-h.mean())

    @torch.no_grad()
    def _fixedpoint_update(self):
        K, n, p = self.K, self.fp_draws, self.p
        var = self.mix.get_variance()
        z = self.mix.mu[:, None, :] + torch.randn(K, n, p, device=var.device) * var.sqrt()[:, None, :]
        Ef = self._h_own(z.reshape(K * n, p)).view(K, n).mean(1)
        la = log_alpha(self.mix)
        la = la + self.fp_eta * (Ef - (la.exp() * Ef).sum())
        la = (la - torch.logsumexp(la, 0)).clamp(min=-self.clip / 2)
        la = la - torch.logsumexp(la, 0)
        self.mix._log_alpha.copy_(la[:-1] - la[-1])

    def run(self, n_steps: int) -> List[float]:
        hist: List[float] = []
        if self.est in ("current", "frozen") and self.sel is not None:
            self.sel.n_iter = n_steps
            hist = list(self.sel.optimize(regularize=False, verbose=False).get("elbo", []))
            self.t += n_steps
            return hist
        for _ in range(n_steps):
            if self.est in ("current", "frozen"):
                hist.append(self._step_generic_current())
            elif self.est == "strat":
                hist.append(self._step_strat())
            elif self.est == "score":
                hist.append(self._step_score())
            elif self.est == "ngvi":
                hist.append(self._step_ngvi())
            elif self.est == "ngvi_mix":
                hist.append(self._step_ngvi_mix())
            elif self.est == "fixedpoint":
                hist.append(self._step_generic_current())
                if (self.t + 1) % self.fp_every == 0:
                    self._fixedpoint_update()
            self.t += 1
        self.sync()
        return hist


def make_runner(sel: BayesianFeatureSelector, estimator: str, **opts) -> WeightFixRunner:
    return WeightFixRunner(sel.mixture, RegressionTarget(sel), estimator, batch_size=sel.batch_size,
                           sel=sel, **opts)


# ----------------------------------------------------------------------------- benchmark wrapper
class WeightFixGEMSSWrapper(MechanismGEMSSWrapper):
    """MechanismGEMSSWrapper('joint') with the optimisation loop replaced by one of
    ESTIMATORS. Same attributes after fit: elbo_, mu_, var_, alpha_, fit_seconds_."""

    def __init__(self, task: str = "regression", estimator: str = "strat", n_components: int = 6,
                 sparsity: int = 5, estimator_opts: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if task != "regression":
            raise ValueError("weightfix estimators: regression only")
        super().__init__(task, "joint", n_components, sparsity, **kwargs)
        self.estimator, self.estimator_opts = estimator, dict(estimator_opts or {})

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, List[int]]]:
        sel_kwargs = {k: v for k, v in self.params.items() if k in self._SELECTOR_KEYS}
        sel = BayesianFeatureSelector(n_features=X.shape[1], n_components=self.n_components, X=X, y=y,
                                      sss_sparsity=self.sparsity, **sel_kwargs)
        runner = make_runner(sel, self.estimator, **self.estimator_opts)
        t0 = time.perf_counter()
        elbo_hist = runner.run(sel.n_iter)
        self.fit_seconds_ = time.perf_counter() - t0
        self.n_iter_ = len(elbo_hist)
        self.n_iter_converged_ = _elbo_plateau_iter(elbo_hist) if elbo_hist else self.n_iter_
        with torch.no_grad():
            z_eval, _ = sel.mixture.sample(max(256, sel.batch_size))
            self.elbo_ = float(sel.elbo(z_eval).item())
        self.mu_ = sel.mixture.mu.detach().cpu().numpy()
        self.var_ = sel.mixture.get_variance().detach().cpu().numpy()
        self.alpha_ = sel.mixture.get_alpha().detach().cpu().numpy()
        D = self.sparsity
        return {f"component_{k}": {"support": np.argsort(np.abs(self.mu_[k]))[::-1][:D].tolist()}
                for k in range(self.n_components)}
