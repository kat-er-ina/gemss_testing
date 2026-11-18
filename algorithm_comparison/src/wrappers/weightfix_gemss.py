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

ESTIMATORS = ("current", "frozen", "strat", "score", "ngvi", "ngvi_mix", "fixedpoint", "ngvi_frozen",
              "f0_joint", "f0_dec", "f0_ind", "ngvi_dec")
# Coupling test (plan sec. 18):
#   f0_joint  F0's step (multinomial draws, Adam, published get_alpha) through the generic path, so that
#             f0_dec differs from it only in log q.
#   f0_dec    as f0_joint, but each draw's log q uses its own component only: log a_k + log N_k(z).
#             Removes exactly the entropy coupling between components.
#   f0_ind    K independent one-component fits: strat_draws (default ceil(16/K)) draws per component,
#             objective sum_k E_qk[log p - log N_k], weights fixed (irrelevant). One Adam over all
#             parameters is equivalent to K separate Adams (Adam is elementwise).
#   ngvi_dec  ngvi with the own-component log q terms (= independent NGD fits; the weights converge to
#             a_k ~ exp(per-component ELBO), i.e. a post-hoc weighting).
F0_GENERIC = ("f0_joint", "f0_dec", "f0_ind")
# ngvi_frozen (plan sec. 16): the ngvi updates for means and precisions with the weight step set to 0,
# i.e. weights stay at their uniform start; separates "NGD components" from "learned weights".
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
        self.omega = 1.0                              # likelihood tempering (plan sec. 19, annealing)

    def logp(self, z):
        P = self.sel.prior
        if isinstance(P, SpikeAndSlabPrior) and torch.is_tensor(P.w_slab):
            # per-coordinate weights (plan sec. 24): GEMSS's log(w1 N1 + w2 N2) underflows to log 0 for a pure-spike
            # coordinate at |z| > ~1.3 in float32; the same density in logaddexp form (scalar path unchanged)
            return self._prior_terms(z)[0] + self.omega * self.sel.log_likelihood(z)
        return self.sel.prior.log_prob(z) + self.omega * self.sel.log_likelihood(z)

    def _prior_terms(self, z):
        P = self.prior
        if isinstance(P, SpikeAndSlabPrior):          # p_j = w1 N(z; 0, s1) + w2 N(z; 0, s2), s = s.d.
            s1, s2, w1, w2 = P.var_slab, P.var_spike, P.w_slab, P.w_spike
            # weights may be per-coordinate tensors (plan sec. 24: Gaussian-nuisance control, slab weight 0 on
            # nuisance coordinates -> log 0 = -inf -> pure spike); the scalar path is unchanged (bit-identical)
            lw = lambda w: torch.log(w.to(z)) if torch.is_tensor(w) else math.log(w)
            l1 = lw(w1) - 0.5 * (z / s1) ** 2 - math.log(s1)
            l2 = lw(w2) - 0.5 * (z / s2) ** 2 - math.log(s2)
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
        w = self.omega
        return lp - 0.5 * w * (res ** 2).sum(-1), gp - w * (res @ self.X), Hp - w * self.colsq[None]


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
                 fp_every: int = 100, fp_draws: int = 64, fp_eta: float = 1.0, logit_clip: float = 80.0,
                 anneal: Optional[tuple] = None, tie: Optional[torch.Tensor] = None):
        if estimator not in ESTIMATORS:
            raise ValueError(f"unknown estimator {estimator}")
        self.mix, self.target, self.est, self.sel = mix, target, estimator, sel
        self.B, self.K, self.p = batch_size, mix.mu.shape[0], mix.mu.shape[1]
        self.strat_n = strat_draws or max(1, math.ceil(batch_size / self.K))
        self.beta, self.beta_w = ngvi_beta, (ngvi_beta if ngvi_beta_w is None else ngvi_beta_w)
        if estimator == "ngvi_frozen":
            self.beta_w = 0.0
        self.fp_every, self.fp_draws, self.fp_eta, self.clip = fp_every, fp_draws, fp_eta, logit_clip
        self.t = 0
        # annealing (plan sec. 19): likelihood weight omega(t) = omega0 ** (1 - t / T) for t < T, then 1
        self.anneal = tuple(anneal) if anneal and anneal[0] is not None and anneal[0] < 1 else None
        if self.anneal and estimator in ("current", "frozen") and sel is not None:
            raise ValueError("annealing needs the generic loop: use f0_joint instead of current")
        if estimator not in ("current", "frozen") + F0_GENERIC:
            mix.get_alpha = types.MethodType(_stable_get_alpha, mix)
        if estimator in ("frozen", "fixedpoint", "f0_ind"):
            mix._log_alpha.requires_grad_(False)
        if estimator in ("current", "frozen", "strat", "score", "fixedpoint") + F0_GENERIC:
            self.opt = sel.opt if sel is not None else torch.optim.Adam(
                [q for q in mix.parameters()], lr=lr)
        if estimator in ("ngvi", "ngvi_mix", "ngvi_frozen", "ngvi_dec"):
            with torch.no_grad():
                self.mu = mix.mu.detach().clone()
                self.prec = 1.0 / mix.get_variance().detach().clone()
                self.logit = mix._log_alpha.detach().clone()
        # localised coupling (plan sec. 27): coordinates with tie[j] = True share one mean and precision across
        # components (the mixture's components differ only on the others). Started at component 0's values.
        self.tie = None
        if tie is not None and bool(torch.as_tensor(tie).any()):
            if estimator != "ngvi":
                raise ValueError("tie (localised coupling) is implemented for ngvi only")
            self.tie = torch.as_tensor(tie, dtype=torch.bool, device=self.mu.device)
            with torch.no_grad():
                self.mu[:, self.tie] = self.mu[0, self.tie]
                self.prec[:, self.tie] = self.prec[0, self.tie]

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
        if self.est not in ("ngvi", "ngvi_mix", "ngvi_frozen", "ngvi_dec"):
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

    def _step_f0(self, decoupled: bool):
        z, idx = self.mix.sample(self.B)
        if decoupled:
            var = self.mix.get_variance()
            lq = (comp_logpdf(z, self.mix.mu, var).gather(1, idx[:, None]).squeeze(1)
                  + torch.log(self.mix.get_alpha())[idx])
        else:
            lq = self.mix.log_prob(z)
        h = self.target.logp(z) - lq
        self.opt.zero_grad(); (-h.mean()).backward(); self.opt.step()
        return float(h.mean())

    def _step_f0_ind(self):
        K, n, p = self.K, self.strat_n, self.p
        mu, var = self.mix.mu, self.mix.get_variance()
        z = mu[:, None, :] + torch.randn(K, n, p, device=var.device) * var.sqrt()[:, None, :]
        lq = -0.5 * (torch.log(2 * torch.pi * var)[:, None, :] + (z - mu[:, None, :]) ** 2 / var[:, None, :]).sum(-1)
        h = self.target.logp(z.reshape(K * n, p)).view(K, n) - lq
        self.opt.zero_grad(); (-h.mean(1).sum()).backward(); self.opt.step()
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
        eps = torch.randn(K, n, p, device=var.device)
        if self.tie is not None:                             # shared coordinates: one draw for all components
            eps[:, :, self.tie] = eps[:1, :, self.tie]
        z = (self.mu[:, None, :] + eps * var.sqrt()[:, None, :]).reshape(K * n, p)
        if self.est == "ngvi_dec":                           # own component only (no coupling)
            c = torch.arange(K, device=var.device).repeat_interleave(n)
            d, vc = z - self.mu[c], var[c]
            lq = la[c] - 0.5 * (torch.log(2 * torch.pi * vc) + d ** 2 / vc).sum(-1)
            gq, Hq = -d / vc, -1.0 / vc
        else:
            lq, gq, Hq, _ = logq_terms(z, self.mu, var, la)
        lp, gp, Hp = self.target.terms(z)
        h = (lq - lp).view(K, n).mean(1)
        gc = (gq - gp).view(K, n, p).mean(1)
        Hc = (Hq - Hp).view(K, n, p).mean(1)
        if self.tie is not None:
            # a shared factor's expected gradient / curvature is taken under the whole mixture: the
            # weight-averaged per-component expectations (identical inputs keep the shared values identical)
            wk = la.exp()[:, None]
            gc[:, self.tie] = (wk * gc[:, self.tie]).sum(0)
            Hc[:, self.tie] = (wk * Hc[:, self.tie]).sum(0)
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

    def omega_at(self, t: int) -> float:
        if not self.anneal:
            return 1.0
        w0, T = self.anneal
        return 1.0 if t >= T else float(w0 ** (1.0 - t / T))

    def run(self, n_steps: int) -> List[float]:
        hist: List[float] = []
        if self.est in ("current", "frozen") and self.sel is not None:
            self.sel.n_iter = n_steps
            hist = list(self.sel.optimize(regularize=False, verbose=False).get("elbo", []))
            self.t += n_steps
            return hist
        for _ in range(n_steps):
            if self.anneal:
                self.target.omega = self.omega_at(self.t)
            if self.est in ("current", "frozen"):
                hist.append(self._step_generic_current())
            elif self.est == "strat":
                hist.append(self._step_strat())
            elif self.est == "score":
                hist.append(self._step_score())
            elif self.est in ("f0_joint", "f0_dec"):
                hist.append(self._step_f0(self.est == "f0_dec"))
            elif self.est == "f0_ind":
                hist.append(self._step_f0_ind())
            elif self.est in ("ngvi", "ngvi_frozen", "ngvi_dec"):
                hist.append(self._step_ngvi())
            elif self.est == "ngvi_mix":
                hist.append(self._step_ngvi_mix())
            elif self.est == "fixedpoint":
                hist.append(self._step_generic_current())
                if (self.t + 1) % self.fp_every == 0:
                    self._fixedpoint_update()
            self.t += 1
        if self.anneal:
            self.target.omega = self.omega_at(self.t)
        self.sync()
        return hist


INITS = ("default", "common_zero", "common_screen", "overlap", "screen_diverse")


def screen_scores(X, y):
    """|corr(X_j, y)| (as in init_probe screen_conf)."""
    Xf = np.nan_to_num(np.asarray(X, float), nan=0.0); yc = np.asarray(y, float) - np.mean(y)
    return np.abs((Xf - Xf.mean(0)).T @ yc / len(yc)) / (Xf.std(0) * yc.std() + 1e-12)


@torch.no_grad()
def apply_init(mix, mode: str, X, y, D: int, seed: int, jitter: float = 0.01, overlap_scale: float = 0.1,
               mag: float = 2.0, conf_var: float = 0.1):
    """Starting point options for the coupling test (plan sec. 18); call BEFORE make_runner.
    default        GEMSS: mu ~ N(0, 1), var = softplus(0), uniform weights (no change)
    common_zero    every component at 0 + jitter * N(0, 1)            (full overlap; default var)
    common_screen  every component at one screen point + jitter       (top-D |corr| features = mag)
    overlap        mu ~ N(0, overlap_scale^2): distinct but overlapping starts (default var)
    screen_diverse init_probe screen_conf: per component D features drawn from the top-|corr| pool,
                   prob. proportional to score, set to mag; var = conf_var"""
    if mode == "default":
        return
    K, p = mix.mu.shape
    g = torch.Generator().manual_seed(10_000 + seed)
    rng = np.random.default_rng(10_000 + seed)
    if mode == "common_zero":
        mu = jitter * torch.randn(K, p, generator=g)
    elif mode == "common_screen":
        mu0 = torch.zeros(p); mu0[np.argsort(screen_scores(X, y))[::-1][:D].copy()] = mag
        mu = mu0[None] + jitter * torch.randn(K, p, generator=g)
    elif mode == "overlap":
        mu = overlap_scale * torch.randn(K, p, generator=g)
    elif mode == "screen_diverse":
        sc = screen_scores(X, y); M = min(p, max(K * D, 4 * D))
        pool = np.argsort(sc)[::-1][:M]; w = sc[pool] / sc[pool].sum()
        mu = torch.zeros(K, p)
        for k in range(K):
            mu[k, rng.choice(pool, size=D, replace=False, p=w)] = mag
        mix._log_var.fill_(math.log(math.expm1(conf_var)))
    else:
        raise ValueError(f"unknown init {mode}")
    mix.mu.copy_(mu.to(mix.mu))


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
