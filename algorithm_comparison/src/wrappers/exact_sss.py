"""Plan sec. 47: the exact structured spike-and-slab density.

GEMSS's StructuredSpikeAndSlabPrior (gemss/feature_selection/models.py) is meant to be the uniform mixture over all
D-subsets S of the coordinates,
    p(z) = C(p, D)^-1 sum_{|S| = D} prod_{j in S} N(z_j; 0, s_slab^2) prod_{j not in S} N(z_j; 0, s_spike^2),
but its log_prob is a log-mean-exp over 100 random supports drawn afresh at each call (exact only for p <= 10, D <= 3),
with the normaliser -0.5 (z/v)^2 - 0.5 log v (v acts as a standard deviation, the log term is halved).

Exactly: with spike_j = log N(z_j; 0, s_spike^2), delta_j = log N(z_j; 0, s_slab^2) - spike_j and r_j = exp(delta_j),
    log p(z) = sum_j spike_j + log e_D(r) - log C(p, D),
e_D = the D-th elementary symmetric polynomial (coefficient of t^D in prod_j (1 + r_j t)). Computed here in log space by
a pairwise tree product of the polynomials truncated at degree D: O(p D^2) work, depth log2 p, vectorised over the
batch. The inclusion probability pi_j = P(j in S | z) = d log e_D / d delta_j (autograd), and since delta_j enters e_D
only through r_j, d pi_j / d delta_j = pi_j (1 - pi_j). Gradient and Hessian diagonal of log p therefore have the same
form as the Monte Carlo version in weightfix_gemss._prior_terms, with the exact pi in place of the sampled weights:
    g_j = spike'_j + pi_j delta'_j,   H_jj = spike''_j + pi_j delta''_j + delta'_j^2 pi_j (1 - pi_j).
Parameters keep GEMSS's meaning (var_slab = 32 and var_spike = 0.1 are used as standard deviations, as GEMSS's code
does), so fits stay on the same scale; the normaliser is the proper Gaussian one, so the ELBO is a bound on the
evidence and comparable across D. For fixed D the normaliser constant cancels in pi and in the gradient.
"""
import math

import torch

from gemss.feature_selection.models import StructuredSpikeAndSlabPrior

NEG = -1e30          # finite stand-in for log 0 (keeps logsumexp gradients finite)
_LOG2PI = math.log(2 * math.pi)


_IDX = {}


def _antidiag_index(D, device):
    """Flat indices into (D+1)^2 (+1 pad slot) giving, for each k, the pairs (a, k - a); pad elsewhere."""
    key = (D, str(device))
    if key not in _IDX:
        n = D + 1
        idx = torch.full((n, n), n * n, dtype=torch.long)
        for k in range(n):
            for a in range(k + 1):
                idx[k, a] = a * n + (k - a)
        _IDX[key] = idx.to(device)
    return _IDX[key]


def log_esp(d: torch.Tensor, D: int) -> torch.Tensor:
    """d (..., p) -> (..., D + 1): log e_0 .. log e_D of exp(d), by a log-space tree product (one gather and one
    logsumexp per level)."""
    shape, p = d.shape[:-1], d.shape[-1]
    d = d.reshape(-1, p)
    B, n = d.shape[0], D + 1
    if D == 0:
        return torch.zeros(*shape, 1, dtype=d.dtype, device=d.device)
    L = torch.cat([torch.zeros_like(d).unsqueeze(-1), d.unsqueeze(-1),
                   torch.full((B, p, D - 1), NEG, dtype=d.dtype, device=d.device)], -1)
    idx = _antidiag_index(D, d.device)
    while L.shape[1] > 1:
        if L.shape[1] % 2:
            one = torch.full((B, 1, n), NEG, dtype=d.dtype, device=d.device)
            one[..., 0] = 0.0
            L = torch.cat([L, one], 1)
        M = (L[:, 0::2].unsqueeze(-1) + L[:, 1::2].unsqueeze(-2)).flatten(-2)        # (B, m, n*n)
        M = torch.cat([M, torch.full_like(M[..., :1], NEG)], -1)                   # pad slot
        L = torch.logsumexp(M[..., idx], -1)                                       # (B, m, n)
    return L[:, 0].reshape(*shape, n)


class ExactStructuredSpikeAndSlabPrior(StructuredSpikeAndSlabPrior):
    """Drop-in replacement for GEMSS's StructuredSpikeAndSlabPrior with the exact density (see module docstring)."""

    exact = True

    def __init__(self, n_features: int, sparsity: int, var_slab: float = 100.0, var_spike: float = 0.1):
        super().__init__(n_features, sparsity, sample_more_priors_coeff=1.0, var_slab=var_slab, var_spike=var_spike)
        self._all_supports = None
        self.log_comb = math.lgamma(n_features + 1) - math.lgamma(sparsity + 1) - math.lgamma(n_features - sparsity + 1)

    def _parts(self, z):
        sl, ss = float(self.var_slab), float(self.var_spike)          # used as standard deviations, as GEMSS does
        spike = -0.5 * (z / ss) ** 2 - math.log(ss) - 0.5 * _LOG2PI
        slab = -0.5 * (z / sl) ** 2 - math.log(sl) - 0.5 * _LOG2PI
        return spike, slab - spike, sl, ss

    def log_prob(self, z: torch.Tensor, n_support_samples: int = 0) -> torch.Tensor:
        spike, delta, _, _ = self._parts(z)
        return spike.sum(-1) + log_esp(delta, self.sparsity)[..., self.sparsity] - self.log_comb

    def inclusion(self, z: torch.Tensor) -> torch.Tensor:
        """pi_j = P(j in S | z), shape of z."""
        with torch.enable_grad():
            d = self._parts(z.detach())[1].requires_grad_(True)
            le = log_esp(d, self.sparsity)[..., self.sparsity]
            (pi,) = torch.autograd.grad(le.sum(), d)
        return pi

    def terms(self, z: torch.Tensor):
        """(log p, gradient, Hessian diagonal) in closed form, for the ngvi estimators."""
        spike, delta, sl, ss = self._parts(z.detach())
        with torch.enable_grad():                                       # one forward pass for log e_D and pi
            d = delta.requires_grad_(True)
            le = log_esp(d, self.sparsity)[..., self.sparsity]
            (pi,) = torch.autograd.grad(le.sum(), d)
        lp = spike.sum(-1) + le.detach() - self.log_comb
        d1 = -z / sl ** 2 + z / ss ** 2
        d2 = -1.0 / sl ** 2 + 1.0 / ss ** 2
        g = -z / ss ** 2 + pi * d1
        H = -1.0 / ss ** 2 + pi * d2 + d1 ** 2 * pi * (1 - pi)
        return lp, g, H


def make_exact(sel):
    """Replace a GEMSS selector's StructuredSpikeAndSlabPrior by the exact one (same D and scales). In place."""
    P = sel.prior
    if not isinstance(P, StructuredSpikeAndSlabPrior):
        raise ValueError("exact prior: the selector does not use the structured spike-and-slab prior")
    sel.prior = ExactStructuredSpikeAndSlabPrior(P.n_features, P.sparsity, var_slab=P.var_slab, var_spike=P.var_spike)
    return sel
