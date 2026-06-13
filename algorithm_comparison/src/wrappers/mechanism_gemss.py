"""GEMSS with a validated anti-collapse mechanism, as a benchmark wrapper.

Makes the mechanisms studied in diagnostics/ runnable inside the main benchmark
(so the A/B sweep emits all the field-aligned + overlap-specialty metrics):

- mechanism='joint'      : standard GEMSS (no extra repulsion). Baseline.
- mechanism='kernel'     : parameter-space RBF repulsion on the component means
                           (SVGD / repulsive-mixtures; validated in validate_svgd.py),
                           the principled replacement for the support-space Jaccard
                           penalty that backfires on overlap.
- mechanism='scalefixed' : frozen log-spaced per-component variance ladder
                           (narrow specialists + broad explorer; best in the probe).

Likelihood tempering is deliberately NOT offered here: it is unvalidated
(validate_daem.py) and stays out of the trusted comparison.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from gemss.feature_selection.inference import BayesianFeatureSelector
from .base import ModelWrapper
from .logistic_gemss import LogisticBayesianFeatureSelector


def _kernel_repulsion(mu: torch.Tensor, bandwidth: Optional[float] = None) -> torch.Tensor:
    K = mu.shape[0]
    if K < 2:
        return mu.sum() * 0.0
    d2 = torch.cdist(mu, mu) ** 2
    iu = torch.triu_indices(K, K, offset=1, device=mu.device)
    off = d2[iu[0], iu[1]]
    h = bandwidth if bandwidth is not None else (off.detach().median() + 1e-8)
    return torch.exp(-off / h).mean()


class MechanismGEMSSWrapper(ModelWrapper):
    """Run GEMSS with a chosen anti-collapse mechanism; return top-D supports."""

    def __init__(
        self,
        task: str = "classification",
        mechanism: str = "joint",
        n_components: int = 6,
        sparsity: int = 5,
        kernel_gamma: float = 1000.0,
        scale_lo: float = 0.02,
        scale_hi: float = 2.0,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.mechanism = mechanism
        self.n_components = n_components
        self.sparsity = sparsity
        self.kernel_gamma = kernel_gamma
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi
        self.params = kwargs

    # keys the underlying selector __init__ accepts (others, e.g. lambda_jaccard
    # / regularize from shared config blocks, are ignored here)
    _SELECTOR_KEYS = {"prior", "sample_more_priors_coeff", "var_slab", "var_spike",
                      "weight_slab", "weight_spike", "student_df", "student_scale",
                      "lr", "batch_size", "n_iter", "device"}

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, List[int]]]:
        cls = LogisticBayesianFeatureSelector if self.task == "classification" else BayesianFeatureSelector
        sel_kwargs = {k: v for k, v in self.params.items() if k in self._SELECTOR_KEYS}
        sel = cls(n_features=X.shape[1], n_components=self.n_components, X=X, y=y,
                  sss_sparsity=self.sparsity, **sel_kwargs)

        if self.mechanism == "scalefixed":
            scales = np.geomspace(self.scale_lo, self.scale_hi, self.n_components).astype(np.float32)
            with torch.no_grad():
                for k in range(self.n_components):
                    sel.mixture._log_var[k].fill_(float(np.log(np.expm1(scales[k]))))
                sel.mixture._log_var.requires_grad_(False)

        if self.mechanism in ("joint", "scalefixed"):
            sel.optimize(regularize=False, lambda_jaccard=0.0, verbose=False)
        elif self.mechanism == "kernel":
            opt = sel.opt
            for _ in range(sel.n_iter):
                z, _c = sel.mixture.sample(sel.batch_size)
                obj = (sel.prior.log_prob(z) + sel.log_likelihood(z) - sel.mixture.log_prob(z)).mean()
                obj = obj - self.kernel_gamma * _kernel_repulsion(sel.mixture.mu)
                opt.zero_grad(); (-obj).backward(); opt.step()
        else:
            raise ValueError(f"unknown mechanism {self.mechanism}")

        mu = sel.mixture.mu.detach().cpu().numpy()
        D = self.sparsity
        results = {f"component_{k}": {"support": np.argsort(np.abs(mu[k]))[::-1][:D].tolist()}
                   for k in range(self.n_components)}
        return results
