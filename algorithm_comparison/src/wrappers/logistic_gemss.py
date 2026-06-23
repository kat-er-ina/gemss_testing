"""Logistic (Bernoulli) variant of the GEMSS feature selector.

The shipped GEMSS core (``gemss.feature_selection.inference.BayesianFeatureSelector``)
uses a single Gaussian / L2 likelihood for *every* task. In "classification" mode
the binary labels y in {0, 1} are therefore fitted by linear regression, not by
logistic regression -- contrary to the model stated in the paper (sigmoid link +
cross-entropy).

This module provides a drop-in subclass that overrides only the likelihood with a
proper Bernoulli model

    log p(y | z, X) = sum_i [ y_i * log sigma(x_i . z) + (1 - y_i) * log(1 - sigma(x_i . z)) ]

so that the classification experiments actually exercise the logistic link. Keeping
both selectors available lets the stress test *measure* whether the fix changes
support recovery, rather than silently swapping the model.

Missing values are handled exactly as in the parent: features with NaN are filled
with zero (so they contribute nothing to the logit x_i . z) and samples with no
observed feature are masked out of the sum.
"""

from typing import Any
import torch
import torch.nn.functional as F

from gemss.feature_selection.inference import BayesianFeatureSelector

from .gemss_wrapper import GEMSSWrapper


class LogisticBayesianFeatureSelector(BayesianFeatureSelector):
    """GEMSS selector with a Bernoulli (logistic) likelihood instead of L2.

    Only the likelihood is changed; the prior, the Gaussian-mixture variational
    posterior, the ELBO, the Jaccard regularization and the optimizer are inherited
    unchanged from :class:`BayesianFeatureSelector`.
    """

    def log_likelihood(self, z: torch.Tensor) -> torch.Tensor:
        """Bernoulli log-likelihood log p(y | z, X), summed over samples.

        Parameters
        ----------
        z : torch.Tensor
            Batch of parameter samples, shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Log-likelihood per batch sample, shape (batch_size,).
        """
        if self._has_missing_X:
            return self._log_likelihood_with_missing(z)

        logits = torch.matmul(z, self.X.T)  # [batch_size, n_samples]
        target = self.y.unsqueeze(0).expand_as(logits)
        # BCE-with-logits returns the negative log-likelihood per element;
        # negate and sum over samples to recover the log-likelihood.
        nll = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return -nll.sum(dim=-1)

    def _log_likelihood_with_missing(self, z: torch.Tensor) -> torch.Tensor:
        """Bernoulli log-likelihood with missing features in X.

        Mirrors the parent's masking strategy: NaNs in X are replaced by zero
        (no contribution to the logit), and samples with no observed feature
        (or a NaN label) are dropped from the sum.
        """
        batch_size = z.shape[0]
        assert self._valid_sample_mask is not None
        assert self._X_filled is not None

        valid_mask = self._valid_sample_mask & ~torch.isnan(self.y)
        if not valid_mask.any():
            return torch.zeros(batch_size, device=z.device)

        logits = torch.matmul(z, self._X_filled.T)  # [batch_size, n_samples]
        target = torch.nan_to_num(self.y, nan=0.0).unsqueeze(0).expand_as(logits)
        nll = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        log_likes = -nll  # [batch_size, n_samples]

        mask = valid_mask.to(dtype=log_likes.dtype, device=log_likes.device)
        return (log_likes * mask.unsqueeze(0)).sum(dim=1)


class LogisticGEMSSWrapper(GEMSSWrapper):
    """Benchmark wrapper that runs GEMSS with the Bernoulli likelihood.

    Identical to :class:`GEMSSWrapper` except for the underlying selector class.
    For ``task='regression'`` it behaves like plain GEMSS (the L2 likelihood is
    the correct model there), so it is only meaningfully different for
    classification.
    """

    SELECTOR_CLS = LogisticBayesianFeatureSelector

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # For regression there is no logistic link; fall back to the L2 selector
        # so this wrapper is a no-op relative to GEMSS on continuous targets.
        if self.task == "regression":
            self.SELECTOR_CLS = BayesianFeatureSelector
