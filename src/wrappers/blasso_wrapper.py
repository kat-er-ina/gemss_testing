"""Adapter for Bayesian Ridge/LASSO."""

from .base import ModelWrapper
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler


class BLASSOWrapper(ModelWrapper):
    def __init__(self, threshold_std=2.0, task="regression", **kwargs):
        """
        Args:
            threshold_std: Num std devs for credible interval (2.0 ~= 95%)
            task: 'regression' or 'classification'
            **kwargs: Arguments for BayesianRidge
        """
        self.threshold_std = threshold_std
        self.task = task
        self.model = BayesianRidge(**kwargs)
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        # 1. Standardize
        X_scaled = self.scaler.fit_transform(X)

        # 2. Fit Model
        # Note: For classification, we treat y as continuous (Linear Probability Model)
        # This is a common fast approximation for feature selection benchmarks
        # when a full Bayesian Logistic Regression sampler is too slow.
        self.model.fit(X_scaled, y)

        # 3. Extract Posterior Statistics
        coef_mean = self.model.coef_
        coef_var = np.diag(self.model.sigma_)
        coef_std = np.sqrt(coef_var)

        # 4. Determine Support (Credible Interval)
        # Select if 0 is NOT in [mean - k*std, mean + k*std]
        lower_bound = np.abs(coef_mean) - (self.threshold_std * coef_std)
        mask = lower_bound > 0

        support_indices = np.where(mask)[0].tolist()

        return {
            "solution_0": {
                "support": support_indices,
                "weights": coef_mean[support_indices].tolist(),
            }
        }
