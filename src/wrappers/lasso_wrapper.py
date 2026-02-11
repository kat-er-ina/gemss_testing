"""Adapter for sklearn/glmnet."""

from .base import ModelWrapper
from sklearn.linear_model import LassoCV, LogisticRegressionCV
import numpy as np


class LassoWrapper(ModelWrapper):
    def __init__(self, task="regression", random_state=42):
        self.task = task
        self.random_state = random_state

    def fit(self, X, y):
        # Select correct model based on task
        if self.task == "classification":
            # Logistic Regression with L1 penalty
            model = LogisticRegressionCV(
                cv=5,
                penalty="l1",
                solver="liblinear",
                random_state=self.random_state,
            )
        else:
            # Standard LASSO
            model = LassoCV(cv=5, random_state=self.random_state)

        model.fit(X, y)

        # Extract coefficients
        if self.task == "classification":
            coef = model.coef_[0]
        else:
            coef = model.coef_

        support = np.where(np.abs(coef) > 1e-5)[0].tolist()

        return {"solution_0": {"support": support}}
