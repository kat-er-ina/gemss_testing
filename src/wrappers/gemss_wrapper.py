"""Adapter for GEMSS package."""

from .base import ModelWrapper
from gemss.feature_selection.inference import BayesianFeatureSelector
from gemss.postprocessing.result_postprocessing import recover_solutions


class GEMSSWrapper(ModelWrapper):
    def __init__(self, task="regression", sparsity=None, **kwargs):
        """
        Initialize GEMSS wrapper.

        Args:
            task: 'regression' or 'classification'
            sparsity: Desired sparsity level (used for solution recovery)
            **kwargs: Additional parameters for BayesianFeatureSelector
        """
        self.task = task
        self.sparsity = sparsity
        self.params = kwargs

    def fit(self, X, y):
        # 1. Initialize
        # Note: We filter kwargs to avoid passing parameters GEMSS doesn't recognize
        # if necessary, but typically GEMSS ignores unknown kwargs or we clean them.
        selector = BayesianFeatureSelector(
            n_features=X.shape[1],
            X=X,
            y=y,
            **self.params,
        )

        # 2. Optimize
        history = selector.optimize(verbose=False)

        # 3. Recover Solutions
        # Use the explicit sparsity passed from the runner if available
        target_sparsity = (
            self.sparsity
            if self.sparsity is not None
            else self.params.get("sparsity", 5)
        )

        _, top_solutions, _, _ = recover_solutions(
            history, desired_sparsity=target_sparsity
        )

        results = {}
        for comp_name, df in top_solutions.items():
            # Convert GEMSS DataFrame format to simple list of indices
            if df.empty:
                results[comp_name] = {"support": []}
            else:
                # df has 'Feature' column like 'feature_12'
                indices = [int(f.split("_")[1]) for f in df["Feature"].values]
                results[comp_name] = {"support": indices}

        return results
