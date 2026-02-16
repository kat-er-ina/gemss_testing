"""Adapter for GEMSS package."""

from .base import ModelWrapper
from gemss.feature_selection.inference import BayesianFeatureSelector
from gemss.postprocessing.result_postprocessing import recover_solutions


class GEMSSWrapper(ModelWrapper):
    def __init__(
        self,
        task="regression",
        n_components=6,
        sparsity=None,
        lambda_jaccard=0,
        regularize=False,
        **kwargs,
    ):
        """
        Initialize GEMSS wrapper.

        Args:
            task: 'regression' or 'classification'
            n_components: Number of alternative solutions to find
            sparsity: Desired sparsity level (used for solution recovery)
            lambda_jaccard: Diversity penalty strength for Jaccard regularization
            regularize: Whether to apply Jaccard diversity penalty
            **kwargs: Additional parameters for BayesianFeatureSelector
        """
        self.task = task
        self.n_components = n_components
        self.sparsity = sparsity
        self.lambda_jaccard = lambda_jaccard
        self.regularize = regularize
        self.params = kwargs

    def fit(self, X, y):
        # 1. Initialize
        # Note: We filter kwargs to avoid passing parameters GEMSS doesn't recognize
        # if necessary, but typically GEMSS ignores unknown kwargs or we clean them.
        selector = BayesianFeatureSelector(
            n_features=X.shape[1],
            n_components=self.n_components,
            X=X,
            y=y,
            **self.params,
        )

        # 2. Optimize
        history = selector.optimize(
            regularize=self.regularize,
            lambda_jaccard=self.lambda_jaccard,
            verbose=False,
        )

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

        # Convert and limit to requested number of components
        results = {}
        solution_count = 0
        for comp_name, df in top_solutions.items():
            if solution_count >= self.n_components:
                break

            # Convert GEMSS DataFrame format to simple list of indices
            if df.empty:
                results[comp_name] = {"support": []}
            else:
                # df has 'Feature' column like 'feature_12'
                indices = [int(f.split("_")[1]) for f in df["Feature"].values]
                results[comp_name] = {"support": indices}

            solution_count += 1

        # Ensure at least one solution is returned even if empty
        if not results:
            results["component_0"] = {"support": []}

        return results
