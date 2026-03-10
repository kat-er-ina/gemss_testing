"""Adapter for GEMSS package.

This wrapper provides a standardized interface to the GEMSS feature selection
algorithm, which uses Bayesian inference to discover multiple alternative sparse
solutions (the Rashomon set).
"""

from typing import Dict, List, Optional, Any
import numpy as np
from .base import ModelWrapper
from gemss.feature_selection.inference import BayesianFeatureSelector
from gemss.postprocessing.result_postprocessing import recover_solutions


class GEMSSWrapper(ModelWrapper):
    """
    Wrapper for the GEMSS feature selection algorithm.

    Parameters
    ----------
    task : str, optional
        Task type: 'regression' or 'classification', by default 'regression'.
    n_components : int, optional
        Number of alternative solutions to find, by default 6.
    sparsity : int, optional
        Desired sparsity level (number of features) for solution recovery,
        by default None. If None, uses the value from kwargs or defaults to 5.
    lambda_jaccard : float, optional
        Diversity penalty strength for Jaccard regularization, by default 0.
        Higher values encourage more diverse solutions.
    regularize : bool, optional
        Whether to apply Jaccard diversity penalty during optimization,
        by default False.
    **kwargs : dict
        Additional parameters passed to BayesianFeatureSelector.

    Attributes
    ----------
    task : str
        The task type (regression or classification).
    n_components : int
        Number of solutions to find.
    sparsity : Optional[int]
        Target sparsity level.
    lambda_jaccard : float
        Diversity penalty coefficient.
    regularize : bool
        Whether diversity regularization is enabled.
    params : dict
        Additional parameters for the GEMSS selector.
    """

    def __init__(
        self,
        task: str = "regression",
        n_components: int = 6,
        sparsity: Optional[int] = None,
        lambda_jaccard: float = 0,
        regularize: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GEMSS wrapper with specified parameters."""
        self.task = task
        self.n_components = n_components
        self.sparsity = sparsity
        self.lambda_jaccard = lambda_jaccard
        self.regularize = regularize
        self.params = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, List[int]]]:
        """
        Fit GEMSS and return multiple sparse feature solutions.

        This method runs the GEMSS optimization procedure to discover multiple
        diverse sparse solutions, then recovers the top solutions at the specified
        sparsity level.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target variable of shape (n_samples,).

        Returns
        -------
        Dict[str, Dict[str, List[int]]]
            Dictionary mapping component names (e.g., 'component_0') to solution
            dictionaries. Each solution contains:

            - 'support' : List[int]
                Zero-based indices of selected features.
        """
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
