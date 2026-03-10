"""Adapter for ALFESE (ALternative FEature SElection) package.

This wrapper provides a standardized interface to the ALFESE feature selection
algorithm, which serves as a wrapper for several feature selection methods to discover
multiple alternative sparse solutions.

Supported Feature Selection Methods
-----------------------------------
ALFESE contains six feature-selection methods:

- **FCBFSelector**: Fast Correlation-Based Filter (FCBF), a multivariate filter
  method based on information theory.
- **GreedyWrapperSelector**: A wrapper method using a prediction model (default:
  decision tree) to evaluate feature subsets greedily.
- **MISelector**: Univariate filter method based on mutual information between
  features and target.
- **ModelImportanceSelector**: Univariate filter using feature importances from
  a prediction model (default: decision tree).
- **MRMRSelector**: Minimum Redundancy Maximum Relevance (mRMR), a multivariate
  filter balancing relevance and redundancy.
- **ManualUnivariateQualitySelector**: Manual specification of feature utilities
  (not used in benchmarking).

The feature-selection method determines the optimization objective for finding
diverse solutions.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import alfese
from .base import ModelWrapper


class AlfeseWrapper(ModelWrapper):
    """
    Wrapper for the ALFESE feature selection algorithm.

    Parameters
    ----------
    n_solutions : int, optional
        Number of alternative solutions to find, by default 3.
        ALFESE returns 1 primary solution plus (n_solutions-1) alternatives.
    task : str, optional
        Task type: 'regression' or 'classification', by default 'regression'.
    selector_type : str, optional
        Type of ALFESE selector, by default 'mrmr'. Valid options:

        - 'mrmr' : Minimum Redundancy Maximum Relevance
        - 'mi' : Mutual Information
        - 'fcbf' : Fast Correlation-Based Filter
        - 'greedy' : Greedy Wrapper (using prediction model)
        - 'importance' : Model-based feature importance

    tau : float, optional
        Diversity parameter between 0 and 1, by default 0.5.
        Higher values enforce more diversity between solutions.
        tau=0 means no diversity constraint (may return similar solutions).
        tau=1 means maximum diversity (solutions must be completely different).
    k : int, optional
        Number of features to select per solution, by default None.
        If None, defaults to 20% of total features.
    n_iter : int, optional
        Maximum number of iterations for the search algorithm, by default None.
        If None, uses ALFESE package default.
    **kwargs : dict
        Additional parameters passed to the ALFESE selector.

    Attributes
    ----------
    n_solutions : int
        Number of solutions to find.
    task : str
        Task type.
    selector_type : str
        Selected ALFESE method.
    tau : float
        Diversity parameter.
    k : Optional[int]
        Target number of features.
    n_iter : Optional[int]
        Maximum iterations.
    kwargs : dict
        Additional selector parameters.
    """

    def __init__(
        self,
        n_solutions: int = 3,
        task: str = "regression",
        selector_type: str = "mrmr",
        tau: float = 0.5,
        k: Optional[int] = None,
        n_iter: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ALFESE wrapper with specified parameters."""
        self.n_solutions = n_solutions
        self.task = task
        self.selector_type = selector_type.lower()
        self.tau = tau
        self.k = k
        self.n_iter = n_iter
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, List[int]]]:
        """
        Fit ALFESE selector and return multiple diverse feature solutions.

        This method converts data to pandas format (required by ALFESE), initializes
        the specified selector, and performs simultaneous search for diverse solutions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target variable of shape (n_samples,).

        Returns
        -------
        Dict[str, Dict[str, List[int]]]
            Dictionary mapping solution identifiers (e.g., 'solution_0') to solution
            dictionaries. Each solution contains:

            - 'support' : List[int]
                Zero-based indices of selected features.
        """
        # Convert to pandas if needed (ALFESE expects pandas)
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            X_df = X

        if not isinstance(y, pd.Series):
            y_series = pd.Series(y, name="target")
        else:
            y_series = y

        # Select appropriate ALFESE selector
        selector_map = {
            "mrmr": alfese.MRMRSelector,
            "mi": alfese.MISelector,
            "fcbf": alfese.FCBFSelector,
            "greedy": alfese.GreedyWrapperSelector,
            "importance": alfese.ModelImportanceSelector,
        }

        if self.selector_type not in selector_map:
            raise ValueError(
                f"Unknown selector_type '{self.selector_type}'. "
                f"Choose from: {list(selector_map.keys())}"
            )

        selector_class = selector_map[self.selector_type]
        selector = selector_class()

        # Set training and test data using proper ALFESE API
        # Use 80-20 split, but ensure minimum test set size for small datasets
        n_samples = len(X_df)
        if n_samples < 30:
            # For very small datasets, use larger training set
            split_idx = max(int(0.9 * n_samples), n_samples - 5)
        else:
            split_idx = int(0.8 * n_samples)

        X_train = X_df.iloc[:split_idx]
        X_test = X_df.iloc[split_idx:]
        y_train = y_series.iloc[:split_idx]
        y_test = y_series.iloc[split_idx:]

        # Use set_data() method to properly initialize selector
        selector.set_data(X_train, X_test, y_train, y_test)

        # Determine number of features to select (k)
        # Use provided k, or default to 20% of features
        if self.k is not None:
            k = self.k
        else:
            k = self.kwargs.get("k", max(1, int(0.2 * X_df.shape[1])))

        # Diversity parameter tau (0-1, higher means more diverse)
        tau = self.tau

        try:
            # Search for alternative solutions
            # Returns a DataFrame with columns: ['selected_idxs', 'train_objective', 'test_objective', ...]
            # Note: ALFESE returns 1 primary + num_alternatives, so we need n_solutions - 1
            search_kwargs = {
                "k": k,
                "num_alternatives": self.n_solutions - 1,
                "tau": tau,
                "objective_agg": "sum",
            }
            # Only pass max_iter if explicitly specified, otherwise use package default
            if self.n_iter is not None:
                search_kwargs["max_iter"] = self.n_iter

            result_df = selector.search_simultaneously(**search_kwargs)

            # Parse results into standardized format
            # ALFESE returns one row per solution with 'selected_idxs' column containing lists
            results = {}

            if "selected_idxs" in result_df.columns:
                # Each row is a solution, selected_idxs contains list of feature indices
                # ALFESE returns 1 primary + num_alternatives = n_solutions total
                num_solutions_returned = len(result_df)

                # Note: ALFESE may return fewer solutions if it cannot find enough diverse ones
                if num_solutions_returned < self.n_solutions:
                    print(
                        f"    INFO: ALFESE returned {num_solutions_returned}/{self.n_solutions} solutions (may indicate difficulty finding diverse solutions)"
                    )

                for i in range(num_solutions_returned):
                    selected_indices = result_df.iloc[i]["selected_idxs"]
                    # Convert to list if needed
                    if not isinstance(selected_indices, list):
                        selected_indices = list(selected_indices)
                    results[f"solution_{i}"] = {"support": selected_indices}

            # If no solutions found, return empty
            if not results:
                results["solution_0"] = {"support": []}

        except Exception as e:
            # If ALFESE fails, return empty solution with error info
            print(f"    ALFESE search failed: {e}")
            results = {"solution_0": {"support": []}}

        return results
