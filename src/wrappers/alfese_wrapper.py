"""Adapter for ALFESE package."""

from .base import ModelWrapper
import alfese
import pandas as pd
import numpy as np


class AlfeseWrapper(ModelWrapper):
    def __init__(
        self,
        n_solutions=3,
        task="regression",
        selector_type="mrmr",
        tau=0.5,
        k=None,
        n_iter=None,
        **kwargs,
    ):
        """
        Args:
            n_solutions: Number of alternative solutions to find
            task: 'regression' or 'classification'
            selector_type: Type of ALFESE selector ('mrmr', 'mi', 'fcbf')
            tau: Diversity parameter (0-1, higher means more diverse)
            k: Number of features to select per solution (defaults to 20% of features)
            n_iter: Maximum number of iterations (None uses package default)
            **kwargs: Extra args for ALFESE selector
        """
        self.n_solutions = n_solutions
        self.task = task
        self.selector_type = selector_type.lower()
        self.tau = tau
        self.k = k
        self.n_iter = n_iter
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        Fit ALFESE feature selector.

        Note: ALFESE has a complex API requiring data setup and solver initialization.
        This wrapper provides a simplified interface.
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
