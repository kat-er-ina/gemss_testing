"""Abstract base class for method wrappers.

This module defines the interface that all feature selection method wrappers
must implement. It ensures consistent input/output formats across different
algorithms for fair benchmarking.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class ModelWrapper(ABC):
    """
    Abstract base class for feature selection algorithm wrappers.

    All feature selection methods must inherit from this class and implement
    the fit() method. This ensures a consistent interface for benchmarking
    different algorithms.

    Methods
    -------
    fit(X, y)
        Fit the feature selection model and return standardized solutions.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, List[int]]]:
        """
        Fit the feature selection model and return multiple solutions.

        This method must be implemented by all subclasses. It should fit the
        feature selection algorithm to the data and return one or more feature
        subsets (solutions) in a standardized dictionary format.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target variable of shape (n_samples,). Can be continuous for
            regression or discrete for classification.

        Returns
        -------
        Dict[str, Dict[str, List[int]]]
            Dictionary mapping solution identifiers to solution dictionaries.
            Each solution dictionary must contain at minimum:

            - 'support' : List[int]
                Zero-based indices of selected features.
        """
        pass
