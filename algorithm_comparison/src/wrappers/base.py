"""Abstract base class for method wrappers."""

from abc import ABC, abstractmethod
import numpy as np


class ModelWrapper(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fits the model and returns a standardized dictionary of solutions.

        Returns:
            dict: {
                'solution_0': {'support': [0, 1, 5], 'weights': [...]},
                'solution_1': {'support': [0, 2, 6], 'weights': [...]},
                ...
            }
        """
        pass
