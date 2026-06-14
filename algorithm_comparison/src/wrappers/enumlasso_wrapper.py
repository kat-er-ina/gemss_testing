"""Wrapper for EnumLasso (Hara & Maehara, AAAI 2017) -- a principled, cheap,
direct competitor for multiple sparse solutions that GEMSS/ALFESE never compared
against. It enumerates the k best Lasso supports (which MAY overlap), with a
recovery guarantee under weaker conditions than ordinary Lasso.

Uses the authors' vendored reference implementation (src/external, MIT). We
standardize features and take the top-``sparsity`` features of each enumerated
solution (by |coef|) so it is scored on the same footing as the other methods.
"""

from typing import Any, Dict, List
import warnings

import numpy as np

from .base import ModelWrapper
from .sklearn_wrappers import _prepare

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from external.enumerate_linear_model import EnumLasso  # noqa: E402


class EnumLassoWrapper(ModelWrapper):
    """Enumerate-Lasso: k best (possibly overlapping) Lasso supports."""

    def __init__(self, n_solutions: int = 6, sparsity: int = 5, task: str = "classification",
                 rho: float = 0.02, seed: int = 42, **kwargs: Any) -> None:
        self.n_solutions = n_solutions
        self.sparsity = sparsity
        self.task = task
        self.rho = rho
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, List[int]]]:
        Xs = _prepare(X)
        y = np.asarray(y).ravel().astype(float)
        modeltype = "classification" if self.task == "classification" else "regression"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl = EnumLasso(modeltype=modeltype, rho=self.rho, enumtype="k",
                            k=self.n_solutions, verbose=0)
            try:
                mdl.fit(Xs, y)
            except Exception:
                return {"solution_0": {"support": []}}
        results: Dict[str, Dict[str, List[int]]] = {}
        for i, a in enumerate(getattr(mdl, "a_", [])):
            a = np.abs(np.asarray(a).ravel())
            if not np.any(a > 0):
                continue
            supp = [int(j) for j in np.argsort(a)[::-1][: self.sparsity] if a[int(j)] > 0]
            if supp:
                results[f"solution_{i}"] = {"support": supp}
        return results or {"solution_0": {"support": []}}
