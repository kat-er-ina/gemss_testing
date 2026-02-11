"""Adapters to make all models look identical.

This package provides a unified interface for different feature selection methods
through the ModelWrapper base class.

Classes:
    ModelWrapper: Abstract base class for all method wrappers
    GEMSSWrapper: Adapter for GEMSS package
    AlfeseWrapper: Adapter for ALFESE package
    LassoWrapper: Adapter for sklearn LASSO
    BLASSOWrapper: Adapter for Bayesian Ridge/LASSO
"""

from .base import ModelWrapper
from .gemss_wrapper import GEMSSWrapper
from .alfese_wrapper import AlfeseWrapper
from .lasso_wrapper import LassoWrapper
from .blasso_wrapper import BLASSOWrapper

__all__ = [
    "ModelWrapper",
    "GEMSSWrapper",
    "AlfeseWrapper",
    "LassoWrapper",
    "BLASSOWrapper",
]
