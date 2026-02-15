"""Source code for the benchmark.

This package contains the core functionality for benchmarking feature selection methods.

Modules:
    data_factory: Data generation utilities wrapping GEMSS data handling
    evaluation: Metrics for evaluating feature selection performance
    wrappers: Adapters for different feature selection methods
"""

from .data_factory import get_benchmark_data
from .evaluation import calculate_metrics, get_empty_metrics

__all__ = [
    "get_benchmark_data",
    "calculate_metrics",
    "get_empty_metrics",
]
