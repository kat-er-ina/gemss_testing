"""Wraps gemss.data_handling for standardized generation.

This module provides simplified data generation for benchmarking feature selection
algorithms. It wraps the GEMSS artificial dataset generation functionality and
extracts the ground truth feature support (Rashomon set).
"""

from typing import Tuple, Set
import numpy as np
from gemss.data_handling.generate_artificial_dataset import generate_artificial_dataset


def get_benchmark_data(
    n_samples: int,
    n_features: int,
    n_generating_solutions: int,
    sparsity: int,
    noise_std: float,
    seed: int,
    nan_ratio: float = 0.0,
    binarize: bool = False,
    binary_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Set[int]]:
    """
    Generate artificial benchmark dataset with known ground truth features.

    This function generates synthetic data using the GEMSS artificial dataset
    generator and extracts the union of all true features (Rashomon set) that
    were used to generate the data. This ground truth set is used for evaluating
    feature selection algorithms.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Total number of features in the dataset.
    n_generating_solutions : int
        Number of alternative generating solutions (equivalent predictive models).
    sparsity : int
        Number of features per generating solution.
    noise_std : float
        Standard deviation of Gaussian noise added to the response variable.
    seed : int
        Random seed for reproducibility.
    nan_ratio : float, optional
        Ratio of missing values (NaN) to introduce in features, by default 0.0.
        Must be between 0 and 1.
    binarize : bool, optional
        Whether to convert the continuous response to binary classification,
        by default False.
    binary_ratio : float, optional
        Controls the split point for binarization if binarize=True, by default 0.5.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Response variable of shape (n_samples,).
    true_support_indices : Set[int]
        Set of zero-based feature indices that comprise the ground truth
        Rashomon set (union of all generating solutions).
    """
    df, y, gen_solutions, params_df = generate_artificial_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_solutions=n_generating_solutions,
        sparsity=sparsity,
        noise_data_std=noise_std,
        random_seed=seed,
        nan_ratio=nan_ratio,
        binarize=binarize,
        binary_response_ratio=binary_ratio,
        print_data_overview=False,
    )

    # Extract the union of ALL true features (Rashomon set)
    # params_df has 'support_indices' column containing the support for each solution
    all_true_indices: Set[int] = set()
    for indices in params_df["support_indices"]:
        all_true_indices.update(indices)

    return df.values, y.values, all_true_indices
