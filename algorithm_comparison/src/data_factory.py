"""Wraps gemss.data_handling for standardized generation."""

from gemss.data_handling.generate_artificial_dataset import generate_artificial_dataset


def get_benchmark_data(
    n_samples,
    n_features,
    n_generating_solutions,
    sparsity,
    noise_std,
    seed,
    nan_ratio=0.0,
    binarize=False,
    binary_ratio=0.5,
):
    """
    Wrapper to generate data and extract the "Ground Truth" union of features.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_generating_solutions: Number of generating solutions
        sparsity: Sparsity level
        noise_std: Noise standard deviation
        seed: Random seed
        nan_ratio: Ratio of NaN values (default: 0.0)
        binarize: Whether to binarize the response (default: False)
        binary_ratio: Ratio for binary split if binarize=True (default: 0.5)

    Returns:
        tuple: (X, y, true_support_indices)
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
    # params_df has 'support_indices' column
    all_true_indices = set()
    for indices in params_df["support_indices"]:
        all_true_indices.update(indices)

    return df.values, y.values, all_true_indices
