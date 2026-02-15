"""Evaluation metrics for the benchmarking suite."""


def calculate_metrics(
    predicted_solutions_dict,
    true_support_indices: set,
    p_total: int,
):
    """
    Compare the UNION of found features against the UNION of true features.

    Args:
        predicted_solutions_dict: Dict of solutions output by wrappers
        true_support_indices: Set of indices of the ground truth features
        p_total: Total number of features in the dataset

    Returns:
        dict: calculated metrics
    """
    # 1. Get Union of all predicted features across all solutions found
    predicted_union = set()
    for sol in predicted_solutions_dict.values():
        support = sol.get("support", [])
        predicted_union.update(support)

    # 2. Calculate Counts
    n_correct = len(predicted_union.intersection(true_support_indices))
    n_missed = len(true_support_indices - predicted_union)
    n_extra = len(predicted_union - true_support_indices)
    p_generating = len(true_support_indices)

    # 3. Metrics
    recall = n_correct / p_generating if p_generating > 0 else 0.0

    n_pred = len(predicted_union)
    precision = (
        n_correct / n_pred if n_pred > 0 else 1.0
    )  # Default to 1 if empty (conservative)
    f1score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Success Index (SI) = (p * Correct) / (True^2)
    si = (p_total * n_correct) / (p_generating**2) if p_generating > 0 else 0.0

    # Adjusted Success Index (ASI) = SI * Precision
    asi = si * precision

    # Jaccard
    union_size = len(predicted_union.union(true_support_indices))
    jaccard = n_correct / union_size if union_size > 0 else 0.0

    return {
        "Recall": recall,
        "Precision": precision,
        "F1_Score": f1score,
        "Success_Index": si,
        "Adjusted_SI": asi,
        "Jaccard": jaccard,
        "Solutions_Found": len(predicted_solutions_dict),
        "Total_Features_Selected": n_pred,
        "n_correct": n_correct,
        "n_missed": n_missed,
        "n_extra": n_extra,
    }


def get_empty_metrics():
    """
    Return the same metric structure as calculate_metrics but with NaN values.

    Useful for cases where metrics cannot be calculated (e.g., failed experiments).

    Returns:
        dict: metrics dictionary with NaN values
    """
    nan = float("nan")
    return {
        "Recall": nan,
        "Precision": nan,
        "F1_Score": nan,
        "Success_Index": nan,
        "Adjusted_SI": nan,
        "Jaccard": nan,
        "Solutions_Found": nan,
        "Total_Features_Selected": nan,
        "n_correct": nan,
        "n_missed": nan,
        "n_extra": nan,
    }
