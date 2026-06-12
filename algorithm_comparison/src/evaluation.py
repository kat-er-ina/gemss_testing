"""Evaluation metrics for the benchmarking suite.

This module provides metrics for comparing feature selection algorithm performance
against ground truth feature sets. Metrics focus on the union of discovered features
across all solutions (the discovered Rashomon set) compared to the true Rashomon set.
"""

from typing import Dict, List, Set, Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_metrics(
    predicted_solutions_dict: Dict[str, Dict[str, Any]],
    true_support_indices: Set[int],
    p_total: int,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics comparing predicted features to ground truth.

    This function compares the union of all features discovered across multiple
    solutions (predicted Rashomon set) against the union of true generating features
    (true Rashomon set). Multiple metrics are computed to assess recall, precision,
    and overall quality of the feature selection.

    Parameters
    ----------
    predicted_solutions_dict : Dict[str, Dict[str, Any]]
        Dictionary of predicted solutions. Each key is a solution identifier,
        and each value is a dict containing at minimum a 'support' key with
        a list of feature indices. Example:
        {
            'solution_0': {'support': [0, 1, 5]},
            'solution_1': {'support': [0, 2, 6]},
        }
    true_support_indices : Set[int]
        Set of zero-based indices representing the ground truth features
        (union of all generating solutions).
    p_total : int
        Total number of features in the dataset.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the following metrics:

        - **Recall** : Proportion of true features that were found
        - **Precision** : Proportion of found features that are true features
        - **F1_Score** : Harmonic mean of precision and recall
        - **Success_Index** : (p * correct) / (true²), scale-aware metric
        - **Adjusted_SI** : Success Index adjusted by precision
        - **Jaccard** : Jaccard similarity coefficient
        - **Solutions_Found** : Number of solutions returned
        - **Total_Features_Selected** : Size of predicted union
        - **n_correct** : Number of correctly identified features
        - **n_missed** : Number of true features not found
        - **n_extra** : Number of false positive features

    Notes
    -----
    The Success Index (SI) and Adjusted Success Index (ASI) are custom metrics
    designed to account for the total feature space size p_total. ASI further
    penalizes false positives by multiplying SI by precision.
    """
    # 1. Get Union of all predicted features across all solutions found
    predicted_union: Set[int] = set()
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


def _jaccard(a: Set[int], b: Set[int]) -> float:
    """Jaccard similarity of two index sets (empty/empty -> 1.0)."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def calculate_structural_metrics(
    predicted_solutions_dict: Dict[str, Dict[str, Any]],
    planted_supports: Dict[str, List[int]],
) -> Dict[str, float]:
    """Per-SOLUTION recovery metrics (not just the union of features).

    The union metrics (:func:`calculate_metrics`) cannot see whether a method
    reproduces the *structure* of overlapping solutions -- in particular the
    shared feature core that disjoint sequential masking is structurally unable
    to place in more than one solution. This function evaluates the recovered
    solution *sets* against the planted supports.

    Parameters
    ----------
    predicted_solutions_dict : dict
        Predicted solutions, each a dict with a 'support' list of indices.
    planted_supports : dict
        The ground-truth planted supports (e.g. from the overlap generator),
        each a list of feature indices. They may share a common core.

    Returns
    -------
    dict with keys
        - **sol_mean_jaccard** : mean over planted supports of the Jaccard with
          its one-to-one (Hungarian) matched predicted solution. Distinct
          predicted solutions are matched to distinct planted ones, so a method
          cannot reuse a single good solution to cover several planted ones.
        - **sol_recovered_0.8** / **sol_exact** : count of planted supports
          matched at Jaccard >= 0.8 / == 1.0.
        - **sol_core_size** : size of the shared core (intersection of all
          planted supports). 0 means disjoint (overlap=0).
        - **sol_preds_with_core** : number of *predicted* solutions that contain
          the full shared core. The discriminator for overlap handling: disjoint
          masking can have at most one; a mixture may have several.
        - **n_predicted** : number of non-empty predicted solutions.
    """
    planted = [set(s) for s in planted_supports.values()]
    predicted = [
        set(s.get("support", []))
        for s in predicted_solutions_dict.values()
        if s.get("support")
    ]
    core: Set[int] = set.intersection(*planted) if planted else set()

    n_p = len(planted)
    matched = np.zeros(n_p)
    if predicted:
        sim = np.array([[_jaccard(p, q) for q in predicted] for p in planted])
        rows, cols = linear_sum_assignment(-sim)  # maximize total Jaccard
        for i, j in zip(rows, cols):
            matched[i] = sim[i, j]

    preds_with_core = (
        sum(1 for q in predicted if core and core.issubset(q)) if core else 0
    )

    return {
        "sol_mean_jaccard": float(matched.mean()) if n_p else 0.0,
        "sol_recovered_0.8": int((matched >= 0.8).sum()),
        "sol_exact": int((matched >= 0.999).sum()),
        "sol_core_size": len(core),
        "sol_preds_with_core": int(preds_with_core),
        "n_predicted": len(predicted),
    }


def get_empty_metrics() -> Dict[str, float]:
    """
    Return metric structure with NaN values for failed experiments.

    This function is useful for maintaining consistent output structure when
    experiments fail or metrics cannot be calculated. All metric keys are
    present with NaN values.

    Returns
    -------
    Dict[str, float]
        Dictionary with same keys as calculate_metrics() but all values
        set to NaN (float('nan')).

    See Also
    --------
    calculate_metrics : Function that returns populated metrics

    Examples
    --------
    >>> empty = get_empty_metrics()
    >>> import math
    >>> math.isnan(empty['Recall'])
    True
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
