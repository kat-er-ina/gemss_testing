"""
Run GEMSS (Gaussian Ensemble for Multiple Sparse Solutions) as a script.

This script provides a complete experiment pipeline including:
- Dataset generation with configurable parameters
- GEMSS optimization for sparse feature selection
- Solution recovery and analysis
- Performance diagnostics
- Comprehensive result reporting
- CSV logging of experiment metrics

Assumes:
- gemss package is installed and available
- gemss.config is properly set up with experiment parameters

Outputs:
- A text file with comprehensive experiment results in the "results" directory
- A CSV file logging experiment metrics in the "results" directory
"""

import csv
import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Set, Tuple, Any, Optional

import gemss.config as C
from gemss.data_handling.generate_artificial_dataset import generate_artificial_dataset
from gemss.feature_selection.inference import BayesianFeatureSelector
from gemss.postprocessing.result_postprocessing import (
    recover_solutions,
    get_features_from_solutions,
    get_unique_features,
)
from gemss.postprocessing.simple_regressions import solve_any_regression
from gemss.diagnostics.performance_tests import run_performance_diagnostics
from gemss.utils.utils import dataframe_to_ascii_table, get_solution_summary_df


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the GEMSS experiment script.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments containing:
        - output: Optional custom output filename
        - diagnostics: Boolean flag to run performance diagnostics
    """
    parser = argparse.ArgumentParser(
        description="Run GEMSS experiment for sparse feature selection."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom filename for output summary (should end with .txt)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run performance diagnostics (default: False)",
    )
    return parser.parse_args()


def generate_dataset() -> Tuple[pd.DataFrame, Any, Dict, Dict, List[str]]:
    """
    Generate artificial dataset for the experiment using configuration parameters.

    Returns
    -------
    tuple
        A 5-tuple containing:
        - pd.DataFrame: Feature matrix with shape (n_samples, n_features)
        - array-like: Target variable (binary or continuous)
        - dict: Generating solutions used to create the data
        - dict: Dataset parameters including support indices and coefficients
        - list[str]: True support feature names in format ['feature_0', 'feature_1', ...]
    """
    print("Generating dataset with:")
    print(f" - {C.N_SAMPLES} samples")
    print(f" - {C.N_FEATURES} features")
    print(f" - noise std: {C.NOISE_STD}")
    print(f" - ratio of missing values: {C.NAN_RATIO}")
    if C.BINARIZE:
        print(
            f" - binary classification problem with {C.BINARY_RESPONSE_RATIO} ratio of positive labels"
        )
    else:
        print(" - regression problem")
    print(
        f" - {C.N_GENERATING_SOLUTIONS} original solutions, each with {C.SPARSITY} supporting vectors"
    )

    df, y, generating_solutions, parameters = generate_artificial_dataset(
        n_samples=C.N_SAMPLES,
        n_features=C.N_FEATURES,
        n_solutions=C.N_GENERATING_SOLUTIONS,
        sparsity=C.SPARSITY,
        noise_data_std=C.NOISE_STD,
        nan_ratio=C.NAN_RATIO,
        binarize=C.BINARIZE,
        binary_response_ratio=C.BINARY_RESPONSE_RATIO,
        random_seed=C.DATASET_SEED,
        save_to_csv=False,
        print_data_overview=False,
    )

    support_indices = parameters["support_indices"].sum()
    true_support_features = [f"feature_{i}" for i in set(support_indices)]

    return df, y, generating_solutions, parameters, true_support_features


def run_feature_selection(df: pd.DataFrame, y: Any) -> Tuple[Any, List[Dict], Tuple]:
    """
    Run GEMSS optimization for sparse feature selection.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with shape (n_samples, n_features)
    y : array-like
        Target variable (binary classification or regression)

    Returns
    -------
    tuple
        A 3-tuple containing:
        - BayesianFeatureSelector: Trained GEMSS selector instance
        - list[dict]: Optimization history with loss and parameter traces
        - tuple: Solution data (full_solutions, top_solutions, outlier_solutions, final_parameters)
    """
    print("\nRunning GEMSS...")

    selector = BayesianFeatureSelector(
        n_features=C.N_FEATURES,
        n_components=C.N_CANDIDATE_SOLUTIONS,
        X=df.values,
        y=y,
        prior=C.PRIOR_TYPE,
        sss_sparsity=C.PRIOR_SPARSITY,
        sample_more_priors_coeff=C.SAMPLE_MORE_PRIORS_COEFF,
        var_slab=C.VAR_SLAB,
        var_spike=C.VAR_SPIKE,
        weight_slab=C.WEIGHT_SLAB,
        weight_spike=C.WEIGHT_SPIKE,
        student_df=C.STUDENT_DF,
        student_scale=C.STUDENT_SCALE,
        lr=C.LEARNING_RATE,
        batch_size=C.BATCH_SIZE,
        n_iter=C.N_ITER,
    )

    history = selector.optimize(
        regularize=C.IS_REGULARIZED,
        lambda_jaccard=C.LAMBDA_JACCARD,
        verbose=False,
    )
    print("Optimization finished.")

    full_solutions, top_solutions, outlier_solutions, final_parameters = (
        recover_solutions(
            search_history=history,
            desired_sparsity=C.DESIRED_SPARSITY,
            min_mu_threshold=C.MIN_MU_THRESHOLD,
            use_median_for_outlier_detection=C.USE_MEDIAN_FOR_OUTLIER_DETECTION,
            outlier_deviation_thresholds=C.OUTLIER_DEVIATION_THRESHOLDS,
        )
    )

    return (
        selector,
        history,
        (full_solutions, top_solutions, outlier_solutions, final_parameters),
    )


def analyze_feature_discovery(
    solutions: Dict,
    true_support_features: List[str],
) -> Tuple[Set, Set, Set]:
    """
    Analyze feature discovery performance for any set of candidate solutions.

    Parameters
    ----------
    solutions : dict
        Solutions by component (can be full, top, or outlier solutions).
        Each key is a component name, each value is a DataFrame with 'Feature' column.
    true_support_features : list[str]
        True support feature names from the generating process

    Returns
    -------
    tuple
        A 3-tuple of sets containing:
        - set: Features found by the algorithm
        - set: True support features that were missed
        - set: Extra features found that are not in true support
    """
    true_support_features = set(true_support_features)

    # Analyze the provided solutions
    features = get_features_from_solutions(solutions)
    features_found = set(get_unique_features(features))
    missing_features = true_support_features - features_found
    extra_features = features_found - true_support_features

    return (
        features_found,
        missing_features,
        extra_features,
    )


def run_diagnostics(history: List[Dict]) -> Optional[List[Dict]]:
    """
    Run performance diagnostics on optimization history.

    Parameters
    ----------
    history : list[dict]
        Optimization history containing loss values, parameter traces,
        and convergence information from the GEMSS optimization

    Returns
    -------
    list[dict] or None
        Performance test results with status, messages, and component details,
        or None if diagnostics failed to run
    """
    print("Running performance diagnostics...")
    try:
        diagnostics = run_performance_diagnostics(
            history=history, desired_sparsity=C.DESIRED_SPARSITY, verbose=False
        )
        performance_results = diagnostics.test_results
        print(f"Performance tests completed: {len(performance_results)} tests run")
        return performance_results
    except Exception as e:
        print(f"Performance diagnostics failed: {e}")
        return None


def format_parameters_section() -> List[str]:
    """
    Format the parameters section for experiment output.

    Returns
    -------
    list[str]
        Lines of text containing the experiment title and all configuration
        parameters from gemss.config formatted as markdown
    """
    lines = ["# GEMSS Experiment Results\n"]
    lines.append("## Parameters and Settings\n")
    params = C.as_dict()
    for k, v in params.items():
        lines.append(f"- {k}: {v}")
    return lines


def format_feature_discovery_overview(
    solution_type: str,
    feature_analysis: Tuple[Set, Set, Set],
    true_support_features: List[str],
) -> List[str]:
    """
    Format the overview of discovered and missed features in a set of candidate solution.

    Parameters
    ----------
    solution_type : str
        Type of solutions being analyzed ('full', 'top', or an outlier variant)
    feature_analysis : tuple[set, set, set]
        Results from analyze_feature_discovery: (found, missing, extra) features
    true_support_features : list[str]
        True support feature names from the generating process

    Returns
    -------
    list[str]
        Formatted text lines summarizing feature discovery performance
        including counts and lists of found, missing, and extra features
    """
    features_found, missing_features, extra_features = feature_analysis

    lines = [
        f"\n## Overview of discovered features for {solution_type.upper()} solutions:\n"
    ]
    lines.append(f" - {len(true_support_features)} unique true support features:")
    lines.append(f"   {sorted(true_support_features)}\n")

    # Solution type summary
    lines.append(f"\n - {len(features_found)} discovered features:")
    lines.append(f"   {sorted(features_found)}")

    lines.append(f"\n - {len(missing_features)} missed true support features:")
    lines.append(f"   {sorted(missing_features)}")

    lines.append(
        f"\n - {len(extra_features)} extra features found (not in true support):"
    )
    lines.append(f"   {sorted(extra_features)}")

    return lines


def format_solution_type_header(
    solution_type: str,
) -> List[str]:
    """
    Format the header section for a specific solution type.

    Parameters
    ----------
    solution_type : str
        Type of solutions ('full', 'top', or outlier variant)

    Returns
    -------
    list[str]
        Formatted header lines explaining the solution type and its parameters
    """
    solution_label = solution_type.upper()
    lines = []
    lines.append(
        f"   ------   ANALYSIS - Solution type: {solution_type.upper()}   ------   "
    )
    lines.append(f"\n## {solution_label} Solutions\n")

    if solution_type == "full":
        lines.append(f"All features with |mu| > {C.MIN_MU_THRESHOLD}\n")
    elif solution_type == "top":
        lines.append(f"Required sparsity = {C.DESIRED_SPARSITY}\n")
    elif "outlier" in solution_type:
        devtype = (
            "standard deviation"
            if "STD" in solution_label
            else "median absolute deviation" if "MAD" in solution_label else "deviation"
        )
        lines.append(f"Features identified as outliers based on {devtype}.\n")
    return lines


def format_features_in_components(
    solutions: Dict,
) -> List[str]:
    """
    Format the detailed breakdown of features within each component.

    Parameters
    ----------
    solutions : dict
        Solutions by component where each key is component name and
        each value is DataFrame with 'Feature' and 'Mu value' columns

    Returns
    -------
    list[str]
        Formatted text lines showing features and their mu values
        organized by component
    """
    lines = []
    lines.append(f"\n\n## Features in components")
    for component, df in solutions.items():
        lines.append(f"\n### {component.upper()} ({df.shape[0]} features):")
        for i, row in df.iterrows():
            lines.append(f" - {row['Feature']}: mu = {row['Mu value']:.4f}")
        lines.append("")

    return lines


def format_performance_diagnostics(
    performance_results: Optional[List[Dict]],
) -> List[str]:
    """
    Format the performance diagnostics section for experiment output.

    Parameters
    ----------
    performance_results : list[dict] or None
        Results from run_diagnostics containing test statuses and messages,
        or None if diagnostics failed

    Returns
    -------
    list[str]
        Formatted text lines showing diagnostic test results with
        summary counts and detailed failure/warning information
    """
    lines = ["\n## Performance Diagnostics\n"]

    if not performance_results:
        lines.append("Performance diagnostics could not be completed.")
        lines.append("")
        return lines

    # Summary counts
    status_counts = {"FAILED": 0, "WARNING": 0, "PASSED": 0}
    for result in performance_results:
        status_counts[result["status"]] += 1

    lines.append(
        f"Total tests: {len(performance_results)} | "
        f"Failed: {status_counts['FAILED']} | "
        f"Warnings: {status_counts['WARNING']} | "
        f"Passed: {status_counts['PASSED']}\n"
    )

    # Results by status (failed first, then warnings, then passed)
    status_symbols = {"FAILED": "[FAIL]", "WARNING": "[WARN]", "PASSED": "[PASS]"}

    for status in ["FAILED", "WARNING", "PASSED"]:
        status_tests = [r for r in performance_results if r["status"] == status]
        if not status_tests:
            continue

        lines.append(f"\n{status} TESTS:")
        lines.append("-" * 40)

        for result in status_tests:
            lines.append(f"{status_symbols[status]} {result['test_name']}")
            lines.append(f"Message: {result['message']}")

            # Add component details for failed/warning tests
            if (
                status in ["FAILED", "WARNING"]
                and "component_details" in result["details"]
            ):
                problem_components = [
                    f"Component {comp['component']} ({comp['component_status']})"
                    for comp in result["details"]["component_details"]
                    if comp["component_status"] != "PASSED"
                ]

                if problem_components:
                    lines.append(f"Problem components: {', '.join(problem_components)}")

            lines.append("")  # Empty line between tests

    lines.append("")
    return lines


def determine_output_path(custom_output: Optional[str]) -> Path:
    """
    Determine the output file path for experiment results.

    Parameters
    ----------
    custom_output : str or None
        Custom output filename provided by user, or None for automatic naming

    Returns
    -------
    pathlib.Path
        Full path where experiment results will be written.
        Uses timestamp-based naming if custom_output is None.
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    if custom_output:
        output_path = Path(custom_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        parent_dir = Path(os.path.dirname(os.getcwd()))
        results_dir = parent_dir / "results"
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / f"experiment_output_{timestamp}.txt"

    return output_path


def write_results(output_lines: List[str], output_path: Path) -> None:
    """
    Write experiment results to file.

    Parameters
    ----------
    output_lines : list[str]
        All formatted text lines to write to the output file
    output_path : pathlib.Path
        Full path where the results file will be created

    Returns
    -------
    None
    """
    print("Writing results...")
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"Experiment summary written to {output_path}")


def calculate_coverage_metrics(
    features_found: List[str] | Set[str],
    true_support_features: List[str] | Set[str],
    n_total_features: int,
) -> dict:
    """
    Calculates coverage metrics based on the GEMSS Design of Experiments.

    Parameters
    ----------
    features_found : List[str] | Set[str]
        Set of feature names identified by the algorithm (F)
    true_support_features : List[str] | Set[str]
        Set of ground truth feature names (P_generating)
    n_total_features : int
        Total number of features in the dataset (p)
    """
    # Convert to sets if lists are provided
    if isinstance(features_found, list):
        features_found = set(features_found)
    if isinstance(true_support_features, list):
        true_support_features = set(true_support_features)

    # Calculate counts [cite: 91]
    f = len(features_found)
    p_generating = len(true_support_features)
    p = n_total_features

    # Intersection (Correctly identified)
    features_correct = features_found.intersection(true_support_features)
    f_correct = len(features_correct)

    # Missed (False Negatives)
    features_missed = true_support_features - features_found
    f_missed = len(features_missed)

    # Extra (False Positives)
    features_extra = features_found - true_support_features
    f_extra = len(features_extra)

    # --- Metrics Calculation  ---

    # 1. Recall (Sensitivity)
    recall = f_correct / p_generating if p_generating > 0 else None

    # 2. Precision
    precision = (
        f_correct / f if f > 0 else None
    )  # Default to 1 if nothing selected (conservative)

    # 3. F1-score
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = None

    # 4. Jaccard Similarity (Intersection over Union)
    union_size = len(features_found.union(true_support_features))
    jaccard = f_correct / union_size if union_size > 0 else None

    # 5. Miss Rate
    miss_rate = f_missed / p_generating if p_generating > 0 else None

    # 6. False Discovery Rate (FDR)
    fdr = f_extra / f if f > 0 else None

    # 7. Global Miss Rate
    global_miss_rate = f_missed / p if p > 0 else None

    # 8. Global FDR
    global_fdr = f_extra / p if p > 0 else None

    # 9. Success Index (SI)
    # Formula: (p * f_correct) / (p_generating^2)
    if p_generating > 0:
        si = (p * f_correct) / (p_generating**2)
    else:
        si = None

    # 10. Adjusted Success Index (ASI)
    # Formula: SI * Precision
    asi = si * precision

    return {
        "n_features_found": f,
        "n_correct": f_correct,
        "n_missed": f_missed,
        "n_extra": f_extra,
        "Recall": recall,
        "Precision": precision,
        "F1_Score": f1,
        "Jaccard": jaccard,
        "Miss_Rate": miss_rate,
        "FDR": fdr,
        "Global_Miss_Rate": global_miss_rate,
        "Global_FDR": global_fdr,
        "Success_Index": si,
        "Adjusted_Success_Index": asi,
    }


def main() -> None:
    """
    Main experiment pipeline for GEMSS sparse feature selection.

    Executes the complete workflow:
    1. Parse command line arguments
    2. Generate artificial dataset with known ground truth
    3. Run GEMSS optimization to find sparse solutions
    4. Analyze feature discovery performance for all solution types and compute metrics
    5. Generate comprehensive results report
    6. Run diagnostics if requested
    7. Write text results to timestamped output file
    8. Append results (parameters + coverage metrics) to a centralized CSV file
    """
    # Define the global order for coverage metrics (for CSV consistency)
    COVERAGE_METRIC_ORDER = [
        "n_features_found",
        "n_correct",
        "n_missed",
        "n_extra",
        "Recall",
        "Precision",
        "F1_Score",
        "Jaccard",
        "Miss_Rate",
        "FDR",
        "Global_Miss_Rate",
        "Global_FDR",
        "Success_Index",
        "Adjusted_Success_Index",
    ]

    # Parse arguments
    args = parse_arguments()
    output_path = determine_output_path(args.output)

    # Generate dataset
    df, y, generating_solutions, parameters, true_support_features = generate_dataset()

    # Run feature selection
    selector, history, solution_data = run_feature_selection(df, y)
    full_solutions, top_solutions, outlier_solutions, final_parameters = solution_data

    # Initialize row for CSV logging with all config parameters
    experiment_results_row = C.as_dict().copy()

    # --- Add Timestamp ---
    experiment_results_row["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    # Write experiment parameters to text output
    output_lines = []
    output_lines.extend(format_parameters_section())
    output_lines.append(
        "\n----------------------------------------------------------------------\n"
    )

    # Analyze results for all solution types
    feature_discovery_analysis = {}
    all_solutions = {
        "full": full_solutions,
        "top": top_solutions,
    }
    for deviation, solutions in outlier_solutions.items():
        all_solutions[f"outlier ({deviation})"] = solutions

    # Prepare ordered lists for CSV fieldnames
    base_fields = list(C.as_dict().keys())  # Must match original definition order
    metric_fields = []

    for solution_type, solutions in all_solutions.items():
        # Get set of found features
        feature_discovery_analysis[solution_type] = analyze_feature_discovery(
            solutions, true_support_features
        )  # [found, missing, extra]
        features_found_set = feature_discovery_analysis[solution_type][0]

        # --- CALCULATE COVERAGE METRICS (Recall, SI, ASI, etc.) ---
        metrics = calculate_coverage_metrics(
            features_found=features_found_set,
            true_support_features=set(true_support_features),
            n_total_features=C.N_FEATURES,
        )

        # --- ADD METRICS TO TEXT OUTPUT ---
        output_lines.extend(format_solution_type_header(solution_type))

        # Add Coverage Metrics Table
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ["Value"]
        output_lines.extend(
            dataframe_to_ascii_table(
                metrics_df, title=f"Coverage Metrics for {solution_type}"
            )
        )

        # Add Feature Discovery Overview
        output_lines.extend(
            format_feature_discovery_overview(
                solution_type,
                feature_discovery_analysis[solution_type],
                true_support_features,
            )
        )

        # Detailed features in components
        summary_df = get_solution_summary_df(solutions)
        output_lines.extend(
            dataframe_to_ascii_table(
                summary_df,
                title=f"\n\nOverview of {solution_type.upper()} solutions",
            )
        )

        # Get regression results and convert them to ASCII table
        metrics_df = solve_any_regression(
            solutions=solutions,
            df=df,
            response=y,
            apply_scaling="standard",
            penalty="l2",
            verbose=False,
            use_markdown=False,
        )

        # Convert to ASCII table and add lines
        output_lines.extend(
            dataframe_to_ascii_table(
                metrics_df,
                title=(
                    f"\n\nRegression results on training data (l2 penalty)"
                    + "\n================================================"
                ),
            )
        )
        output_lines.append(
            "\n----------------------------------------------------------------------\n"
        )

        # --- ACCUMULATE METRICS FOR CSV LOG ---
        # Prefix keys with solution type to avoid column collision
        clean_type = solution_type.replace(" ", "_").replace("(", "").replace(")", "")
        for metric_name in COVERAGE_METRIC_ORDER:
            field_name = f"{clean_type}_{metric_name}"
            # Use the calculated value from the metrics dictionary
            experiment_results_row[field_name] = metrics.get(metric_name, np.nan)
            # Build the ordered list of field names once
            if field_name not in metric_fields:
                metric_fields.append(field_name)

    # Run diagnostics once for the entire optimization, if requested
    if args.diagnostics:
        performance_results = run_diagnostics(history)
        output_lines.extend(format_performance_diagnostics(performance_results))
        output_lines.append(
            "\n----------------------------------------------------------------------\n"
        )

    # Write text results
    write_results(output_lines, output_path)

    # --- WRITE CSV SUMMARY ---
    summary_csv_path = output_path.parent / "tier_summary_metrics.csv"

    # Add output filename to row for reference
    experiment_results_row["output_file"] = output_path.name

    # Construct the final ordered list of fieldnames for the CSV writer
    # [Config Params] + [Timestamp] + [Output File] + [Metric Fields (Ordered)]
    final_fields = base_fields + ["timestamp", "output_file"] + metric_fields

    # We must ensure the experiment_results_row contains all the necessary fieldnames
    # This is implicitly done above, but let's confirm padding for non-metric fields
    for field in final_fields:
        if field not in experiment_results_row:
            experiment_results_row[field] = np.nan  # Use np.nan for consistency

    file_exists = summary_csv_path.exists()
    try:
        with open(summary_csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=final_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(experiment_results_row)
        print(f"Metrics summary appended to {summary_csv_path}")
    except Exception as e:
        print(f"Failed to write CSV summary: {e}")


if __name__ == "__main__":
    main()
