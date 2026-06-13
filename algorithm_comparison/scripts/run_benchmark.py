"""Main entry point for running feature selection benchmarks.

This script executes benchmarking experiments comparing GEMSS and ALFESE feature
selection algorithms on synthetic datasets with known ground truth. Results are
saved incrementally to CSV files for analysis.

The script supports:
- Running all experiments or a subset via command-line arguments
- Running all methods or specific methods
- Configurable experimental parameters via YAML config file
- Progress tracking with detailed status messages
- Incremental result saving (resilient to interruptions)
- Automatic error handling and logging

Command-Line Usage
------------------
Run all experiments with default methods::

    uv run python run_benchmark.py

Run specific experiments only::

    uv run python run_benchmark.py -e 1 2 3

Run all experiments with GEMSS only::

    uv run python run_benchmark.py -m GEMSS

Run specific experiment with specific methods::

    uv run python run_benchmark.py -e 1 -m GEMSS ALFESE_mrmr_tau1.0

Default Methods
---------------
If -m/--methods is not specified, the following methods are run:

- GEMSS
- ALFESE_mi_tau1.0
- ALFESE_greedy_tau1.0
- ALFESE_importance_tau1.0

Note: ALFESE_mrmr and ALFESE_fcbf have extreme runtimes and memory usage for
p > 1000, so they are excluded from defaults.

Configuration
-------------
Experimental parameters are defined in:
``algorithm_comparison/configs/benchmark_config.yaml``

Results
-------
Results are saved to:
``algorithm_comparison/results/benchmark_results_<timestamp>.csv``

Each row contains:
- Experiment metadata (ID, description, parameters)
- Dataset characteristics (n_samples, n_features, sparsity, etc.)
- Method name and runtime
- Evaluation metrics (Recall, Precision, F1, Success Index, etc.)
- Status (Success or Failed with error message)

See Also
--------
algorithm_comparison.src.data_factory : Data generation utilities
algorithm_comparison.src.evaluation : Metric calculation functions
algorithm_comparison.src.wrappers : Algorithm wrapper implementations
"""

import os
import sys
import copy
import yaml
import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_factory import get_benchmark_data
from src.hard_data_factory import generate_overlapping_dataset
from src.wrappers.gemss_wrapper import GEMSSWrapper
from src.wrappers.logistic_gemss import LogisticGEMSSWrapper
from src.wrappers.alfese_wrapper import AlfeseWrapper
from src.wrappers.sklearn_wrappers import MaskingWrapper, StabilitySelectionWrapper
from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper
from src.evaluation import (
    calculate_metrics,
    calculate_structural_metrics,
    predictive_quality,
    get_empty_metrics,
)


def main() -> None:
    """
    Execute the benchmarking suite for feature selection algorithms.

    This function orchestrates the complete benchmarking workflow:

    1. Parse command-line arguments for experiment and method selection
    2. Load experimental configuration from YAML file
    3. Filter experiments based on user selection
    4. For each experiment:

       a. Generate synthetic dataset with known ground truth
       b. Run each selected method (GEMSS, ALFESE variants)
       c. Evaluate solutions against ground truth
       d. Save results incrementally to CSV

    5. Provide progress tracking and error handling

    The function creates a timestamped output CSV file and writes results
    incrementally as experiments complete. This ensures partial results are
    preserved if the script is interrupted.

    Command-Line Arguments
    ----------------------
    -e, --experiments : List[int], optional
        Experiment numbers to run (e.g., 1 2 3). If not specified, runs all
        experiments defined in the config file.
    -m, --methods : List[str], optional
        Method names to run. If not specified, uses default set:
        ['GEMSS', 'ALFESE_mi_tau1.0', 'ALFESE_greedy_tau1.0',
         'ALFESE_importance_tau1.0']

    Raises
    ------
    SystemExit
        If invalid experiment numbers are provided or no matching experiments
        are found in the configuration.

    Notes
    -----
    Each experiment-method combination is tracked with a progress bar.
    Failed experiments are logged with error messages but do not stop execution.

    The benchmark uses stratified metrics that compare the union of found features
    across all solutions to the union of ground truth features (Rashomon set).

    Output CSV columns include:
    - Exp_Number, Experiment_ID, Experiment_Description
    - Method, Task, Runtime_Sec, Status
    - Dataset parameters: n_samples, n_features, sparsity, noise_std, etc.
    - Evaluation metrics: Recall, Precision, F1_Score, Success_Index, etc.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run GEMSS benchmarking experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_benchmark.py                          # Run all experiments with default methods
  python run_benchmark.py -e 1 2 3                 # Run only experiments 1, 2, 3
  python run_benchmark.py -m GEMSS                 # Run all experiments with GEMSS only
  python run_benchmark.py -e 1 -m GEMSS ALFESE_mrmr_tau0.3  # Run experiment 1 with specific methods
        """,
    )
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="+",
        default=None,
        help="List of experiment numbers to run (e.g., 1 2 3). Default: all experiments",
    )
    parser.add_argument(
        "-m",
        "--methods",
        nargs="+",
        default=[
            "GEMSS",
            "GEMSS_noreg",
            "GEMSS_logistic",
            "GEMSS_single",
            "GEMSS_kernel",
            "GEMSS_scalefixed",
            "Masking_lasso",
            "Masking_elasticnet",
            "Masking_logistic",
            "StabilitySelection",
            "ALFESE_mi_tau1.0",
            "ALFESE_greedy_tau1.0",
            "ALFESE_importance_tau1.0",
            # "ALFESE_mrmr_tau1.0", # extreme run times, esp. for p > 1000
            # "ALFESE_fcbf_tau1.0", # memory issues for p > 1000, very long run times, esp. for p > 1000
        ],
        help="List of methods to run. Default: GEMSS variants + masking/stability baselines + quick ALFESE variants. Only methods configured for a given experiment are run.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="benchmark_config.yaml",
        help="Config filename under algorithm_comparison/configs/ (default: benchmark_config.yaml; use stresstest_config.yaml / hardtest_config.yaml).",
    )
    parser.add_argument(
        "-s",
        "--seeds",
        nargs="+",
        default=None,
        help="Optional list of seeds (e.g. 42 7 123). Each experiment is repeated once per seed (overriding the config seed) for robustness.",
    )
    parser.add_argument(
        "-b",
        "--budget",
        type=int,
        default=None,
        help="Override the shared solution budget (n_solutions) for ALL experiments, e.g. 3 to match the true number of solutions (tight budget, no over-selection).",
    )
    args = parser.parse_args()

    # Load Config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        args.config,
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Filter experiments based on command-line argument
    all_experiments = config["experiments"]
    if args.experiments:
        # Convert to set of integers for faster lookup
        try:
            selected_exp_numbers = set(int(e) for e in args.experiments)
        except ValueError:
            print(f"ERROR: Experiment numbers must be integers: {args.experiments}")
            sys.exit(1)

        filtered_experiments = [
            exp
            for exp in all_experiments
            if exp.get("exp_number") in selected_exp_numbers
        ]
        if not filtered_experiments:
            print(f"ERROR: No experiments found matching: {args.experiments}")
            print(
                f"Available experiment numbers: {[exp.get('exp_number') for exp in all_experiments]}"
            )
            sys.exit(1)
        experiments = filtered_experiments
    else:
        experiments = all_experiments

    # Expand over seeds (robustness): one copy of each experiment per seed.
    if args.seeds:
        try:
            seed_list = [int(s) for s in args.seeds]
        except ValueError:
            print(f"ERROR: Seeds must be integers: {args.seeds}")
            sys.exit(1)
        expanded = []
        for exp in experiments:
            for sd in seed_list:
                e = copy.deepcopy(exp)
                e["dataset"] = dict(e["dataset"])
                e["dataset"]["seed"] = sd
                e["id"] = f"{exp['id']}_s{sd}"
                expanded.append(e)
        experiments = expanded
        print(f"Seed expansion: {len(seed_list)} seeds -> {len(experiments)} runs")

    # Include PID so concurrent runs (e.g. parallel SLURM jobs submitted in the
    # same second) never collide on the same output filename.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_log = []
    # Prepare output directory and file for incremental saving
    os.makedirs(config["output_dir"], exist_ok=True)
    output_file = os.path.join(
        config["output_dir"], f"benchmark_results_{timestamp}_pid{os.getpid()}.csv"
    )
    header_written = False

    # Pre-calculate which methods are available for each experiment.
    # Keyed by unique id (not exp_number) so seed-expanded copies don't collide.
    experiment_method_map = {}
    for exp in experiments:
        exp_id = exp["id"]
        available_methods = [m for m in args.methods if m in exp["methods"].keys()]
        if available_methods:
            experiment_method_map[exp_id] = available_methods

    # Calculate total number of experiment-method combinations for progress tracking
    total_combinations = sum(len(methods) for methods in experiment_method_map.values())

    print(f"Starting Benchmark: {config['experiment_name']}")
    print(
        f"Total experiments to run: {len(experiments)} (out of {len(all_experiments)} total)"
    )
    print(f"Requested methods: {', '.join(args.methods)}")
    print(
        f"Experiments with available methods: {len(experiment_method_map)}/{len(experiments)}"
    )
    print(f"Total experiment-method combinations: {total_combinations}")
    print(f"Output Directory: {config['output_dir']}\n")

    # Show which methods will run for each experiment
    if len(experiment_method_map) < len(experiments):
        skipped = len(experiments) - len(experiment_method_map)
        print(
            f"Note: {skipped} experiment(s) skipped (no requested methods configured)\n"
        )

    # Create progress bar for all experiment-method combinations
    pbar = tqdm(
        total=total_combinations,
        desc="Benchmark progress",
        unit="run",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    # Iterate over experiments
    for exp_idx, experiment in enumerate(experiments, 1):
        exp_id = experiment["id"]
        exp_number = experiment.get("exp_number", "N/A")
        description = experiment["description"]

        pbar.write(f"\n[{exp_idx}/{len(experiments)}] Exp {exp_number}: {exp_id}")
        pbar.write(f"  {description}")

        # Get dataset parameters
        ds_params = experiment["dataset"]
        task_type = "classification" if ds_params["binarize"] else "regression"

        try:
            # Generate Data. `generator: overlap` uses the honest/hard generator
            # with overlapping valid solutions + equal-variance features; the
            # default reproduces the original linear-Gaussian benchmark.
            generator = experiment.get("generator", "linear")
            planted_supports = None  # set only by the overlap generator
            if generator == "overlap":
                X, y, true_support, planted_supports = generate_overlapping_dataset(
                    n_samples=ds_params["n_samples"],
                    n_features=ds_params["n_features"],
                    n_solutions=ds_params["n_generating_solutions"],
                    sparsity=ds_params["sparsity"],
                    latent_rank=ds_params.get("latent_rank", 2),
                    overlap=ds_params.get("overlap", 2),
                    noise_std=ds_params["noise_std"],
                    nan_ratio=ds_params["nan_ratio"],
                    binarize=ds_params["binarize"],
                    binary_response_ratio=ds_params["binary_response_ratio"],
                    seed=ds_params["seed"],
                )
            else:
                X, y, true_support = get_benchmark_data(
                    n_samples=ds_params["n_samples"],
                    n_features=ds_params["n_features"],
                    n_generating_solutions=ds_params["n_generating_solutions"],
                    sparsity=ds_params["sparsity"],
                    noise_std=ds_params["noise_std"],
                    nan_ratio=ds_params["nan_ratio"],
                    binarize=ds_params["binarize"],
                    binary_ratio=ds_params["binary_response_ratio"],
                    seed=ds_params["seed"],
                )

            # Get shared parameters.
            # `n_solutions` is the FAIR shared solution budget used by every
            # multi-solution method (stress-test config). The legacy keys
            # gemss_n_solutions / alfese_n_solutions are honoured as fallbacks so
            # the original benchmark_config.yaml still reproduces unchanged.
            # --budget overrides the per-experiment shared budget (e.g. 3 to
            # match the true solution count -> no over-selection slack).
            shared_budget = args.budget if args.budget is not None else experiment.get("n_solutions")
            gemss_n_solutions = (
                shared_budget
                if shared_budget is not None
                else experiment.get("gemss_n_solutions", 6)
            )
            alfese_n_solutions = (
                shared_budget
                if shared_budget is not None
                else experiment.get("alfese_n_solutions", 6)
            )
            desired_sparsity = experiment.get("desired_sparsity", 5)
            seed = ds_params.get("seed", 42)

            # Get method configurations and filter to only requested methods available for this experiment
            all_method_configs = experiment["methods"]
            method_configs = {
                method_name: method_params
                for method_name, method_params in all_method_configs.items()
                if method_name in args.methods
            }

            # Skip experiments with no available methods (expected behavior)
            if not method_configs:
                pbar.write(
                    "  SKIP: No requested methods configured for this experiment"
                )

            # Instantiate and run each method
            for method_name, method_params in method_configs.items():
                pbar.set_postfix_str(f"Exp {exp_number}: {method_name}")

                try:
                    t_start = pd.Timestamp.now()

                    # Instantiate model
                    if method_name in ("GEMSS", "GEMSS_noreg", "GEMSS_jaccard"):
                        model = GEMSSWrapper(
                            task=task_type,
                            n_components=gemss_n_solutions,
                            sparsity=desired_sparsity,  # Pass shared sparsity parameter
                            **method_params,
                        )

                    elif method_name == "GEMSS_logistic":
                        # Proper Bernoulli likelihood (the "classification fix").
                        model = LogisticGEMSSWrapper(
                            task=task_type,
                            n_components=gemss_n_solutions,
                            sparsity=desired_sparsity,
                            **method_params,
                        )

                    elif method_name == "GEMSS_single":
                        # Single-Gaussian ablation: no mixture, one mode only.
                        single_params = {
                            k: v for k, v in method_params.items() if k != "n_components"
                        }
                        model = GEMSSWrapper(
                            task=task_type,
                            n_components=1,
                            sparsity=desired_sparsity,
                            **single_params,
                        )

                    elif method_name in ("GEMSS_kernel", "GEMSS_scalefixed"):
                        mech = "kernel" if method_name.endswith("kernel") else "scalefixed"
                        model = MechanismGEMSSWrapper(
                            task=task_type,
                            mechanism=mech,
                            n_components=gemss_n_solutions,
                            sparsity=desired_sparsity,
                            **method_params,
                        )

                    elif method_name.startswith("Masking_"):
                        model = MaskingWrapper(
                            estimator=method_params.get("estimator", "lasso"),
                            n_solutions=gemss_n_solutions,
                            sparsity=desired_sparsity,
                            task=task_type,
                            alpha=method_params.get("alpha", 0.01),
                            seed=seed,
                        )

                    elif method_name == "StabilitySelection":
                        model = StabilitySelectionWrapper(
                            sparsity=desired_sparsity,
                            task=task_type,
                            alpha=method_params.get("alpha", 0.02),
                            n_bootstrap=method_params.get("n_bootstrap", 100),
                            threshold=method_params.get("threshold", 0.6),
                            seed=seed,
                        )

                    elif method_name.startswith("ALFESE_"):
                        model = AlfeseWrapper(
                            n_solutions=alfese_n_solutions,
                            task=task_type,
                            selector_type=method_params["selector_type"],
                            tau=method_params["tau"],
                            k=desired_sparsity,
                        )

                    else:
                        pbar.write(f"    {method_name}: SKIP (unknown method)")
                        pbar.update(1)
                        continue

                    # Fit model (this is where the time is spent)
                    with tqdm(
                        total=0,
                        desc=f"    Running {method_name}",
                        bar_format="{desc}: {elapsed}",
                        leave=False,
                        position=1,
                    ) as inner_pbar:
                        solutions = model.fit(X, y)

                    duration = (pd.Timestamp.now() - t_start).total_seconds()

                    # Validate solutions returned
                    if not solutions:
                        raise ValueError("Model returned no solutions")

                    # Evaluate: union metrics always; per-solution structural
                    # metrics when the planted supports are known (overlap gen).
                    metrics = calculate_metrics(
                        predicted_solutions_dict=solutions,
                        true_support_indices=true_support,
                        p_total=ds_params["n_features"],
                    )
                    if planted_supports is not None:
                        metrics.update(
                            calculate_structural_metrics(solutions, planted_supports)
                        )
                    # Predictive quality of recovered solutions (field's currency;
                    # near-constant on synthetic-equivalent data, informative on real).
                    try:
                        metrics.update(predictive_quality(X, y, solutions, task_type))
                    except Exception:
                        metrics.update({"pred_mean": float("nan"), "pred_best": float("nan")})

                    # Log results
                    entry = {
                        "Exp_Number": exp_number,
                        "Experiment_ID": exp_id,
                        "Experiment_Description": description,
                        "Method": method_name,
                        "Task": task_type,
                        **ds_params,
                        **metrics,
                        "Runtime_Sec": duration,
                        "Status": "Success",
                    }
                    results_log.append(entry)
                    pbar.write(
                        f"    {method_name}: OK (ASI={metrics.get('Adjusted_SI', 0):.2f}, {duration:.1f}s)"
                    )
                    pbar.update(1)

                    # Save result to CSV immediately (append mode)
                    df_entry = pd.DataFrame([entry])
                    if not header_written:
                        df_entry.to_csv(output_file, mode="w", index=False)
                        header_written = True
                    else:
                        df_entry.to_csv(
                            output_file, mode="a", index=False, header=False
                        )

                except Exception as e:
                    pbar.write(f"    {method_name}: FAIL ({str(e)})")
                    pbar.update(1)
                    metrics = get_empty_metrics()
                    fail_entry = {
                        "Exp_Number": exp_number,
                        "Experiment_ID": exp_id,
                        "Experiment_Description": description,
                        "Method": method_name,
                        "Task": task_type,
                        **ds_params,
                        **metrics,
                        "Runtime_Sec": None,
                        "Status": f"Failed: {str(e)}",
                    }
                    results_log.append(fail_entry)
                    # Save failed result to CSV immediately (append mode)
                    df_entry = pd.DataFrame([fail_entry])
                    if not header_written:
                        df_entry.to_csv(output_file, mode="w", index=False)
                        header_written = True
                    else:
                        df_entry.to_csv(
                            output_file, mode="a", index=False, header=False
                        )

        except Exception as data_err:
            pbar.write(f"  CRITICAL: Data generation failed: {data_err}")
            continue

    # Close progress bar
    pbar.close()

    print("\n✅ Benchmark Complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total experiments run: {len(results_log)}")


if __name__ == "__main__":
    main()
