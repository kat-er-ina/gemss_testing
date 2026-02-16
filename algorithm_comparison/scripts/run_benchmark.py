"""Main entry point for running benchmarks."""

import os
import sys
import yaml
import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_factory import get_benchmark_data
from src.wrappers.gemss_wrapper import GEMSSWrapper
from src.wrappers.alfese_wrapper import AlfeseWrapper
from src.evaluation import calculate_metrics, get_empty_metrics


def main():
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
            "ALFESE_mi_tau1.0",
            "ALFESE_greedy_tau1.0",
            "ALFESE_importance_tau1.0",
            # "ALFESE_mrmr_tau1.0", # extreme run times, esp. for p > 1000
            # "ALFESE_fcbf_tau1.0", # memory issues for p > 1000, very long run times, esp. for p > 1000
        ],
        help="List of methods to run (e.g., GEMSS ALFESE_mrmr_tau1.0). Default: GEMSS + a quick ALFESE-MI variants with tau=1.0",
    )
    args = parser.parse_args()

    # Load Config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "benchmark_config.yaml",
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Filter experiments based on command-line argument
    all_experiments = config["experiments"]
    if args.experiments:
        # Convert to set of integers for faster lookup
        try:
            selected_exp_numbers = set(int(e) for e in args.experiments)
        except ValueError as e:
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_log = []
    # Prepare output directory and file for incremental saving
    os.makedirs(config["output_dir"], exist_ok=True)
    output_file = os.path.join(
        config["output_dir"], f"benchmark_results_{timestamp}.csv"
    )
    header_written = False

    # Pre-calculate which methods are available for each experiment
    experiment_method_map = {}
    for exp in experiments:
        exp_id = exp["id"]
        exp_number = exp.get("exp_number", "N/A")
        available_methods = [m for m in args.methods if m in exp["methods"].keys()]
        if available_methods:
            experiment_method_map[exp_number] = available_methods

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
            # Generate Data
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

            # Get shared parameters
            n_desired_solutions = experiment.get("n_desired_solutions", 6)
            desired_sparsity = experiment.get("desired_sparsity", 5)

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
                    f"  SKIP: No requested methods configured for this experiment"
                )

            # Instantiate and run each method
            for method_name, method_params in method_configs.items():
                pbar.set_postfix_str(f"Exp {exp_number}: {method_name}")

                try:
                    t_start = pd.Timestamp.now()

                    # Instantiate model
                    if method_name == "GEMSS":
                        model = GEMSSWrapper(
                            task=task_type,
                            n_components=n_desired_solutions,
                            sparsity=desired_sparsity,  # Pass shared sparsity parameter
                            **method_params,
                        )

                    elif method_name.startswith("ALFESE_"):
                        model = AlfeseWrapper(
                            n_solutions=n_desired_solutions,
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

                    # Evaluate
                    metrics = calculate_metrics(
                        predicted_solutions_dict=solutions,
                        true_support_indices=true_support,
                        p_total=ds_params["n_features"],
                    )

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

    print(f"\n✅ Benchmark Complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total experiments run: {len(results_log)}")


if __name__ == "__main__":
    main()
