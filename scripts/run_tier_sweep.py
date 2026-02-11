import sys
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_factory import get_benchmark_data
from src.evaluation import calculate_metrics
from src.wrappers.gemss_wrapper import GEMSSWrapper
from src.wrappers.alfese_wrapper import AlfeseWrapper
from src.wrappers.lasso_wrapper import LassoWrapper
from src.wrappers.blasso_wrapper import BLASSOWrapper


def parse_combination(combo_str, format_str):
    keys = format_str.split(",")
    values = combo_str.split(",")

    def convert(val):
        try:
            return int(val)
        except:
            try:
                return float(val)
            except:
                return val

    return {k: convert(v) for k, v in zip(keys, values)}


def run_sweep():
    # 1. Load Configurations
    config_dir = "configs"
    exp_config_path = os.path.join(config_dir, "experiment_config.yaml")
    method_config_path = os.path.join(config_dir, "method_params.yaml")

    if not os.path.exists(exp_config_path):
        print(f"Error: Config file not found at {exp_config_path}")
        return

    # Load Main Experiment Config
    with open(exp_config_path, "r") as f:
        config = yaml.safe_load(f)  # Renamed from exp_config to config

    # Load Method Hyperparameters
    method_defaults = {}
    if os.path.exists(method_config_path):
        with open(method_config_path, "r") as f:
            method_defaults = yaml.safe_load(f) or {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    print(f"Starting Benchmark Sweep: {config['experiment_name']}")
    print(f"Output Directory: {config['output_dir']}")

    # 2. Iterate Tiers
    for tier_key, tier_data in config["tiers"].items():
        print(f"\n{'='*60}")
        print(f"Running {tier_key}: {tier_data['name']}")

        # Global tier settings
        tier_algo_params = tier_data.get("algorithm_parameters", {})
        is_binary = tier_algo_params.get("BINARIZE", False)
        task_type = "classification" if is_binary else "regression"

        # 3. Iterate Combinations
        for combo_str in tier_data["combinations"]:
            # Parse specific parameters for this run (N, P, etc.)
            run_params = parse_combination(combo_str, config["parameter_format"])
            run_params.update(config.get("fixed_parameters", {}))

            print(
                f"\n  > Case: N={run_params['N_SAMPLES']}, P={run_params['N_FEATURES']}, "
                f"Sols={run_params['N_GENERATING_SOLUTIONS']}, Task={task_type}"
            )

            try:
                # 4. Generate Data
                X, y, true_support = get_benchmark_data(
                    n_samples=run_params["N_SAMPLES"],
                    n_features=run_params["N_FEATURES"],
                    n_generating_solutions=run_params["N_GENERATING_SOLUTIONS"],
                    sparsity=run_params["SPARSITY"],
                    noise_std=run_params["NOISE_STD"],
                    nan_ratio=run_params.get("NAN_RATIO", 0.0),
                    binarize=is_binary,
                    binary_ratio=run_params.get("BINARY_RESPONSE_RATIO", 0.5),
                    seed=run_params["DATASET_SEED"],
                )

                # 5. Define Models
                # Get active methods from config, default to all if not specified
                active_methods = config.get(
                    "active_methods", ["GEMSS", "LASSO", "B-LASSO", "ALFESE"]
                )
                models = {}

                # GEMSS
                if "GEMSS" in active_methods:
                    gemss_params = method_defaults.get("GEMSS", {}).copy()

                    # Parameter name mapping: config_name -> GEMSS_name
                    param_mapping = {
                        "prior_type": "prior",
                        "learning_rate": "lr",
                        "min_mu_threshold": None,  # Not a BayesianFeatureSelector param
                        "is_regularized": None,  # Not a BayesianFeatureSelector param
                        "binarize": None,  # Data generation param, not model param
                    }

                    # Convert tier_algo_params to lowercase and map to GEMSS params
                    tier_params_clean = {}
                    for k, v in tier_algo_params.items():
                        k_lower = k.lower()
                        # Skip if explicitly excluded
                        if k_lower in param_mapping and param_mapping[k_lower] is None:
                            continue
                        # Map to correct name
                        mapped_key = param_mapping.get(k_lower, k_lower)
                        tier_params_clean[mapped_key] = v

                    gemss_params.update(tier_params_clean)

                    # Remove parameters that are passed explicitly to avoid conflicts
                    gemss_params.pop("n_components", None)
                    gemss_params.pop("sparsity", None)

                    models["GEMSS"] = GEMSSWrapper(
                        n_components=run_params.get("N_CANDIDATE_SOLUTIONS", 6),
                        task=task_type,
                        sparsity=run_params["SPARSITY"],
                        **gemss_params,
                    )

                # LASSO
                if "LASSO" in active_methods:
                    lasso_params = method_defaults.get("LASSO", {}).copy()
                    models["LASSO"] = LassoWrapper(
                        task=task_type, random_state=42, **lasso_params
                    )

                # Bayesian LASSO
                if "B-LASSO" in active_methods:
                    blasso_params = method_defaults.get("BLASSO", {}).copy()
                    models["B-LASSO"] = BLASSOWrapper(task=task_type, **blasso_params)

                # ALFESE
                if "ALFESE" in active_methods:
                    alfese_params = method_defaults.get("ALFESE", {}).copy()
                    try:
                        import alfese

                        # Remove n_solutions if it's in params to avoid duplicate
                        alfese_params.pop("n_solutions", None)

                        models["ALFESE"] = AlfeseWrapper(
                            n_solutions=run_params.get("N_CANDIDATE_SOLUTIONS", 6),
                            task=task_type,
                            **alfese_params,
                        )
                    except ImportError:
                        print("Warning: ALFESE requested but not installed.")
                        pass

                # 6. Fit & Evaluate
                for model_name, model in models.items():
                    print(f"    Evaluating {model_name}...", end=" ", flush=True)
                    try:
                        t_start = pd.Timestamp.now()
                        solutions = model.fit(X, y)
                        duration = (pd.Timestamp.now() - t_start).total_seconds()

                        metrics = calculate_metrics(
                            solutions, true_support, run_params["N_FEATURES"]
                        )

                        log_entry = {
                            "Tier": tier_key,
                            "Method": model_name,
                            "Task": task_type,
                            **run_params,
                            **metrics,
                            "Runtime_Sec": duration,
                            "Status": "Success",
                        }
                        results.append(log_entry)
                        print(f"OK (ASI={metrics.get('Adjusted_SI', 0):.2f})")

                    except Exception as e:
                        print(f"FAIL ({str(e)})")
                        results.append(
                            {
                                "Tier": tier_key,
                                "Method": model_name,
                                "Status": "Failed",
                                "Error": str(e),
                                **run_params,
                            }
                        )

            except Exception as data_err:
                print(f"  CRITICAL: Data generation failed: {data_err}")

    # 7. Save Results
    os.makedirs(config["output_dir"], exist_ok=True)
    out_file = os.path.join(config["output_dir"], f"benchmark_results_{timestamp}.csv")
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\nSweep Completed. Results saved to {out_file}")


if __name__ == "__main__":
    run_sweep()
