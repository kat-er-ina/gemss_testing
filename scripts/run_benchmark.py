"""Main entry point for running benchmarks."""

import yaml
import pandas as pd
from src.data_factory import get_benchmark_data
from src.wrappers.gemss_wrapper import GEMSSWrapper
from src.wrappers.alfese_wrapper import AlfeseWrapper
from src.wrappers.lasso_wrapper import LassoWrapper
from src.evaluation import calculate_metrics


def main():
    # Load Config
    with open("configs/experiment_config.yaml") as f:
        config = yaml.safe_load(f)

    results_log = []

    # Iterate over settings (e.g., Tiers)
    for tier in config["tiers"]:
        print(f"Running Tier: {tier['name']}")

        # Generate Data
        X, y, true_support = get_benchmark_data(
            n_samples=tier["n_samples"],
            n_features=tier["n_features"],
            n_generating_solutions=tier["n_gen_solutions"],
            sparsity=tier["sparsity"],
            noise_std=tier["noise_std"],
            seed=42,
        )

        # Instantiate Models
        models = {
            "GEMSS": GEMSSWrapper(**config["methods"]["gemss"]),
            "ALFESE": AlfeseWrapper(**config["methods"]["alfese"]),
            "LASSO": LassoWrapper(),
        }

        # Run Benchmark
        for name, model in models.items():
            print(f"  Evaluating {name}...")
            try:
                # Fit
                solutions = model.fit(X, y)

                # Evaluate
                metrics = calculate_metrics(solutions, true_support, tier["n_features"])

                # Log
                entry = {**tier, "Method": name, **metrics}
                results_log.append(entry)

            except Exception as e:
                print(f"  Failed {name}: {e}")

    # Save Results
    pd.DataFrame(results_log).to_csv("results/benchmark_summary.csv", index=False)
    print("Benchmark Complete.")


if __name__ == "__main__":
    main()
