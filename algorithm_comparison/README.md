# Algorithm Comparison - Quantitative Benchmarking

**Purpose:** Objective performance comparison of GEMSS vs ALFESE using controlled synthetic data with automated metrics.

## Overview

This workflow evaluates feature selection algorithms on synthetic datasets with known ground truth, enabling quantitative assessment through metrics like F1 Score, Precision, Recall, and Success Index.

## Configuration

16 experiments defined in [`configs/benchmark_config.yaml`](configs/benchmark_config.yaml) spanning 4 tiers:

- **Tier 1** (Exp 1-5): Basic validation, clean data
- **Tier 2** (Exp 6-11): High-dimensional (P ≥ 1000)
- **Tier 4** (Exp 12-14): Adversity (noise, missing data)
- **Tier 7** (Exp 15-16): Class imbalance

## Methods Compared

- **GEMSS**: Bayesian multi-solution selector
- **ALFESE** (variants): Model-X knockoffs with different selectors (mrmr, mi, greedy, importance, fcbf) and tau values

## Evaluation Metrics

- **Primary**: F1 Score, Adjusted Success Index, Runtime
- **Secondary**: Recall, Precision, Jaccard Index

## Usage

```bash
# Run all experiments with default methods
uv run python algorithm_comparison/scripts/run_benchmark.py

# Run specific experiments
uv run python algorithm_comparison/scripts/run_benchmark.py -e 1 2 3

# Run with specific methods
uv run python algorithm_comparison/scripts/run_benchmark.py -m GEMSS ALFESE_mi_tau1.0

# Run specific experiments with specific methods
uv run python algorithm_comparison/scripts/run_benchmark.py -e 1 2 3 -m GEMSS

# Analyze results
uv run jupyter notebook algorithm_comparison/notebooks/analyze_results.ipynb
```

## Structure

```
algorithm_comparison/
├── configs/          # Experiment configurations
├── src/              # Benchmarking infrastructure
│   ├── data_factory.py      # Synthetic data generation
│   ├── evaluation.py        # Metrics calculation
│   └── wrappers/            # Algorithm wrappers (GEMSS, ALFESE)
├── scripts/          # Benchmark execution scripts
│   └── run_benchmark.py     # Main benchmark runner
├── notebooks/        # Analysis notebooks
│   └── analyze_results.ipynb  # Results analysis notebook
└── results/          # Output CSV files (git-ignored)
```

## Output

Results are saved to [`results/`](results/) as timestamped CSV files with columns:
- Experiment metadata (ID, description, task type)
- Dataset parameters (n_samples, n_features, sparsity, etc.)
- Evaluation metrics
- Runtime and status

## Key Features

- Synthetic data with known ground truth
- Controlled experimental conditions (sparsity=5, seed=42)
- Incremental result saving (resume-friendly)
- Automated result aggregation and visualization
- Flexible experiment and method selection via CLI
