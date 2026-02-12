# GEMSS Explorer App

An interactive [marimo](https://marimo.io/) application for exploring multiple sparse solutions in your data using the GEMSS feature selection algorithm.


## Running the app

### 1. Install uv
If you do not have `uv` installed, run the following:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Launch the app

From the `gemss_benchmarking` root folder:

```bash
uv run marimo run app/gemss_explorer_noncommercial.py
```

The app will open in your browser at `http://localhost:2718`.

## Using the app

The app provides an interactive interface with:
- **No coding required** — all parameters set via widgets
- **Interactive plots** — hover, zoom, and export with Plotly controls
- **Export tables** — download any table as CSV
- **Session state** — progress persists while browser tab is open

## Data requirements

- **Format:** CSV with numeric features (missing values OK)
- **Structure:** Features in columns, samples in rows
- **Required columns:** Index column and target/label column
- **Supported tasks:** Binary classification or regression

## Workflow

1. Configure output directory and file names
2. Upload CSV and select index/target columns
3. Set algorithm parameters (components, sparsity, optimization)
4. Run Bayesian feature selection
5. Assess convergence via ELBO and trajectory plots
6. Recover sparse feature sets
7. Evaluate solutions with logistic/linear regression
8. Evaluate the solutions with [TabPFN](https://huggingface.co/Prior-Labs/tabpfn_2_5) and explain the model with Shapley values

⚠️ **Note:** TabPFN usage requires agreement with its [license terms](https://huggingface.co/Prior-Labs/tabpfn_2_5#licensing).

## Output files

Results are saved to `experiment_<ID>/` in your chosen directory:
- `search_history_results.json` — optimization history
- `search_setup.json` — algorithm configuration
- `all_candidate_solutions.json` — feature sets (JSON)
- `all_candidate_solutions.txt` — feature sets (human-readable)

## Help

The app includes expandable help panels (📖) throughout for guidance on parameters and results interpretation.
