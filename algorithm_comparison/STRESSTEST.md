# Stress-test sweep — fair-budget comparison + classification fix

This extends Workflow 1 to build defensible ground for the paper rewrite. It
addresses the three structural weaknesses identified in the review of the
rejected manuscript:

1. **Unfair solution budget.** The original comparison gave GEMSS 6–12 candidate
   solutions but ALFESE only 3, while scoring the *union* of discovered features
   against 15 ground-truth features. Here **every multi-solution method gets the
   same budget** (`n_solutions: 6`, `desired_sparsity: 5`).
2. **Misrepresented classification model.** The shipped GEMSS core uses an L2
   (Gaussian) likelihood for *every* task, so "classification" is linear
   regression on {0,1} labels — not the logistic model the paper describes. We
   add `GEMSS_logistic` (a Bernoulli-likelihood subclass) and run it **alongside**
   the L2 `GEMSS` so the effect of the fix is *measured*, not hidden.
3. **Weak baselines.** The ALFESE filters (MI, mRMR, FCBF, greedy, importance)
   are weak on linear-Gaussian data. We add the strong sparse-linear baselines a
   practitioner would actually use.

## Methods

| Method | What it is | Role in the story |
|---|---|---|
| `GEMSS` | GEMSS as shipped (L2 likelihood) | reference |
| `GEMSS_logistic` | GEMSS + Bernoulli/cross-entropy likelihood | does the "correct" classification model help? |
| `GEMSS_single` | GEMSS with **one** Gaussian (no mixture) | ablation: is the mixture necessary? |
| `Masking_lasso` / `_elasticnet` / `_logistic` | iterative-masking ("peeling") sequential selection | the direct competitor to simultaneous discovery |
| `StabilitySelection` | Meinshausen–Bühlmann stable core | demonstrates Rashomon collapse to one point |
| `ALFESE_{mi,greedy,importance}` | original wrappers | continuity with the previous comparison |

All sklearn baselines **standardize features** and **mean-impute NaNs** (they
cannot ingest missing values — recorded honestly, and a fair disadvantage to
flag against GEMSS's native NaN handling).

## Run

Local (smoke / small sweep):
```bash
uv run python algorithm_comparison/scripts/run_benchmark.py -c stresstest_config.yaml
uv run python algorithm_comparison/scripts/run_benchmark.py -c stresstest_config.yaml -e 1 2   # subset
```
RCI cluster (contract: `~/agents/compute/rci.md`): see
[`.slurm/stresstest.sbatch`](.slurm/stresstest.sbatch). CPU-only; provision the
venv with `uv venv --python 3.13 && uv sync`.

## Early signal (exp 1, N=50 P=100, classification, 6-solution budget)

| Method | Recall | Precision | F1 | union\|features |
|---|---|---|---|---|
| GEMSS (L2) | 1.00 | 1.00 | **1.00** | 14 |
| Masking (lasso/enet/logit) | 1.00 | 0.47 | 0.64 | 30 |
| GEMSS_logistic | 0.50 | 0.78 | 0.61 | 9 |
| GEMSS_single (ablation) | 0.36 | 1.00 | 0.53 | 5 |
| StabilitySelection | 0.21 | 1.00 | 0.35 | 3 |
| ALFESE_mi | 0.86 | 0.40 | 0.55 | 30 |

Reading: at equal budget the **mixture self-prunes** to exactly the true
features (GEMSS union = 14 vs masking's forced 30) — so GEMSS's edge is
**parsimony, not recall** (strong masking baselines also recover everything).
The single-Gaussian ablation recovers only one solution, motivating the mixture.
Stability selection collapses to a core. The "correct" logistic likelihood is
**not** obviously better at equal tuning — a finding to characterize over the
full sweep (and likely needs logistic-specific hyperparameters).

## Known asymmetries / next steps (deliberately deferred)

- **GEMSS sees raw (unstandardized) features; baselines see standardized.** On
  the current generator the relevant features have ~10–100× the variance of the
  noise features, which favours GEMSS. The right fix is in the **generator**
  (equal-variance features), the planned next phase — not silently rescaling
  GEMSS's input.
- **Supports are disjoint** in the current generator. GEMSS's headline claim —
  recovering *overlapping* alternative solutions that masking cannot — is **not
  yet tested**. The honest/hard generator with controlled overlap is the key
  next experiment.
- `GEMSS_logistic` and the baselines use default-ish hyperparameters; a fair
  characterization of the logistic fix needs a small per-method tuning pass.
