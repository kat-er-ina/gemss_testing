# Reproducing the GEMSS paper

This directory reproduces the results in the GEMSS paper (Discovery Science).
Every table/number in the paper maps to one recipe below: *bundled raw results*
→ *aggregate* → *the number in the paper*.

The sweeps that produced the raw results were run as array jobs on an internal
cluster; **those job scripts are not distributed**. What ships here is the
comparison harness that the jobs called (`diagnostics/tuned_compare.py` for
synthetic sweeps, `scripts/run_*.py` for real data) **plus the per-trial CSVs
they produced**, under `results/paper/<run>/`. So every paper number reproduces
**offline** by running the aggregation script over the bundled raw data — no
cluster needed. To regenerate the raw CSVs from scratch, run the named harness
entry point at the parameters stated in each recipe.

> Status: ✅ all aggregation scripts reproduce the paper numbers — run any
> `scripts/make_*.py` / `explorer_dissim.py`; each prints a MATCHES-PAPER check.

---

## 0. Setup

```bash
uv venv && uv sync                      # .venv from pyproject.toml + uv.lock
# BB-SSL baseline needs R:
#   module load R/4.2.1-foss-2022a ; export R_LIBS_USER=$HOME/R/bbssl_lib ; unset PYTHONPATH
```

**Data.** Synthetic data is generated on the fly (`src/hard_data_factory.py`). Real
datasets (diabetes MTBLS1, *Arabidopsis* MTBLS2, PCOS MTBLS12968) live under
`real_world_applications/data/`.

**Run the aggregators** with `.venv/bin/python scripts/<name>.py` from this directory
(`results/paper/` is the default input).

---

## 1. Paper result → recipe

### Table 1 — HP grids (`tab:tuning`)
Documentation only; the grids are defined in `diagnostics/tuned_compare.py::grid()`.

### Table 2 — overlap sweep, union-F1 + recovered dissimilarity (`tab:rq2_fair`)
- **Raw (bundled):** `results/paper/{jointrq2,rq2stat}/` — GEMSS(joint) and the baselines
  (ALFESE, masking, BB-SSL, RLE, EnumLASSO) from the harness (`tuned_compare.py`),
  $D{=}10$, ov$\{0,2,4,8\}$, 25 seeds.
- ✅ `scripts/make_table2.py` — per-method union-F1 + recovered dissim per overlap + analytic
  GT; **verified: reproduces every cell (Δ=0.00)**. (mean±CI / win-rate also via `diagnostics/stats_agg.py`.)

### Table — RQ3 real data, diabetes + PCOS (`tab:rq3`)
- **Run directly:** `scripts/run_rq3_clean.py -d diabetes -K 12 -m 8 -D 10` and
  `scripts/run_pcos_preterm.py -K 12 -m 8 -D 10`.
- These emit per-method rows (#sol, F1 mean/min/best, dissim) directly — table-ready.
- Leak fix (drop `stratification_groups`) is built into `run_rq3_clean.load`.

### Table — RQ4 ensemble budget curve (`tab:rq4`)
- **Raw (bundled):** `results/paper/{rq4stat,jointrq2}/` — ensemble recovery vs. restart count, 15 seeds.
- ✅ `scripts/make_rq4.py` (best-α-per-budget union-F1 vs GEMSS); **verified: all cells match**.

### Table — RQ5 dimensionality, mean rank (`tab:RQ5_lown`)
- **Raw (bundled):** `results/paper/{mechsweep,jointrq6,rq6grid}/` —
  $n\{30,50,100\}\times p\{1000,2000,5000\}\times$ ov$\{2,6\}$, 15 seeds.
- ✅ `scripts/make_rq5_meanrank.py`; **verified: GEMSS 1.17 (15/18), ALFESE 2.06, ensemble 2.78**.

### Table — RQ6 noise & missingness (`tab:RQ6`)
- **Raw (bundled):** `results/paper/{jointrq7,rq7grid}/` — $p{=}200$, ov$4$, $D{=}10$.
- ✅ `scripts/make_rq6.py` (union-F1 + dissim by noise/missing level); **verified: all cells match**.

### Jaccard penalty paragraph (§GEMSS)
- **Raw (bundled):** `results/paper/jjsweep/` — GEMSSJjoint, $\lambda\in\{0,\dots,10^6\}$ × overlaps.
- ✅ `scripts/make_jaccard.py` (union-F1 + recovered-vs-true dissim per λ);
  **verified: ov8 gap 0.27→0.62, uF1 0.88→0.37**.

### §5 real-world summary + dissimilarity (`tab:rw_summary`)
- **Source:** the **GEMSS Explorer** application (the no-code app demonstration).
  Exports vendored under `explorer_reports/{diabetes,arabidopsis}_report.html`.
- ✅ `scripts/explorer_dissim.py` — mean pairwise $1-$Jaccard over the 8 candidates;
  **verified: diabetes 0.91, Arabidopsis 0.97**.
- ⚠️ food-science (0.92 in the paper) is **not reproduced here**: its data is not public
  and is deliberately excluded from this companion.

### RQ5 honest-negative (count test) — **dropped from the paper**
Code retained for completeness: `diagnostics/rq5_count.py`. Not cited in the paper.

---

## 2. Core code (the closed dependency set behind the above)
- generators + metrics: `src/hard_data_factory.py`, `src/evaluation.py`
- GEMSS: `src/wrappers/mechanism_gemss.py` (joint; depends on `logistic_gemss.py`), `gemss_wrapper.py`
- baselines: `src/wrappers/{sklearn_wrappers,enumlasso_wrapper,alfese_wrapper}.py`, `src/external/enumerate_linear_model.py`
- harness: `diagnostics/{tuned_compare,ensemble_parity,stats_agg,bb_ssl_probe}.py`, `bb_ssl_bridge.R`
- entry scripts: `scripts/{run_rq3_clean,run_pcos_preterm}.py`, `diagnostics/rq5_count.py`
- aggregation: `scripts/{make_table2,make_rq4,make_rq5_meanrank,make_rq6,make_jaccard,explorer_dissim}.py`

---

## 3. Not in this companion
Exploratory and superseded experiments (alternative mechanisms, probes, the
validation suite, post-submission anti-collapse work) are **not** part of the
paper and live on the development branch, not here. The internal cluster job
scripts are likewise not distributed. `results/` is gitignored except the
curated `results/paper/` subset that backs the tables.
