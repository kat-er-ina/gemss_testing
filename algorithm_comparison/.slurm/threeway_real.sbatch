#!/bin/bash
# Real-data nested CV (RQ3): all instances incl. BB-SSL on the 3 real datasets.
# Needs R + BBSSL and the gemss preprocessed datasets at ~/gemss/data/.
#SBATCH --job-name=gemss-3way-real
#SBATCH --partition=cpu
#SBATCH --time=05:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/threeway-real-%j.log
#SBATCH --error=logs/threeway-real-%j.err

set -euo pipefail
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
cd "$WORKDIR"
mkdir -p logs

module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
module load libffi/3.4.4-GCCcore-13.2.0 2>/dev/null || true
module load R/4.2.1-foss-2022a
export R_LIBS_USER="$HOME/R/bbssl_lib"
unset PYTHONPATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

[ -f .venv/bin/python ] || { echo "[!] no .venv"; exit 1; }
cd algorithm_comparison
echo "[*] real-data nested CV on $(hostname) at $(date)"
../.venv/bin/python scripts/run_real.py --nested
echo "[*] done at $(date). -> algorithm_comparison/results/real_data_nested.csv"
