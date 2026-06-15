#!/bin/bash
# BB-SSL fair-tuning sweep (instance B) on the synthetic overlap generator.
#SBATCH --job-name=bbssl-tune
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/bbssl-tune-%j.log
#SBATCH --error=logs/bbssl-tune-%j.err

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
echo "[*] BB-SSL tuning on $(hostname) at $(date)"
../.venv/bin/python diagnostics/bbssl_tune.py \
    --lambda0 5 10 30 100 300 --lambda1 0.1 0.5 1.0 \
    --overlaps 2 4 --seeds 42 7 123 --out results/bbssl_tune.csv
echo "[*] done at $(date)."
