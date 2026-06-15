#!/bin/bash
# BB-SSL fair-tuning sweep (instance B), parallelised as an array over lambda0 so
# no single task can hit the wall limit; each writes rows incrementally.
#SBATCH --job-name=bbssl-tune
#SBATCH --partition=cpu
#SBATCH --time=01:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3
#SBATCH --output=logs/bbssl-tune-%A_%a.log
#SBATCH --error=logs/bbssl-tune-%A_%a.err

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

L0S=(5 10 30 100)
L0="${L0S[$SLURM_ARRAY_TASK_ID]}"
echo "[*] BB-SSL tuning lambda0=$L0 on $(hostname) at $(date)"
../.venv/bin/python diagnostics/bbssl_tune.py \
    --lambda0 "$L0" --lambda1 0.1 0.5 1.0 \
    --overlaps 2 4 --seeds 42 7 123 --nsample 200 \
    --out "results/bbssl_tune_l0_${L0}.csv"
echo "[*] done lambda0=$L0 at $(date)."
