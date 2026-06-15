#!/bin/bash
# BB-SSL fair-tuning, MAX parallelism: one array task per grid point
# (lambda0 x lambda1 x overlap x seed = 135 single-fit tasks). Each writes its
# own one-row CSV under results/bbssl_pt/; concatenate afterwards. Small per-task
# footprint (2 cpus, 4G) so the scheduler can run as many concurrently as fit.
#SBATCH --job-name=bbssl-grid
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-134
#SBATCH --output=logs/bbssl-grid-%A_%a.log
#SBATCH --error=logs/bbssl-grid-%A_%a.err

set -euo pipefail
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
cd "$WORKDIR"
mkdir -p logs

module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
module load libffi/3.4.4-GCCcore-13.2.0 2>/dev/null || true
module load R/4.2.1-foss-2022a
export R_LIBS_USER="$HOME/R/bbssl_lib"
unset PYTHONPATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

[ -f .venv/bin/python ] || { echo "[!] no .venv"; exit 1; }
cd algorithm_comparison
mkdir -p results/bbssl_pt

# Decode SLURM_ARRAY_TASK_ID -> (lambda0, lambda1, overlap, seed)
L0=(5 10 30 100 300); L1=(0.1 0.5 1.0); OV=(0 2 4); SD=(42 7 123)
i="$SLURM_ARRAY_TASK_ID"
sd="${SD[$((i % 3))]}";  i=$((i / 3))
ov="${OV[$((i % 3))]}";  i=$((i / 3))
l1="${L1[$((i % 3))]}";  i=$((i / 3))
l0="${L0[$((i % 5))]}"

echo "[*] pt $SLURM_ARRAY_TASK_ID: l0=$l0 l1=$l1 ov=$ov sd=$sd on $(hostname) $(date)"
../.venv/bin/python diagnostics/bbssl_tune.py \
    --lambda0 "$l0" --lambda1 "$l1" --overlaps "$ov" --seeds "$sd" --nsample 200 \
    --out "results/bbssl_pt/pt_${SLURM_ARRAY_TASK_ID}.csv"
echo "[*] done pt $SLURM_ARRAY_TASK_ID $(date)"
