#!/bin/bash
# Ensemble parity sweep: one array task per (alpha, overlap, seed) = 54 tasks.
# Each runs the nested restart-budget prefix-curve (R_max=3000). Pure sklearn,
# no R needed. Writes one CSV per task under results/ens_parity_pt/.
#SBATCH --job-name=ens-parity
#SBATCH --partition=cpu
#SBATCH --time=00:40:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-53
#SBATCH --output=logs/ens-parity-%A_%a.log
#SBATCH --error=logs/ens-parity-%A_%a.err

set -euo pipefail
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
cd "$WORKDIR"
mkdir -p logs

module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
module load libffi/3.4.4-GCCcore-13.2.0 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

[ -f .venv/bin/python ] || { echo "[!] no .venv"; exit 1; }
cd algorithm_comparison
mkdir -p results/ens_parity_pt

A=(0.005 0.01 0.02 0.05 0.1 0.2); OV=(0 2 4); SD=(42 7 123)
i="$SLURM_ARRAY_TASK_ID"
sd="${SD[$((i % 3))]}"; i=$((i / 3))
ov="${OV[$((i % 3))]}"; i=$((i / 3))
a="${A[$((i % 6))]}"

echo "[*] pt $SLURM_ARRAY_TASK_ID: alpha=$a ov=$ov sd=$sd on $(hostname) $(date)"
../.venv/bin/python diagnostics/ensemble_parity.py \
    --alphas "$a" --overlaps "$ov" --seeds "$sd" \
    --restarts 50 100 300 1000 3000 \
    --out "results/ens_parity_pt/pt_${SLURM_ARRAY_TASK_ID}.csv"
echo "[*] done pt $SLURM_ARRAY_TASK_ID $(date)"
