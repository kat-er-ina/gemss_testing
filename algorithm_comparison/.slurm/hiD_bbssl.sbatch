#!/bin/bash
# High-D upgrade, BB-SSL: per (lambda0, lambda1, overlap, seed) = 150 tasks, so
# the three-way tables stay consistent at D=10. Per-point + incremental writes.
#SBATCH --job-name=hiD-bbssl
#SBATCH --partition=cpu
#SBATCH --time=00:40:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-149
#SBATCH --output=logs/hiD-bbssl-%A_%a.log
#SBATCH --error=logs/hiD-bbssl-%A_%a.err

set -euo pipefail
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"; cd "$WORKDIR"; mkdir -p logs
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
module load libffi/3.4.4-GCCcore-13.2.0 2>/dev/null || true
module load R/4.2.1-foss-2022a
export R_LIBS_USER="$HOME/R/bbssl_lib"
unset PYTHONPATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
[ -f .venv/bin/python ] || { echo "[!] no .venv"; exit 1; }
cd algorithm_comparison; mkdir -p results/hiD

L0=(5 10 30); L1=(0.1 0.5); OV=(0 2 4 6 8); SD=(42 7 123 1 2)
i="$SLURM_ARRAY_TASK_ID"
sd="${SD[$((i % 5))]}"; i=$((i / 5))
ov="${OV[$((i % 5))]}"; i=$((i / 5))
l1="${L1[$((i % 2))]}"; i=$((i / 2))
l0="${L0[$((i % 3))]}"

echo "[*] pt $SLURM_ARRAY_TASK_ID: l0=$l0 l1=$l1 ov=$ov sd=$sd D=10 on $(hostname) $(date)"
../.venv/bin/python diagnostics/bbssl_tune.py \
    --lambda0 "$l0" --lambda1 "$l1" --overlaps "$ov" --seeds "$sd" \
    --D 10 --K 6 --nsample 200 \
    --out "results/hiD/bbssl_pt_${SLURM_ARRAY_TASK_ID}.csv"
echo "[*] done $(date)"
