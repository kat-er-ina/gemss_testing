#!/bin/bash
# High-D upgrade, ensemble parity: per (alpha, overlap, seed) = 75 tasks.
# Restart curve extended to 30000 (10x) to show the ensemble cannot close the
# high-overlap gap with budget. D=10, overlaps {0,2,4,6,8}, seeds {42,7,123,1,2}.
#SBATCH --job-name=hiD-ens
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-74
#SBATCH --output=logs/hiD-ens-%A_%a.log
#SBATCH --error=logs/hiD-ens-%A_%a.err

set -euo pipefail
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"; cd "$WORKDIR"; mkdir -p logs
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
module load libffi/3.4.4-GCCcore-13.2.0 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
[ -f .venv/bin/python ] || { echo "[!] no .venv"; exit 1; }
cd algorithm_comparison; mkdir -p results/hiD

A=(0.005 0.01 0.02); OV=(0 2 4 6 8); SD=(42 7 123 1 2)
i="$SLURM_ARRAY_TASK_ID"
sd="${SD[$((i % 5))]}"; i=$((i / 5))
ov="${OV[$((i % 5))]}"; i=$((i / 5))
a="${A[$((i % 3))]}"

echo "[*] pt $SLURM_ARRAY_TASK_ID: alpha=$a ov=$ov sd=$sd D=10 on $(hostname) $(date)"
../.venv/bin/python diagnostics/ensemble_parity.py \
    --alphas "$a" --overlaps "$ov" --seeds "$sd" --D 10 --K 6 \
    --restarts 50 100 300 1000 3000 10000 30000 \
    --out "results/hiD/ens_pt_${SLURM_ARRAY_TASK_ID}.csv"
echo "[*] done $(date)"
