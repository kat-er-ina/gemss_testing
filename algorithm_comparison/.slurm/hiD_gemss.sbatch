#!/bin/bash
# High-D upgrade, GEMSS + foils: per (overlap, seed) = 25 tasks. D=10 so overlap
# can climb to 8 (80%); overlaps {0,2,4,6,8}, seeds {42,7,123,1,2}.
#SBATCH --job-name=hiD-gemss
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-24
#SBATCH --output=logs/hiD-gemss-%A_%a.log
#SBATCH --error=logs/hiD-gemss-%A_%a.err

set -euo pipefail
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"; cd "$WORKDIR"; mkdir -p logs
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
module load libffi/3.4.4-GCCcore-13.2.0 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
[ -f .venv/bin/python ] || { echo "[!] no .venv"; exit 1; }
cd algorithm_comparison; mkdir -p results/hiD

OV=(0 2 4 6 8); SD=(42 7 123 1 2)
i="$SLURM_ARRAY_TASK_ID"
sd="${SD[$((i % 5))]}"; i=$((i / 5))
ov="${OV[$((i % 5))]}"

echo "[*] pt $SLURM_ARRAY_TASK_ID: ov=$ov sd=$sd D=10 on $(hostname) $(date)"
../.venv/bin/python diagnostics/three_way_sweep.py \
    --methods A_GEMSS EnumLasso StabilitySel \
    --D 10 --K 6 --overlaps "$ov" --seeds "$sd" \
    --out "results/hiD/gemss_pt_${SLURM_ARRAY_TASK_ID}.csv"
echo "[*] done $(date)"
