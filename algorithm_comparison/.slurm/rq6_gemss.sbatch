#!/bin/bash
# RQ6 scaling, GEMSS + foils: per (p, overlap, seed) = 30 tasks. n=100 fixed,
# D=10, overlaps {2 (low),6 (high)}, p in {500,1000,2000}, seeds {42,7,123,1,2}.
#SBATCH --job-name=rq6-gemss
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-29
#SBATCH --output=logs/rq6-gemss-%A_%a.log
#SBATCH --error=logs/rq6-gemss-%A_%a.err
set -euo pipefail
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"; cd "$WORKDIR"; mkdir -p logs
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
module load libffi/3.4.4-GCCcore-13.2.0 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
[ -f .venv/bin/python ] || { echo "[!] no .venv"; exit 1; }
cd algorithm_comparison; mkdir -p results/rq6
P=(500 1000 2000); OV=(2 6); SD=(42 7 123 1 2)
i="$SLURM_ARRAY_TASK_ID"
sd="${SD[$((i % 5))]}"; i=$((i / 5))
ov="${OV[$((i % 2))]}"; i=$((i / 2))
p="${P[$((i % 3))]}"
echo "[*] pt $SLURM_ARRAY_TASK_ID: p=$p ov=$ov sd=$sd D=10 $(date)"
../.venv/bin/python diagnostics/three_way_sweep.py \
    --methods A_GEMSS EnumLasso StabilitySel \
    --D 10 --K 6 --n 100 --p "$p" --overlaps "$ov" --seeds "$sd" \
    --out "results/rq6/gemss_pt_${SLURM_ARRAY_TASK_ID}.csv"
echo "[*] done $(date)"
