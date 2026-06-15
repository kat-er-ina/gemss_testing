#!/bin/bash
# RQ7 fair ensemble: tuned randomised-Lasso ensemble (restart curve, alpha=0.01)
# under noise / missingness, to pair honestly with GEMSS. idx 0-19 noise,
# 20-39 missing. D=10, overlap=4, p=200, seeds {42,7,123,1,2}.
#SBATCH --job-name=rq7-fair
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-39
#SBATCH --output=logs/rq7fair-%A_%a.log
#SBATCH --error=logs/rq7fair-%A_%a.err
set -euo pipefail
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"; cd "$WORKDIR"; mkdir -p logs
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
module load libffi/3.4.4-GCCcore-13.2.0 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
[ -f .venv/bin/python ] || { echo "[!] no .venv"; exit 1; }
cd algorithm_comparison; mkdir -p results/rq7fair
NOISE=(0.05 0.2 0.5 1.0); NAN=(0.0 0.1 0.3 0.5); SD=(42 7 123 1 2)
i="$SLURM_ARRAY_TASK_ID"
if [ "$i" -lt 20 ]; then
  j=$i; sd="${SD[$((j % 5))]}"; j=$((j / 5)); lvl="${NOISE[$((j % 4))]}"
  ARG="--noise $lvl --nan 0.0"
else
  j=$((i - 20)); sd="${SD[$((j % 5))]}"; j=$((j / 5)); lvl="${NAN[$((j % 4))]}"
  ARG="--noise 0.05 --nan $lvl"
fi
echo "[*] pt $SLURM_ARRAY_TASK_ID: $ARG sd=$sd $(date)"
../.venv/bin/python diagnostics/ensemble_parity.py \
    --alphas 0.01 --D 10 --K 6 --n 100 --p 200 --overlaps 4 --seeds "$sd" $ARG \
    --restarts 50 300 3000 \
    --out "results/rq7fair/ens_pt_${SLURM_ARRAY_TASK_ID}.csv"
echo "[*] done $(date)"
