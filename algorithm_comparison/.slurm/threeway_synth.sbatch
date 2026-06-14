#!/bin/bash
# Three-way synthetic sweep (RQ1 + RQ2): GEMSS / BB-SSL / RandLasso + foils on
# regression overlap data. Needs R + BBSSL (R_LIBS_USER) for instance B.
#SBATCH --job-name=gemss-3way-synth
#SBATCH --partition=cpu
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/threeway-synth-%j.log
#SBATCH --error=logs/threeway-synth-%j.err

set -euo pipefail
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
cd "$WORKDIR"
mkdir -p logs

module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
module load libffi/3.4.4-GCCcore-13.2.0 2>/dev/null || true
module load R/4.2.1-foss-2022a
export R_LIBS_USER="$HOME/R/bbssl_lib"
unset PYTHONPATH               # keep R's foss-numpy out of the .venv (Rscript uses R_LIBS)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

[ -f .venv/bin/python ] || { echo "[!] no .venv"; exit 1; }
cd algorithm_comparison
echo "[*] three-way synth sweep on $(hostname) at $(date)"
../.venv/bin/python diagnostics/three_way_sweep.py \
    --seeds 42 7 123 --overlaps 0 2 4 --n 100 --p 200 --D 5 --K 6 \
    --out results/three_way_sweep.csv
echo "[*] done at $(date). -> algorithm_comparison/results/three_way_sweep.csv"
