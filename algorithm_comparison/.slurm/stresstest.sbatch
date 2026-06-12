#!/bin/bash
# Stress-test sweep on the RCI cluster (CTU). Contract: ~/agents/compute/rci.md.
# CPU-only: GEMSS runs on torch-CPU (device='cpu'); no GPU needed.
#
# One-time setup on the login node (login3), from your working folder:
#   git clone <gemss_testing remote> && cd gemss_testing
#   git checkout stresstest/fair-comparison-logistic
#   module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
#   module load libffi/3.4.4-GCCcore-13.2.0 2>/dev/null || true   # scipy/sklearn C-ext
#   uv venv --python 3.13        # pyproject requires >=3.13; uv fetches it (outbound HTTPS ok)
#   uv sync
#   sbatch algorithm_comparison/.slurm/stresstest.sbatch
#
# Override the working dir at submit time:  sbatch --export=WORKDIR=/mnt/data/<alloc>/gemss_testing ...
#
#SBATCH --job-name=gemss-stress
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/stresstest-%j.log
#SBATCH --error=logs/stresstest-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
cd "$WORKDIR"
mkdir -p logs

module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
module load libffi/3.4.4-GCCcore-13.2.0 2>/dev/null || true

[ -f .venv/bin/python ] || { echo "[!] no .venv — run 'uv venv --python 3.13 && uv sync' first"; exit 1; }

# Keep torch from oversubscribing the allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Config + optional seeds overridable at submit time, e.g.
#   sbatch --export=ALL,CFG=hardtest_config.yaml,SEEDS="42 7 123" stresstest.sbatch
CFG="${CFG:-stresstest_config.yaml}"
SEED_ARG=""
[ -n "${SEEDS:-}" ] && SEED_ARG="-s ${SEEDS}"

cd algorithm_comparison
echo "[*] Sweep starting on $(hostname) at $(date): config=$CFG seeds='${SEEDS:-<config default>}'"
../.venv/bin/python scripts/run_benchmark.py -c "$CFG" $SEED_ARG
echo "[*] Done at $(date). Results in algorithm_comparison/results/"
