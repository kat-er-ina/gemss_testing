#!/bin/bash
# Plan sec. 23 (experiment G-amvi): Li et al.'s mixture on experiment G's connected cells (diag nuisance, w fixed).
# Submits only (no compute on the login node). Results: ~/amvi_compare/results/amvi_dimdist/GA_*.csv
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
A=$BASE/adaptive-mixture-spike-slab            # their package, commit b8341d8 (run only, never vendored)
R=$BASE/results/amvi_dimdist
L=$BASE/logs/amvi_dimdist
mkdir -p $R $L
IDS=$L/jobids.txt
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1; export PYTHONPATH=$A/src"
PY="../.venv/bin/python"
COMMON="--reps 20 --rep0 2 --slab 2 --w 0.2 --max-seconds 600 --max-components 10"   # reps 0-1: smoke datasets

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem=8G -t "$2" -J "ga_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

declare -A CELL
CELL[A_M_r0.9_n25]="--design M --rho 0.9 --n 25"
CELL[A_U_r0.9_n25]="--design U --rho 0.9 --n 25"
CELL[A_M_r0_n25]="--design M --rho 0 --n 25"
CELL[A_M_r0.9_n100]="--design M --rho 0.9 --n 100"

for c in "${!CELL[@]}"; do
  for P in 0 30 100 300 1000; do
    t=GA_${c}_P$P
    [ $P -ge 30 ] && T=10:00:00 || T=4:00:00
    sub $t $T "$HDR; $PY diagnostics/amvi_dimdist.py ${CELL[$c]} --P $P $COMMON --out $R/$t.csv"
  done
done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
