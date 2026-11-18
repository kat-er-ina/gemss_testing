#!/bin/bash
# Plan sec. 27: localised coupling on experiment G's connected cells. Same datasets, seeds and starts as G (diag
# nuisance, w fixed); the joint fit's components are free only on a kept set (the block, or the top-20 / top-50
# features by marginal score), all other coordinates shared. F3J rerun as a reproducibility check against G; the
# ensemble is taken from G. Submits only (no compute on the login node). Results: ~/amvi_compare/results/loccoup/L_*.csv
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
R=$BASE/results/loccoup
L=$BASE/logs/loccoup
mkdir -p $R $L
IDS=$L/jobids.txt
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1; unset PYTHONPATH"
PY="../.venv/bin/python"
COMMON="--reps 20 --rep0 2 --K 6 --arms F3J F3Jloc_or F3Jloc20 F3Jloc50 --nuis diag --wmodes fixed --inits default oracle"

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem=6G -t "$2" -J "lc_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

declare -A CELL
CELL[A_M_r0.9_n25]="--design M --rho 0.9 --p 10 --n 25 --slab 2 --w 0.2"
CELL[A_U_r0.9_n25]="--design U --rho 0.9 --p 10 --n 25 --slab 2 --w 0.2"
CELL[A_M_r0_n25]="--design M --rho 0 --p 10 --n 25 --slab 2 --w 0.2"
CELL[A_M_r0.9_n100]="--design M --rho 0.9 --p 10 --n 100 --slab 2 --w 0.2"

for c in "${!CELL[@]}"; do
  for P in 30 100 300 1000; do                      # block size: every localised arm equals F3J (shared with G)
    t=L_${c}_P$P
    sub $t 4:00:00 "$HDR; $PY diagnostics/overlap_map.py ${CELL[$c]} $COMMON --P $P --out $R/$t.csv"
  done
done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
