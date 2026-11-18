#!/bin/bash
# Plan sec. 28 (experiment S): spike width on experiment G's connected cells. Target spike s.d. 0.01 (exact posterior
# at 0.01); spike fixed vs a geometric path 0.1 -> 0.01 over the first 4000 of 6000 steps; default and oracle
# starts; P in {block, 100, 1000}; same datasets and seeds as G. Submits only.
# Results: ~/amvi_compare/results/spike/S_*.csv
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
R=$BASE/results/spike
L=$BASE/logs/spike
mkdir -p $R $L
IDS=$L/jobids.txt
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1; unset PYTHONPATH"
PY="../.venv/bin/python"
COMMON="--reps 20 --rep0 2 --K 6 --arms F3J F3Ens --nuis diag --wmodes fixed --inits default oracle --spike 0.01 \
  --spike-paths none 0.1/4000"

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem=6G -t "$2" -J "sp_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

declare -A CELL
CELL[A_M_r0.9_n25]="--design M --rho 0.9 --p 10 --n 25 --slab 2 --w 0.2"
CELL[A_U_r0.9_n25]="--design U --rho 0.9 --p 10 --n 25 --slab 2 --w 0.2"
CELL[A_M_r0_n25]="--design M --rho 0 --p 10 --n 25 --slab 2 --w 0.2"
CELL[A_M_r0.9_n100]="--design M --rho 0.9 --p 10 --n 100 --slab 2 --w 0.2"

for c in "${!CELL[@]}"; do
  for P in 0 100 1000; do
    t=S_${c}_P$P
    [ $P -ge 1000 ] && T=10:00:00 || T=4:00:00
    sub $t $T "$HDR; $PY diagnostics/overlap_map.py ${CELL[$c]} $COMMON --P $P --out $R/$t.csv"
  done
done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
