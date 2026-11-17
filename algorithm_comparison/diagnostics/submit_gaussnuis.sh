#!/bin/bash
# Plan sec. 24: Gaussian-nuisance control for experiment G. Same datasets, seeds and starts as G's connected cells
# (diag nuisance, w fixed); only the nuisance prior changes (slab weight 0 -> spike alone, a Gaussian).
# Submits only (no compute on the login node). Results: ~/amvi_compare/results/gaussnuis/N_*.csv
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
R=$BASE/results/gaussnuis
L=$BASE/logs/gaussnuis
mkdir -p $R $L
IDS=$L/jobids.txt
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1; unset PYTHONPATH"
PY="../.venv/bin/python"
COMMON="--reps 20 --rep0 2 --K 6 --arms F3J F3Ens --nuis diag --nuis-prior gauss --wmodes fixed --inits default oracle"

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem=6G -t "$2" -J "gn_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

declare -A CELL
CELL[A_M_r0.9_n25]="--design M --rho 0.9 --p 10 --n 25 --slab 2 --w 0.2"
CELL[A_U_r0.9_n25]="--design U --rho 0.9 --p 10 --n 25 --slab 2 --w 0.2"
CELL[A_M_r0_n25]="--design M --rho 0 --p 10 --n 25 --slab 2 --w 0.2"
CELL[A_M_r0.9_n100]="--design M --rho 0.9 --p 10 --n 100 --slab 2 --w 0.2"

for c in "${!CELL[@]}"; do
  for P in 30 100 300 1000; do                      # block size is shared with experiment G (no nuisance)
    t=N_${c}_P$P
    [ $P -ge 300 ] && T=10:00:00 || T=4:00:00
    sub $t $T "$HDR; $PY diagnostics/overlap_map.py ${CELL[$c]} $COMMON --P $P --out $R/$t.csv"
  done
done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
