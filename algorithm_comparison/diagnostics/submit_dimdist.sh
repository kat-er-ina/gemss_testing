#!/bin/bash
# Plan sec. 21 (experiment G): dimension or distance? Submits only (no compute on the login node).
# Results: ~/amvi_compare/results/dimdist/G_*.csv; logs ~/amvi_compare/logs/dimdist.
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
R=$BASE/results/dimdist
L=$BASE/logs/dimdist
mkdir -p $R $L
IDS=$L/jobids.txt
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1; unset PYTHONPATH"
PY="../.venv/bin/python"
COMMON="--reps 20 --rep0 2 --K 6 --arms F3J F3Ens"   # reps 0-1 were the smoke datasets

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem=6G -t "$2" -J "dd_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

# block geometries with exact posteriors: connected (A designs; mini-RQ1 with slab 2) to separated (mini-RQ1, slab 100)
declare -A CELL
CELL[A_M_r0.9_n25]="--design M --rho 0.9 --p 10 --n 25 --slab 2 --w 0.2"
CELL[A_U_r0.9_n25]="--design U --rho 0.9 --p 10 --n 25 --slab 2 --w 0.2"
CELL[A_M_r0_n25]="--design M --rho 0 --p 10 --n 25 --slab 2 --w 0.2"
CELL[A_M_r0.9_n100]="--design M --rho 0.9 --p 10 --n 100 --slab 2 --w 0.2"
for ov in 0 1; do
  for ys in 1 5; do CELL[R_s2_ys${ys}_ov$ov]="--design R --p 12 --n 100 --slab 2 --w 0.25 --yscales $ys --r-overlap $ov"; done
  for ys in 1 3; do CELL[R_s100_ys${ys}_ov$ov]="--design R --p 12 --n 100 --slab 100 --w 0.25 --yscales $ys --r-overlap $ov"; done
done

for c in "${!CELL[@]}"; do
  # pure dimension: mutually orthogonal nuisance; prior weight fixed and scaled; default and oracle starts
  for P in 0 30 100 300 1000; do
    t=G_${c}_diag_P$P
    [ $P -ge 300 ] && T=10:00:00 || T=4:00:00
    sub $t $T "$HDR; $PY diagnostics/overlap_map.py ${CELL[$c]} $COMMON --nuis diag --P $P --wmodes fixed scaled \
      --inits default oracle --out $R/$t.csv"
  done
  # realistic p > n: collinear nuisance in the block's orthogonal complement, prior weight scaled, P <= 2n (as RQ1)
  case $c in *_n25) PS="30 50";; *) PS="30 100 200";; esac
  for P in $PS; do
    t=G_${c}_coll_P$P
    sub $t 4:00:00 "$HDR; $PY diagnostics/overlap_map.py ${CELL[$c]} $COMMON --nuis coll --P $P --wmodes scaled \
      --inits default --out $R/$t.csv"
  done
done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
