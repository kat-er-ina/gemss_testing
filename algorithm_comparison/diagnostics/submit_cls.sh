#!/bin/bash
# Plan sec. 30 (experiment C): joint fit vs ensemble for classification (binarised RQ1 generator) with the Bernoulli
# likelihood in the F3 update (LogisticTarget). p 200, n 100, overlap {0, 4, 8}, priors ss and sss, slab s.d. 32,
# K 6 and 12, 25 seeds. Submits only. Results: ~/amvi_compare/results/cls/C_*.csv
set -u
STEP=${1:-fixed}                     # plan sec. 37: F3 step rule (fixed = as first run; invL = 1/L rule)
[ "$STEP" = fixed ] && SUF= || SUF=_$STEP
[ "$STEP" = fixed ] && CPS="2000 6000" || CPS="2000 6000 12000"
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
R=$BASE/results/cls$SUF
L=$BASE/logs/cls$SUF
mkdir -p $R $L
IDS=$L/jobids.txt
SEEDS25="$(seq -s ' ' 1 25)"
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1; unset PYTHONPATH"
TRAJ="../.venv/bin/python diagnostics/trajectory.py"
declare -A ARM=([F3J]="ngvi" [F3Ens]="ngvi_dec --strat-draws 16")

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem=4G -t "$2" -J "cl${SUF}_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

for a in F3J F3Ens; do for pr in ss sss; do for ov in 0 4 8; do
  t=C_${a}_${pr}_ov$ov
  sub $t 6:00:00 "$HDR; $TRAJ --task classification --var-slab 32 --prior $pr --estimator ${ARM[$a]} --arm $a \
    --overlap $ov --p 200 --seeds $SEEDS25 --K 6 12 --checkpoints $CPS --step-rule $STEP --out $R/$t.csv"
done; done; done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
