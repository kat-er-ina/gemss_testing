#!/bin/bash
# Experiment E (plan section pending: F3 diverges at p >> n under ss, needs a decision): joint fit vs ensemble on the paper's RQ5 dimensionality grid, n {30, 50, 100} x
# p {1000, 2000, 5000} x overlap {2, 6}, 15 seeds, priors ss and sss, K 12, with fit seconds (one core each, so
# times are comparable). Submits only. Results: ~/amvi_compare/results/highp/E_*.csv
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
R=$BASE/results/highp
L=$BASE/logs/highp
mkdir -p $R $L
IDS=$L/jobids.txt
SEEDS15="$(seq -s ' ' 1 15)"
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1; unset PYTHONPATH"
TRAJ="../.venv/bin/python diagnostics/trajectory.py"
declare -A ARM=([F3J]="ngvi" [F3Ens]="ngvi_dec --strat-draws 16")

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem=8G -t "$2" -J "hp_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

for a in F3J F3Ens; do for pr in ss sss; do for p in 1000 2000 5000; do for n in 30 50 100; do for ov in 2 6; do
  t=E_${a}_${pr}_p${p}_n${n}_ov$ov
  [ $p -ge 5000 ] && T=16:00:00 || T=8:00:00
  sub $t $T "$HDR; $TRAJ --prior $pr --estimator ${ARM[$a]} --arm $a --overlap $ov --n $n --p $p \
    --seeds $SEEDS15 --K 12 --checkpoints 2000 6000 --out $R/$t.csv"
done; done; done; done; done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
