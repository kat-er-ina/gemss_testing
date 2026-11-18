#!/bin/bash
# Plan sec. 29 (experiment B): joint fit vs ensemble on the RQ1 generator across evidence strength:
# feature noise {0.05, 0.2, 0.5, 1.0} (the paper's RQ6 levels) x n {50, 100, 200}; p 200, overlap 4 and 8,
# priors ss and sss, K 6 and 12, 25 seeds. Submits only. Results: ~/amvi_compare/results/noisen/B_*.csv
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
R=$BASE/results/noisen
L=$BASE/logs/noisen
mkdir -p $R $L
IDS=$L/jobids.txt
SEEDS25="$(seq -s ' ' 1 25)"
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1; unset PYTHONPATH"
TRAJ="../.venv/bin/python diagnostics/trajectory.py"
declare -A ARM=([F3J]="ngvi" [F3Ens]="ngvi_dec --strat-draws 16")

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem=4G -t "$2" -J "nx_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

for a in F3J F3Ens; do for pr in ss sss; do for nz in 0.05 0.2 0.5 1.0; do for n in 50 100 200; do
  cmd="$HDR"
  for ov in 4 8; do
    t=B_${a}_${pr}_nz${nz}_n${n}_ov$ov
    cmd="$cmd; $TRAJ --prior $pr --estimator ${ARM[$a]} --arm $a --overlap $ov --noise $nz --n $n --p 200 \
      --seeds $SEEDS25 --K 6 12 --checkpoints 2000 6000 --out $R/$t.csv"
  done
  sub B_${a}_${pr}_nz${nz}_n${n} 6:00:00 "$cmd"
done; done; done; done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
