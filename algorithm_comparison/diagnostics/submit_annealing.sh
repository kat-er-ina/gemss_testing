#!/bin/bash
# Plan sec. 19: A (overlap map, exact posterior) and F (annealing) on RQ1 and E3. Submits only.
# Results: ~/amvi_compare/results/ann/{A,F,E3F,dumps}; logs ~/amvi_compare/logs/ann.
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
AM=$BASE/adaptive-mixture-spike-slab
R=$BASE/results/ann
L=$BASE/logs/ann
mkdir -p $R/A $R/F $R/E3F $R/dumps $L
IDS=$L/jobids.txt
SEEDS25="$(seq -s ' ' 1 25)"
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1; unset PYTHONPATH"
PY="../.venv/bin/python"

sub() {  # sub NAME TIME CPUS CMD
  local id
  id=$(sbatch -p cpu -c "$3" --mem=6G -t "$2" -J "an_$1" -o "$L/$1_%j.out" --parsable --wrap "$4") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

# A: overlap map vs exact posterior; all arms and three schedules in each task
for d in M U; do for rho in 0 0.5 0.7 0.9 0.95 0.99; do for n in 25 50 100 400; do
  t=A_${d}_rho${rho}_n$n
  sub $t 6:00:00 1 "$HDR; $PY diagnostics/overlap_map.py --design $d --rho $rho --n $n --reps 30 \
    --anneals none 0.03/4000 0.3/1000 --out $R/A/$t.csv"
done; done; done

# F on RQ1: schedules x starts x arms x priors; overlaps 0/4/8 in one task
declare -A ARM=([F3J]="ngvi" [F3Dec]="ngvi_dec" [F3Ind16]="ngvi_dec --strat-draws 16" [F0J]="f0_joint" [F0Dec]="f0_dec")
run_f() {  # run_f ARM PRIOR INIT W0 T   (W0 = none for no annealing)
  local a=$1 pr=$2 init=$3 w0=$4 T=$5 cmd="$HDR; export OMP_NUM_THREADS=2" annflag=""
  [ "$w0" != none ] && annflag="--anneal-omega0 $w0 --anneal-steps $T"
  for ov in 0 4 8; do
    local t=F_${a}_${w0}_${T}_${init}_${pr}_ov$ov
    cmd="$cmd; $PY diagnostics/trajectory.py --prior $pr --estimator ${ARM[$a]} --arm $a --init $init $annflag \
      --overlap $ov --seeds $SEEDS25 --K 6 12 24 --checkpoints 1000 2000 4000 6000 --dump-dir $R/dumps/$t \
      --out $R/F/$t.csv"
  done
  sub F_${a}_${w0}_${T}_${init}_${pr} 12:00:00 2 "$cmd"
}
for a in F3J F3Dec F3Ind16 F0J F0Dec; do for pr in sss ss; do
  for init in default overlap; do for s in "0.03 1000" "0.03 4000" "0.3 1000" "0.3 4000"; do
    run_f $a $pr $init $s
  done; done
  run_f $a $pr overlap none 0
done; done

# F on E3: joint / decoupled F3 and joint F0, two schedules
for e in ngvi ngvi_dec f0_joint; do for s in "0.03 4000" "0.3 1000"; do set -- $s; w0=$1; T=$2
  for c in "one 0.7" "one 0.9" "two 0.7" "two 0.9" "two 0.99"; do set -- $c
    t=E3F_${e}_${w0}_${T}_$1_$2
    sub $t 6:00:00 2 "echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=2; export PYTHONPATH=$AM/src; \
      $PY diagnostics/amvi_e3.py --amvi-root $AM --design $1 --rho $2 --arm ss --estimator $e --ngvi-beta 0.02 \
      --anneal-omega0 $w0 --anneal-steps $T --out $R/E3F/$t.csv"
  done
done; done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
