#!/bin/bash
# Stage B of the coupling test (special-issue plan sec. 18). Run on the RCI login node from anywhere;
# it only submits. Results: ~/amvi_compare/results/coup/{B1,B2,B3,B4,E3,dumps}; logs ~/amvi_compare/logs/coup.
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
A=$BASE/adaptive-mixture-spike-slab
R=$BASE/results/coup
L=$BASE/logs/coup
mkdir -p $R/B1 $R/B2 $R/B3 $R/B4 $R/E3 $R/dumps $L
IDS=$L/jobids.txt
SEEDS25="$(seq -s ' ' 1 25)"
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=2; unset PYTHONPATH"
TRAJ="../.venv/bin/python diagnostics/trajectory.py"

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 2 --mem=8G -t "$2" -J "cp_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

# label -> "estimator [extra flags]"
declare -A ARM=(
  [F0J]="f0_joint" [F0Dec]="f0_dec" [F0Ind16]="f0_ind --strat-draws 16" [F0IndK]="f0_ind"
  [F3J]="ngvi" [F3Dec]="ngvi_dec" [F3Ind16]="ngvi_dec --strat-draws 16"
  [probeJ]="current" [probeInd]="probe_ind")

# B1: RQ1 overlap sweep, all arms, 25 seeds, K 3/6/12/24, checkpoints 2k/6k
for a in F0J F0Dec F0Ind16 F0IndK F3J F3Dec F3Ind16; do for pr in sss ss; do for ov in 0 2 4 8; do
  t=B1_${a}_${pr}_ov$ov
  sub $t 8:00:00 "$HDR; $TRAJ --prior $pr --estimator ${ARM[$a]} --arm $a --overlap $ov --seeds $SEEDS25 \
    --K 3 6 12 24 --checkpoints 2000 6000 --dump-dir $R/dumps/$t --out $R/B1/$t.csv"
done; done; done

# B2: the old repulsion probe reproduced (classification, D 5, K 6, 4k steps, var_slab 32), 10 seeds
for a in probeJ probeInd F0J F0Dec F0Ind16 F0IndK; do for p in 100 500; do for ov in 0 2 4; do
  t=B2_${a}_p${p}_ov$ov
  sub $t 8:00:00 "$HDR; $TRAJ --prior sss --task classification --var-slab 32 --D 5 --p $p --estimator ${ARM[$a]} \
    --arm $a --overlap $ov --seeds 42 7 123 1 2 3 4 5 6 8 --K 6 --checkpoints 4000 --dump-dir $R/dumps/$t \
    --out $R/B2/$t.csv"
done; done; done

# B3: common and overlapping starts, overlaps 4 and 8 in one task, K 6/12/24
for init in common_zero common_screen overlap; do for a in F0J F0Dec F0IndK F3J F3Dec; do for pr in sss ss; do
  cmd="$HDR"
  for ov in 4 8; do
    t=B3_${init}_${a}_${pr}_ov$ov
    cmd="$cmd; $TRAJ --prior $pr --estimator ${ARM[$a]} --arm $a --init $init --overlap $ov --seeds $SEEDS25 \
      --K 6 12 24 --checkpoints 2000 6000 --dump-dir $R/dumps/$t --out $R/B3/$t.csv"
  done
  sub B3_${init}_${a}_${pr} 8:00:00 "$cmd"
done; done; done

# B4: multi-start ensembles of the same model (R one-component fits, 16 draws each) from random and
# from diverse screen starts; joint fits from the same diverse screen start
for a in F0Ind16 F3Ind16; do for init in default screen_diverse; do for pr in sss ss; do for ov in 0 4 8; do
  t=B4_${a}_${init}_${pr}_ov$ov
  sub $t 12:00:00 "$HDR; $TRAJ --prior $pr --estimator ${ARM[$a]} --arm ${a}_R --init $init --overlap $ov \
    --seeds $SEEDS25 --K 18 60 120 --checkpoints 2000 6000 --dump-dir $R/dumps/$t --out $R/B4/$t.csv"
done; done; done; done
for a in F0J F3J; do for pr in sss ss; do for ov in 0 4 8; do
  t=B4_${a}_screen_diverse_${pr}_ov$ov
  sub $t 8:00:00 "$HDR; $TRAJ --prior $pr --estimator ${ARM[$a]} --arm $a --init screen_diverse --overlap $ov \
    --seeds $SEEDS25 --K 6 12 24 --checkpoints 2000 6000 --dump-dir $R/dumps/$t --out $R/B4/$t.csv"
done; done; done

# E3 (Li et al. p = 10, 250 datasets): decoupled fits, F3 (planned) and F0 (added)
for e in ngvi_dec f0_dec; do for c in "one 0.7" "one 0.9" "two 0.7" "two 0.9" "two 0.99"; do set -- $c
  t=E3_${e}_$1_$2
  sub $t 6:00:00 "echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=2; export PYTHONPATH=$A/src; \
    ../.venv/bin/python diagnostics/amvi_e3.py --amvi-root $A --design $1 --rho $2 --arm ss --estimator $e \
    --ngvi-beta 0.02 --out $R/E3/${e}_$1_$2.csv"
done; done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
