#!/bin/bash
# Plan sec. 50: the prior as a design choice. Semi-synthetic (SuSiE protocol on the real X) full-data fits, F3J,
# D in {S, 10}; the paired partners are sec. 46 (sss estimator, spike 0.1) and sec. 47 X32 (exact sss, spike 0.1).
#   T   Student-t pilot: diabetes X, S {1,3,5} x PVE {0.3,0.6}, replicates 1-10, df {1,3} x scale {1,0.1,0.01}
#   1   stage 1a: diabetes and PCOS X, S {1,3,5,10} x PVE {0.3,0.6}, replicates 1-20, seven variants:
#       sss estimator spike 0.03/0.01, exact sss spike 0.03/0.01, standard ss (slab weight D/p) spike 0.1/0.03/0.01
# Usage: submit_prior.sh T|1   (submits only). Results: ~/amvi_compare/results/prior/
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
R=$BASE/results/prior
L=$BASE/logs/prior
mkdir -p $R $L
IDS=$L/jobids.txt
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; unset PYTHONPATH"
PY="../.venv/bin/python diagnostics/d_select.py semi --no-cv --arm F3J"

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem=6G -t "$2" -J "pr_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}
dlist() { [ "$1" = 10 ] && echo 10 || echo "$1 10"; }

case "${1:-}" in
T)
  for S in 1 3 5; do for v in 0.3 0.6; do for df in 1 3; do for sc in 1 0.1 0.01; do
    t=T_diabetes_S${S}_pve${v}_df${df}_sc${sc}
    sub $t 3:00:00 "$HDR; $PY --prior student --t-df $df --t-scale $sc --xdata diabetes --S $S --pve $v \
      --D $(dlist $S) --reps $(seq -s ' ' 1 10) --out $R/$t.csv"
  done; done; done; done ;;
1)
  for x in diabetes pcos; do for S in 1 3 5 10; do for v in 0.3 0.6; do
    for spec in "mc 0.03" "mc 0.01" "exact 0.03" "exact 0.01" "ss 0.1" "ss 0.03" "ss 0.01"; do
      set -- $spec; pr=$1; sp=$2
      [ $pr = exact ] && tl=12:00:00 || tl=3:00:00
      t=P_${x}_S${S}_pve${v}_${pr}_sp${sp}
      sub $t $tl "$HDR; $PY --prior $pr --var-spike $sp --xdata $x --S $S --pve $v --D $(dlist $S) \
        --reps $(seq -s ' ' 1 20) --out $R/$t.csv"
    done
  done; done; done ;;
*) echo "usage: submit_prior.sh T|1"; exit 1 ;;
esac
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
