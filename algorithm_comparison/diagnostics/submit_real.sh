#!/bin/bash
# Plan sec. 31 (experiment D): joint fit vs ensemble on the paper's real datasets (RQ3 protocol), 20 torch seeds.
# Needs the gemss repository (data/preprocessed_datasets, gemss.postprocessing) at ~/amvi_compare/gemss, cloned
# from GitHub. Submits only. Results: ~/amvi_compare/results/real/D_*.csv
set -u
STEP=${1:-fixed}                     # plan sec. 37: F3 step rule
[ "$STEP" = fixed ] && SUF= || SUF=_$STEP
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
R=$BASE/results/real$SUF
L=$BASE/logs/real$SUF
mkdir -p $R $L
[ -d $BASE/gemss/data/preprocessed_datasets ] || { echo "missing $BASE/gemss (git clone the gemss repo there)"; exit 1; }
IDS=$L/jobids.txt
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1; unset PYTHONPATH"
PY="../.venv/bin/python"

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem=6G -t "$2" -J "rd${SUF}_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

for ds in diabetes arabidopsis preterm; do for a in F3J F3Ens; do
  t=D_${ds}_$a
  sub $t 8:00:00 "$HDR; $PY diagnostics/realdata_pairs.py --dataset $ds --arm $a --seeds $(seq -s ' ' 1 20) --step-rule $STEP --out $R/$t.csv"
done; done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
