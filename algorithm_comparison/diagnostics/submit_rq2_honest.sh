#!/bin/bash
# Plan sec. 44: the paper's real-data table (tab:RQ2) with the feature selection inside the CV, next to the published
# protocol rerun in the same job. 3 datasets (diabetes, preterm; pcos = negative control) x 6 methods x 3 repetitions
# (rep 0 = the paper's folds). BB-SSL needs R (bb_ssl_bridge.R, packages in ~/R/bbssl_lib).
# Submits only. Results: ~/amvi_compare/results/rq2_honest/
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
R=$BASE/results/rq2_honest
L=$BASE/logs/rq2_honest
mkdir -p $R $L
IDS=$L/jobids.txt
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
RHDR="$HDR; . /etc/profile >/dev/null 2>&1; type module >/dev/null 2>&1 || . /opt/ohpc/admin/lmod/lmod/init/bash; \
export MODULEPATH=/opt/ohpc/pub/modulefiles; module load R/4.2.1-foss-2022a; export R_LIBS_USER=\$HOME/R/bbssl_lib; \
command -v Rscript || exit 3"
PY="../.venv/bin/python diagnostics/rq2_honest.py"

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem=8G -t "$2" -J "rq_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

for ds in diabetes preterm pcos; do for meth in GEMSS_default RandLasso_ensemble EnumLasso Masking_logistic ALFESE_mi BBSSL; do
  for rep in 0 1 2; do
    t=${ds}_${meth}_r$rep
    h="$HDR"; [ $meth = BBSSL ] && h="$RHDR"
    sub $t 12:00:00 "$h; unset PYTHONPATH; $PY --dataset $ds --method $meth --rep $rep --out $R/$t.csv"
  done
done; done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
