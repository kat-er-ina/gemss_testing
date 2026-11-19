#!/bin/bash
# Plan sec. 42: data-chosen D (5-fold CV, one-standard-error rule) for F3J and F3Ens; SuSiE's EB size as reference.
#   R  real data: diabetes, preterm, pcos (PCOS diagnosis), arabidopsis (ceiling check) x F3J/F3Ens x D grid;
#      5 CV repeats x 5 folds + 5 full-data seeds (RQ3 scoring) per job.
#   S  semi-synthetic, SuSiE's protocol on real X: X {diabetes, pcos} x S {1,3,5,10} x PVE {0.3,0.6} x F3J/F3Ens,
#      replicates 1-20 split in two jobs; all D in one job.
#   U  SuSiE: real (L 10) and semi (L 10 / 20), one job each.
# Submits only. Results: ~/amvi_compare/results/dselect/   Argument U: submit the SuSiE jobs only.
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
R=$BASE/results/dselect
L=$BASE/logs/dselect
mkdir -p $R $L
IDS=$L/jobids.txt
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; unset PYTHONPATH"
RHDR="$HDR; . /etc/profile >/dev/null 2>&1; type module >/dev/null 2>&1 || . /opt/ohpc/admin/lmod/lmod/init/bash; \
export MODULEPATH=/opt/ohpc/pub/modulefiles; module load R/4.2.1-foss-2022a; export R_LIBS_USER=\$HOME/R/lib-4.2.1; \
command -v Rscript || exit 3; unset PYTHONPATH"  # the R module sets PYTHONPATH
PY="../.venv/bin/python diagnostics/d_select.py"

sub() {  # sub NAME TIME MEM CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem="$3" -t "$2" -J "ds_$1" -o "$L/$1_%j.out" --parsable --wrap "$4") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

ONLY=${1:-all}
[ "$ONLY" = U ] || {
for ds in diabetes preterm pcos arabidopsis; do for arm in F3J F3Ens; do for D in 1 2 3 4 5 7 10 15; do
  t=R_${ds}_${arm}_D$D
  sub $t 4:00:00 4G "$HDR; $PY real --dataset $ds --arm $arm --D $D --reps 5 --full-seeds 1 2 3 4 5 --out $R/$t.csv"
done; done; done
for x in diabetes pcos; do for S in 1 3 5 10; do for v in 0.3 0.6; do for arm in F3J F3Ens; do
  for half in a b; do
    [ $half = a ] && reps="$(seq -s ' ' 1 10)" || reps="$(seq -s ' ' 11 20)"
    t=S_${x}_S${S}_pve${v}_${arm}_$half
    sub $t 10:00:00 4G "$HDR; $PY semi --xdata $x --S $S --pve $v --arm $arm --reps $reps --out $R/$t.csv"
  done
done; done; done; done
}
sub U_real 1:00:00 8G "$RHDR; $PY susie-real --out $R/U_real.csv"
sub U_semi 2:00:00 8G "$RHDR; $PY susie-semi --xdata diabetes pcos --S 1 3 5 10 --pve 0.3 0.6 \
--reps $(seq -s ' ' 1 20) --out $R/U_semi.csv"
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
