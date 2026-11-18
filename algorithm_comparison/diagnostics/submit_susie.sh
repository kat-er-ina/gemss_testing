#!/bin/bash
# Plan sec. 39: SuSiE (susieR, defaults, L = D = 10) on E's grid (RQ4: n {30,50,100} x p {1000,2000,5000} x overlap
# {2,6}, 15 seeds), B's grid (p 200: noise {0.05,0.2,0.5,1.0} x n {50,100,200} x overlap {4,8}, 25 seeds) and D (real
# data, RQ3 protocol, SuSiE-samp over 20 seeds). Same data as the GEMSS runs (same generator call and seeds).
# Needs susieR in ~/R/lib-4.2.1 (module R/4.2.1-foss-2022a). Submits only. Results: ~/amvi_compare/results/susie/
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
R=$BASE/results/susie
L=$BASE/logs/susie
mkdir -p $R $L
IDS=$L/jobids.txt
HDR="echo host=\$(hostname); source /etc/profile; module load R/4.2.1-foss-2022a; export R_LIBS_USER=\$HOME/R/lib-4.2.1; \
cd $G; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; unset PYTHONPATH"
PY="../.venv/bin/python diagnostics/susie_compare.py"

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem=8G -t "$2" -J "su_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

for p in 1000 2000 5000; do for n in 30 50 100; do for ov in 2 6; do
  t=E_susie_p${p}_n${n}_ov$ov
  sub $t 4:00:00 "$HDR; $PY synth --n $n --p $p --overlap $ov --noise 0.05 --seeds $(seq -s ' ' 1 15) --out $R/$t.csv"
done; done; done
for nz in 0.05 0.2 0.5 1.0; do for n in 50 100 200; do for ov in 4 8; do
  t=B_susie_nz${nz}_n${n}_ov$ov
  sub $t 2:00:00 "$HDR; $PY synth --n $n --p 200 --overlap $ov --noise $nz --seeds $(seq -s ' ' 1 25) --out $R/$t.csv"
done; done; done
for ds in diabetes arabidopsis preterm; do
  t=D_${ds}_susie
  sub $t 8:00:00 "$HDR; $PY real --dataset $ds --seeds $(seq -s ' ' 1 20) --out $R/$t.csv"
done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
