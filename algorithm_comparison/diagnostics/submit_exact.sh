#!/bin/bash
# Plan sec. 47: GEMSS's F3 arms with the EXACT structured spike-and-slab density (src/wrappers/exact_sss.py) instead
# of GEMSS's Monte Carlo estimator; same data, seeds and D grid as secs. 42-46 (the Monte Carlo runs are the pairs).
#   R      real data (diabetes, preterm, pcos) x F3J/F3Ens x D grid: CV (5 x 5) + 5 full-data seeds, slab 32
#   X32    semi-synthetic (SuSiE protocol, real X) full-data fits only, slab 32, standardised y: F3J and F3Ens
#   X1     same, F3J, slab 1 (natural scale for standardised data)
#   X1o    same, F3J, slab 1, y divided by the true noise s.d. (GEMSS's unit noise variance then correct)
# Submits only. Results: ~/amvi_compare/results/exact/
set -u
BASE=$HOME/amvi_compare
G=$BASE/gemss_testing/algorithm_comparison
R=$BASE/results/exact
L=$BASE/logs/exact
mkdir -p $R $L
IDS=$L/jobids.txt
HDR="echo host=\$(hostname); cd $G; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; unset PYTHONPATH"
PY="../.venv/bin/python diagnostics/d_select.py"

sub() {  # sub NAME TIME CMD
  local id
  id=$(sbatch -p cpu -c 1 --mem=6G -t "$2" -J "ex_$1" -o "$L/$1_%j.out" --parsable --wrap "$3") \
    && echo "$id $1" >> $IDS || echo "FAILED $1" >> $IDS
}

for ds in diabetes preterm pcos; do for arm in F3J F3Ens; do for D in 1 2 3 4 5 7 10 15; do
  t=R_${ds}_${arm}_D$D
  [ $arm = F3Ens ] && tl=23:00:00 || tl=8:00:00
  sub $t $tl "$HDR; $PY real --prior exact --dataset $ds --arm $arm --D $D --reps 5 --full-seeds 1 2 3 4 5 --out $R/$t.csv"
done; done; done
for x in diabetes pcos; do for S in 1 3 5 10; do for v in 0.3 0.6; do
  c="semi --prior exact --no-cv --xdata $x --S $S --pve $v"
  t=X32_${x}_S${S}_pve${v}_F3J;  sub $t 16:00:00 "$HDR; $PY $c --arm F3J --reps $(seq -s ' ' 1 20) --out $R/$t.csv"
  t=X1_${x}_S${S}_pve${v}_F3J;   sub $t 16:00:00 "$HDR; $PY $c --arm F3J --var-slab 1 --reps $(seq -s ' ' 1 20) --out $R/$t.csv"
  t=X1o_${x}_S${S}_pve${v}_F3J;  sub $t 16:00:00 "$HDR; $PY $c --arm F3J --var-slab 1 --y-scale oracle --reps $(seq -s ' ' 1 20) --out $R/$t.csv"
  # F3Ens is ~7x slower with the exact prior: halves (diabetes X) or quarters (PCOS X) of the 20 replicates
  [ $x = diabetes ] && np=2 || np=4
  for q in $(seq 1 $np); do
    w=$((20 / np)); reps=$(seq -s ' ' $(((q - 1) * w + 1)) $((q * w)))
    t=X32_${x}_S${S}_pve${v}_F3Ens_q$q; sub $t 23:00:00 "$HDR; $PY $c --arm F3Ens --reps $reps --out $R/$t.csv"
  done
done; done; done
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
