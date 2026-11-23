#!/bin/bash
# Plan sec. 50: the prior as a design choice. Semi-synthetic (SuSiE protocol on the real X) full-data fits, F3J,
# D in {S, 10}; the paired partners are sec. 46 (sss estimator, spike 0.1) and sec. 47 X32 (exact sss, spike 0.1).
#   T   Student-t pilot: diabetes X, S {1,3,5} x PVE {0.3,0.6}, replicates 1-10, df {1,3} x scale {1,0.1,0.01}
#   1   stage 1a: diabetes and PCOS X, S {1,3,5,10} x PVE {0.3,0.6}, replicates 1-20, seven variants:
#       sss estimator spike 0.03/0.01, exact sss spike 0.03/0.01, standard ss (slab weight D/p) spike 0.1/0.03/0.01
#   T1  stage 1a for Student-t (user 2026-11-22: "keep testing student"): the stage-1a grid, df {1,3} x scale
#       {0.03,0.01,0.003} (the pilot's best scale 0.01 was at the edge of its grid)
#   B   stage 1b: RQ1 benchmark (prior_rq1.py, E1c setting, seeds 1-25), all stage-1 variants incl. Student-t
#   E   sec. 50.4, empirical-Bayes Student-t (scale fitted inside the loop, started wide at 1): stage-1a grid (PE_*)
#       and RQ1 (B_student_eb*), three variants: scale only with df 3, scale only with df 1, df and scale (df <= 10)
#   C   stage 1c (sec. 50.6, user 2026-11-22: "ok go"): seven designs x six priors, seeds 1-25 (prior_rq1.py):
#       distinct blocks, latent rank 5, latent rank 10, block-equicorrelation, distinct classification, latent rank-2
#       classification, latent rank 2 at p 1000; priors: estimator 0.1 (shipped) and 0.03, exact 0.03, Student-t
#       df 3 scale 0.01, EB-scale Student-t df 3 (start 1), ss 0.03
#   D   sec. 50.8 diagnosis (user 2026-11-23: "investigate 2"): distinct design, 4 priors (estimator 0.03, exact 0.03,
#       ss 0.03, EB-scale Student-t) x start {default, planted} x noise {unit, eb, oracle}, seeds 1-25
# Usage: submit_prior.sh T|1|T1|B|E|C|D   (submits only). Results: ~/amvi_compare/results/prior/
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
T1)
  for x in diabetes pcos; do for S in 1 3 5 10; do for v in 0.3 0.6; do for df in 1 3; do for sc in 0.03 0.01 0.003; do
    t=PT_${x}_S${S}_pve${v}_df${df}_sc${sc}
    sub $t 3:00:00 "$HDR; $PY --prior student --t-df $df --t-scale $sc --xdata $x --S $S --pve $v --D $(dlist $S) \
      --reps $(seq -s ' ' 1 20) --out $R/$t.csv"
  done; done; done; done; done ;;
B)
  RQ="../.venv/bin/python diagnostics/prior_rq1.py --seeds $(seq -s ' ' 1 25)"
  for spec in "mc 0.1" "mc 0.03" "mc 0.01" "exact 0.1" "exact 0.03" "exact 0.01" "ss 0.1" "ss 0.03" "ss 0.01"; do
    set -- $spec; [ $1 = exact ] && tl=12:00:00 || tl=3:00:00
    t=B_$1_sp$2; sub $t $tl "$HDR; $RQ --prior $1 --var-spike $2 --out $R/$t.csv"
  done
  for df in 1 3; do for sc in 0.03 0.01 0.003; do
    t=B_student_df${df}_sc${sc}; sub $t 3:00:00 "$HDR; $RQ --prior student --t-df $df --t-scale $sc --out $R/$t.csv"
  done; done ;;
E)
  RQ="../.venv/bin/python diagnostics/prior_rq1.py --seeds $(seq -s ' ' 1 25)"
  for spec in "scale 3" "scale 1" "both 3"; do
    set -- $spec; eb=$1; df=$2
    for x in diabetes pcos; do for S in 1 3 5 10; do for v in 0.3 0.6; do
      t=PE_${x}_S${S}_pve${v}_eb${eb}_df${df}
      sub $t 3:00:00 "$HDR; $PY --prior student --t-eb $eb --t-df $df --t-scale 1 --xdata $x --S $S --pve $v \
        --D $(dlist $S) --reps $(seq -s ' ' 1 20) --out $R/$t.csv"
    done; done; done
    t=B_student_eb${eb}_df${df}; sub $t 3:00:00 "$HDR; $RQ --prior student --t-eb $eb --t-df $df --t-scale 1 --out $R/$t.csv"
  done ;;
C)
  RQ="../.venv/bin/python diagnostics/prior_rq1.py --seeds $(seq -s ' ' 1 25)"
  for design in "distinct|--generator distinct" "latr5|--generator latent --rank 5" "latr10|--generator latent --rank 10" \
                "equicorr|--generator equicorr" "distinctcls|--generator distinct --binarize" \
                "latr2cls|--generator latent --binarize" "latr2p1000|--generator latent --p 1000"; do
    dn=${design%%|*}; dargs=${design#*|}
    for spec in "mc_0.1|--prior mc --var-spike 0.1" "mc_0.03|--prior mc --var-spike 0.03" \
                "exact_0.03|--prior exact --var-spike 0.03" "t_df3_sc0.01|--prior student --t-df 3 --t-scale 0.01" \
                "teb_df3|--prior student --t-eb scale --t-df 3 --t-scale 1" "ss_0.03|--prior ss --var-spike 0.03"; do
      vn=${spec%%|*}; vargs=${spec#*|}
      case $vn in exact*) tl=12:00:00 ;; *) tl=3:00:00 ;; esac
      [ $dn = latr2p1000 ] && { case $vn in exact*) tl=23:00:00 ;; *) tl=8:00:00 ;; esac; }
      t=C_${dn}_${vn}; sub $t $tl "$HDR; $RQ $dargs $vargs --out $R/$t.csv"
    done
  done ;;
D)
  RQ="../.venv/bin/python diagnostics/prior_rq1.py --generator distinct --seeds $(seq -s ' ' 1 25)"
  for spec in "mc_0.03|--prior mc --var-spike 0.03" "exact_0.03|--prior exact --var-spike 0.03" \
              "ss_0.03|--prior ss --var-spike 0.03" "teb_df3|--prior student --t-eb scale --t-df 3 --t-scale 1"; do
    vn=${spec%%|*}; vargs=${spec#*|}
    case $vn in exact*) tl=12:00:00 ;; *) tl=3:00:00 ;; esac
    for init in default planted; do for nz in unit eb oracle; do
      t=D_${vn}_${init}_${nz}; sub $t $tl "$HDR; $RQ $vargs --init $init --noise $nz --out $R/$t.csv"
    done; done
  done ;;
*) echo "usage: submit_prior.sh T|1|T1|B|E|C|D"; exit 1 ;;
esac
echo "submitted: $(grep -vc FAILED $IDS)  failed: $(grep -c FAILED $IDS)"
