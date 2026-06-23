"""PCOS-PRETERM curiosity leg (KH's RQ3 request): predict PRETERM birth from the
488 metabolites PLUS the PCOS diagnosis kept as an explicit feature, with the
stratification leak removed.

Same protocol as run_rq3_clean.py (leak fix + best-of-11 end-model F1 under nested
CV, each method clustered to m solutions) so the numbers are commensurable with
Table tab:rq3. The only differences:
  * we load the response=PRETERM file (target = preterm birth),
  * we KEEP the PCOS diagnosis as a feature (metadata__PCOS=pcos -> "PCOS_diagnosis"),
    which the RQ3 loader normally drops with the other metadata,
  * we still drop stratification_groups -- it PERFECTLY determines PRETERM here
    (4 groups, none mixed), the same leak that inflated PCOS to F1=1.0.

We also report whether the PCOS feature was selected by GEMSS (it should not be:
PCOS diagnosis is not associated with PRETERM, chi2 p=0.93).

Usage: .venv/bin/python scripts/run_pcos_preterm.py --out results/pcos_preterm.csv
"""

import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diagnostics"))
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

from scripts.run_rq3_clean import factories, dissim_stats, best_model_f1, END_MODELS
from ensemble_parity import cluster_supports

DATA_GLOB = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                         "gemss", "data", "preprocessed_datasets",
                         "pcos_*response=PRETERM*.csv")

PCOS_FEATURE = "PCOS_diagnosis"


def load_preterm(path):
    """(df_features, y, feature_names). Drop ids/metadata/leak, but KEEP PCOS
    diagnosis as the feature 'PCOS_diagnosis'."""
    df = pd.read_csv(path)
    drop = [c for c in df.columns
            if c == "Sample Name" or c == "stratification_groups"
            or c == "metadata__PCOS=control" or c == "response"]
    y = df["response"].to_numpy().astype(float)
    feat = df.drop(columns=drop)
    feat = feat.rename(columns={"metadata__PCOS=pcos": PCOS_FEATURE})
    if feat.isnull().values.any():
        feat = feat.fillna(feat.mean())
    return feat, y, list(feat.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-K", type=int, default=6)
    ap.add_argument("-D", type=int, default=5)
    ap.add_argument("-m", type=int, default=3)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--out", default="results/pcos_preterm.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print(f"[!] no PRETERM file matched {DATA_GLOB}"); sys.exit(1)
    path = paths[0]

    facs = factories(args.K, args.D)
    if args.methods:
        facs = {k: v for k, v in facs.items() if k in args.methods}

    df_feat, y, fnames = load_preterm(path)
    n, p = df_feat.shape
    outer_cv = "loo" if n < 40 else 5
    print(f"=== PCOS-PRETERM: n={n} p={p} pos_rate={y.mean():.2f} outer_cv={outer_cv} "
          f"[PCOS kept as feature '{PCOS_FEATURE}', leak removed] ===")
    Xstd = StandardScaler().fit_transform(df_feat.values)
    Xstd = np.nan_to_num(Xstd)

    rows = []
    for mname, fac in facs.items():
        t = time.time()
        try:
            sols = fac().fit(Xstd, y)
            supp = [tuple(sorted(s["support"])) for s in sols.values() if s.get("support")]
            if not supp:
                print(f"  {mname:18s} no supports"); continue
            clustered = cluster_supports(supp, p, args.m, args.D)
            cl_supports = [sorted(v["support"]) for v in clustered.values() if v.get("support")]
            solutions_named = {f"{mname}_s{i}": [fnames[j] for j in c]
                               for i, c in enumerate(cl_supports)}
            per_sol = best_model_f1(solutions_named, df_feat, y, outer_cv)
            f1s = [v[0] for v in per_sol.values() if v[0] >= 0]
            fb = max(f1s) if f1s else float("nan")
            fm = float(np.mean(f1s)) if f1s else float("nan")
            fmin = min(f1s) if f1s else float("nan")
            dmean, dlo, dhi = dissim_stats(cl_supports)
            bm = {k: v[1] for k, v in per_sol.items()}
            pcos_sel = sum(1 for c in solutions_named.values() if PCOS_FEATURE in c)
        except Exception as e:
            print(f"  {mname:18s} FAIL: {str(e)[:60]}"); continue
        print(f"  {mname:18s} F1[{fmin:.2f}-{fb:.2f}] dissim[{dlo:.2f}-{dhi:.2f}] "
              f"nsol={len(f1s)} PCOS_in={pcos_sel}/{len(solutions_named)} {time.time()-t:6.1f}s")
        rows.append(dict(dataset="pcos_preterm", method=mname, nsol=len(f1s),
                         f1_min=fmin, f1_mean=fm, f1_best=fb,
                         dissim=dmean, dissim_lo=dlo, dissim_hi=dhi,
                         pcos_selected=pcos_sel, n_sol_total=len(solutions_named),
                         n=n, p=p, outer_cv=str(outer_cv), best_models=str(bm)))
    if rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
