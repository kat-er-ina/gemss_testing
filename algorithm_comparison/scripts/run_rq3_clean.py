"""Clean RQ3 real-data leg (leak-fixed + KH's end-model F1).

Two corrections over the original RQ3:
  1. LEAK FIX: `stratification_groups` (added for GEMSS Explorer) uniquely
     determines `response` in every preprocessed dataset -> it is a perfect
     label leak that made PCOS hit F1=1.0 for ALL methods. We drop it (and
     ids / metadata) so the comparison is honest.
  2. KH's END-MODEL F1: instead of a single LogisticRegression, every
     candidate feature set is scored by the BEST of KH's 11-model registry
     (logistic_{l2,l1,en}, svm, knn, xgboost, rf, dtree, nb, lda, qda) under
     nested CV -- the same `evaluate_all_solutions` procedure that produced the
     paper's tab:rw_summary in section 5. This matches her real-world table.

PROTOCOL (matches tab:rw_summary, NOT the synthetic nested-select leg):
  features are selected on the full standardized data, then the END-MODEL is
  scored by nested CV on the fixed candidate set. This has selection bias
  (relative comparison only) but is exactly the procedure section 5 reports,
  so RQ3 and section 5 are commensurable. We FLAG this caveat in the output.

Datasets: diabetes + arabidopsis (PCOS dropped -- it was the worst leak victim
and KH's section 5 does not rely on it). Arabidopsis is tiny -> LOO outer CV.

Each method is clustered to m solutions (agglomerative Jaccard consensus, same
as the synthetic fair-comparison protocol) so GEMSS's K components and the
ensemble's restart cloud are scored on equal footing.

Usage: .venv/bin/python scripts/run_rq3_clean.py -d diabetes --methods GEMSS_default
       .venv/bin/python scripts/run_rq3_clean.py            # all datasets+methods
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

from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper
from src.wrappers.sklearn_wrappers import (
    MaskingWrapper,
    StabilitySelectionWrapper,
    RandomizedLassoEnsembleWrapper,
)
from src.wrappers.alfese_wrapper import AlfeseWrapper
from src.wrappers.enumlasso_wrapper import EnumLassoWrapper
from src.evaluation import _jaccard
from ensemble_parity import cluster_supports

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "gemss"))
from gemss.postprocessing import result_modeling as rm

DATA_GLOB = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                         "gemss", "data", "preprocessed_datasets", "*.csv")

GEMSS_HP = dict(prior="sss", var_slab=32.0, var_spike=0.1, lr=0.01, n_iter=6000,
                batch_size=16, weight_slab=0.9, weight_spike=0.1)

# KH's 11-model classification registry (best per candidate = the table number)
END_MODELS = rm._get_available_models("classification")

# PCOS dropped: worst leak victim + not central to section 5.
SKIP_DATASETS = {"pcos"}


class BBSSLAdapter:
    def __init__(self, K, D, lambda0=50.0, lambda1=0.5, nsample=200):
        self.K, self.D = K, D
        self.lambda0, self.lambda1, self.nsample = lambda0, lambda1, nsample

    def fit(self, X, y):
        from bb_ssl_probe import run_bbssl
        sols, _ = run_bbssl(X, np.asarray(y, dtype=float), self.D, self.K,
                            self.lambda0, self.lambda1, self.nsample)
        return sols


def load(path):
    """Return (df_features, y, feature_names). LEAK FIX: drop stratification_groups."""
    df = pd.read_csv(path)
    drop = [c for c in df.columns
            if c == "Sample Name" or c.startswith("metadata__")
            or c == "stratification_groups"]
    y = df["response"].to_numpy().astype(float)
    feat = df.drop(columns=drop + ["response"])
    # mean-impute any NaNs, keep column names
    if feat.isnull().values.any():
        feat = feat.fillna(feat.mean())
    return feat, y, list(feat.columns)


def factories(K, D):
    sf_hp = {k: v for k, v in GEMSS_HP.items() if k != "var_slab"}
    return {
        "GEMSS_default": lambda: MechanismGEMSSWrapper("classification", "joint", K, D, **GEMSS_HP),
        "GEMSS_scalefixed": lambda: MechanismGEMSSWrapper("classification", "scalefixed", K, D, **sf_hp),
        "Masking_logistic": lambda: MaskingWrapper("logistic", K, D, "classification", alpha=0.05),
        "BBSSL": lambda: BBSSLAdapter(K, D),
        "RandLasso_ensemble": lambda: RandomizedLassoEnsembleWrapper(n_solutions=K, sparsity=D, task="classification", alpha=0.05, n_restarts=300),
        "EnumLasso": lambda: EnumLassoWrapper(n_solutions=K, sparsity=D, task="classification", rho=0.02),
        "StabilitySelection": lambda: StabilitySelectionWrapper(sparsity=D, task="classification", alpha=0.05),
        "ALFESE_mi": lambda: AlfeseWrapper(n_solutions=K, task="classification", selector_type="mi", tau=1.0, k=D),
    }


def dissim_stats(supports):
    """Return (mean, min, max) of pairwise 1-Jaccard over the supports."""
    sets = [set(s) for s in supports if s]
    if len(sets) < 2:
        return 0.0, 0.0, 0.0
    pw = [1 - _jaccard(sets[a], sets[b])
          for a in range(len(sets)) for b in range(a + 1, len(sets))]
    return float(np.mean(pw)), float(min(pw)), float(max(pw))


def best_model_f1(solutions_named, df_feat, y, outer_cv):
    """Per solution, F1 of the BEST of KH's 11 end-models (nested CV). Returns
    dict sol_name -> (best_f1, best_model)."""
    per_sol = {name: (-1.0, None) for name in solutions_named}
    for model in END_MODELS:
        try:
            res = rm.evaluate_all_solutions(
                solutions_named, df_feat, y, model_name=model,
                apply_scaling="standard", outer_cv_folds=outer_cv,
                random_state=0, verbose=False, use_markdown=False)
        except Exception:
            continue
        if res is None or res.empty or "f1_score" not in res.columns:
            continue
        for name in solutions_named:
            if name in res.index:
                f1 = float(res.loc[name, "f1_score"])
                if f1 > per_sol[name][0]:
                    per_sol[name] = (f1, model)
    return per_sol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default=None)
    ap.add_argument("-K", type=int, default=6)
    ap.add_argument("-D", type=int, default=5)
    ap.add_argument("-m", type=int, default=3, help="cluster each method to m solutions")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--out", default="results/rq3_clean.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(DATA_GLOB))
    paths = [p for p in paths if os.path.basename(p).split("_")[0].lower() not in SKIP_DATASETS]
    if args.dataset:
        paths = [p for p in paths if args.dataset.lower() in os.path.basename(p).lower()]

    facs = factories(args.K, args.D)
    if args.methods:
        facs = {k: v for k, v in facs.items() if k in args.methods}

    rows = []
    for path in paths:
        name = os.path.basename(path).split("_")[0]
        df_feat, y, fnames = load(path)
        n, p = df_feat.shape
        outer_cv = "loo" if n < 40 else 5
        print(f"\n=== {name}: n={n} p={p} pos_rate={y.mean():.2f} "
              f"outer_cv={outer_cv} [select-on-full-data, end-model=best-of-{len(END_MODELS)}] ===")
        print(f"  {'method':18s} {'F1_best':>7s} {'F1_mean':>7s} {'dissim':>6s} {'sec':>6s}")
        Xstd = StandardScaler().fit_transform(df_feat.values)
        Xstd = np.nan_to_num(Xstd)
        for mname, fac in facs.items():
            t = time.time()
            try:
                sols = fac().fit(Xstd, y)
                supp = [tuple(sorted(s["support"])) for s in sols.values() if s.get("support")]
                if not supp:
                    print(f"  {mname:18s} no supports"); continue
                clustered = cluster_supports(supp, p, args.m, args.D)
                # cluster_supports returns {s0: {"support": [...]}, ...}
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
            except Exception as e:
                print(f"  {mname:18s} FAIL: {str(e)[:50]}"); continue
            print(f"  {mname:18s} F1[{fmin:.2f}-{fb:.2f}] dissim[{dlo:.2f}-{dhi:.2f}] nsol={len(f1s)} {time.time()-t:6.1f}")
            rows.append(dict(dataset=name, method=mname, nsol=len(f1s),
                             f1_min=fmin, f1_mean=fm, f1_best=fb,
                             dissim=dmean, dissim_lo=dlo, dissim_hi=dhi,
                             n=n, p=p, outer_cv=str(outer_cv), best_models=str(bm)))
    if rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
