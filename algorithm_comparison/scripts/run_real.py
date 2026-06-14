"""Real-data leg: run GEMSS (default config) + validated baselines on real
datasets. No ground truth -> report predictive quality + solution diversity.

Two evaluation modes:
- default (full-data): select on all data, CV-score the selected sets. Fast but
  selection-bias optimistic (relative comparison only).
- --nested: NESTED cross-validation -- features are selected INSIDE each outer
  train fold and scored on the held-out fold (no selection bias). Features are
  standardized within each fold (StandardScaler fit on train) for EVERY method,
  so the default GEMSS is evaluated on its own footing (it expects standardized
  features). This is the rigorous number.

"Default GEMSS" = logistic (clf) / Gaussian (reg) + standardized + SSS
(var_spike 0.1, var_slab 32) + K~2*N_sol + no Jaccard. The anti-collapse mechanism
(kernel / scale_fixed) is an OPTIONAL add-on, included here as extra variants.

Datasets: preprocessed CSVs in ../gemss/data/preprocessed_datasets ('response'
column; 'Sample Name' / 'metadata__*' excluded).

Usage: uv run python scripts/run_real.py --nested
       uv run python scripts/run_real.py --nested -d diabetes
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
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper
from src.wrappers.sklearn_wrappers import (
    MaskingWrapper,
    StabilitySelectionWrapper,
    RandomizedLassoEnsembleWrapper,
)
from src.wrappers.alfese_wrapper import AlfeseWrapper
from src.evaluation import predictive_quality, _jaccard

DATA_GLOB = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                         "gemss", "data", "preprocessed_datasets", "*.csv")

GEMSS_HP = dict(prior="sss", var_slab=32.0, var_spike=0.1, lr=0.01, n_iter=6000,
                batch_size=16, weight_slab=0.9, weight_spike=0.1)


def load(path):
    df = pd.read_csv(path)
    drop = [c for c in df.columns if c == "Sample Name" or c.startswith("metadata__")]
    y = df["response"].to_numpy().astype(float)
    X = df.drop(columns=drop + ["response"]).to_numpy(dtype=float)
    # mean-impute any NaNs (datasets are mostly complete; keeps all methods aligned)
    if np.isnan(X).any():
        cm = np.nan_to_num(np.nanmean(X, 0)); ix = np.where(np.isnan(X)); X = X.copy(); X[ix] = np.take(cm, ix[1])
    return X, y


def factories(K, D):
    """name -> callable producing a FRESH wrapper (needed for per-fold refits)."""
    sf_hp = {k: v for k, v in GEMSS_HP.items() if k != "var_slab"}
    return {
        "GEMSS_default": lambda: MechanismGEMSSWrapper("classification", "joint", K, D, **GEMSS_HP),
        "GEMSS_kernel": lambda: MechanismGEMSSWrapper("classification", "kernel", K, D, kernel_gamma=100.0, **GEMSS_HP),
        "GEMSS_scalefixed": lambda: MechanismGEMSSWrapper("classification", "scalefixed", K, D, **sf_hp),
        "Masking_logistic": lambda: MaskingWrapper("logistic", K, D, "classification", alpha=0.05),
        "RandLasso_ensemble": lambda: RandomizedLassoEnsembleWrapper(n_solutions=K, sparsity=D, task="classification", alpha=0.05, n_restarts=300),
        "StabilitySelection": lambda: StabilitySelectionWrapper(sparsity=D, task="classification", alpha=0.05),
        "ALFESE_mi": lambda: AlfeseWrapper(n_solutions=K, task="classification", selector_type="mi", tau=1.0, k=D),
    }


def diversity(sols):
    sets = [set(s["support"]) for s in sols.values() if s.get("support")]
    nd = len({frozenset(s) for s in sets})
    if len(sets) < 2:
        return len(sets), nd, 0.0
    dis = [1 - _jaccard(sets[a], sets[b]) for a in range(len(sets)) for b in range(a + 1, len(sets))]
    return len(sets), nd, float(np.mean(dis))


def nested_eval(X, y, factory, n_splits=5, seed=0):
    """Per-solution test-F1 under nested CV (select in train fold, score on test);
    features standardized within each fold. Returns (best_mean, mean_mean, dissim)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_best, fold_mean, diss = [], [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)
        sols = factory().fit(Xtr, y[tr])
        f1s = []
        for s in sols.values():
            supp = s.get("support", [])
            if not supp:
                continue
            m = LogisticRegression(max_iter=1000).fit(Xtr[:, supp], y[tr])
            f1s.append(f1_score(y[te], m.predict(Xte[:, supp]), zero_division=0))
        if f1s:
            fold_best.append(max(f1s)); fold_mean.append(float(np.mean(f1s)))
        diss.append(diversity(sols)[2])
    return (float(np.mean(fold_best)) if fold_best else float("nan"),
            float(np.mean(fold_mean)) if fold_mean else float("nan"),
            float(np.mean(diss)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default=None)
    ap.add_argument("-K", type=int, default=6)
    ap.add_argument("-D", type=int, default=5)
    ap.add_argument("--nested", action="store_true", help="nested CV (rigorous, no selection bias)")
    args = ap.parse_args()

    paths = sorted(glob.glob(DATA_GLOB))
    if args.dataset:
        paths = [p for p in paths if args.dataset.lower() in os.path.basename(p).lower()]
    rows = []
    for path in paths:
        name = os.path.basename(path).split("_")[0]
        X, y = load(path)
        tag = "NESTED-CV test-F1" if args.nested else "full-data CV-F1 (optimistic)"
        print(f"\n=== {name}: n={X.shape[0]} p={X.shape[1]} pos_rate={y.mean():.2f} [{tag}] ===")
        print(f"  {'method':18s} {'F1_best':>7s} {'F1_mean':>7s} {'dissim':>6s} {'sec':>6s}")
        for mname, fac in factories(args.K, args.D).items():
            t = time.time()
            try:
                if args.nested:
                    fb, fm, dis = nested_eval(X, y, fac)
                else:
                    sols = fac().fit(X, y)
                    pq = predictive_quality(X, y, sols, "classification")
                    fb, fm, dis = pq["pred_best"], pq["pred_mean"], diversity(sols)[2]
            except Exception as e:
                print(f"  {mname:18s} FAIL: {str(e)[:45]}")
                continue
            print(f"  {mname:18s} {fb:7.2f} {fm:7.2f} {dis:6.2f} {time.time()-t:6.1f}")
            rows.append(dict(dataset=name, method=mname, f1_best=fb, f1_mean=fm, dissim=dis, nested=args.nested))
    if rows:
        os.makedirs("results", exist_ok=True)
        out = "results/real_data_nested.csv" if args.nested else "results/real_data_results.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
