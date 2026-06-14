"""Real-data leg: run GEMSS variants + validated baselines on real datasets,
where there is NO ground truth, so we report the field's optics that apply:
predictive quality of the recovered solutions (CV) + their diversity/overlap.

The novel question for OUR claim: on real, correlated (metabolomics) data, do
methods find MULTIPLE, predictive, OVERLAPPING solutions? GEMSS can share
features across solutions; masking/ALFESE (tau=1) force disjoint sets. We report
per method: #solutions, #distinct, mean pairwise dissimilarity of solutions,
predictive F1 (CV; best & mean over solutions), runtime.

Datasets: the preprocessed CSVs in ../gemss/data/preprocessed_datasets (feature
columns + a 'response' column; 'Sample Name' and 'metadata__*' columns excluded).

Usage: uv run python scripts/run_real.py            # all datasets
       uv run python scripts/run_real.py -d diabetes
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

from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper
from src.wrappers.gemss_wrapper import GEMSSWrapper
from src.wrappers.sklearn_wrappers import MaskingWrapper, StabilitySelectionWrapper
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
    return X, y


def methods(K, D):
    return {
        "GEMSS_logistic": MechanismGEMSSWrapper("classification", "joint", K, D, **GEMSS_HP),
        "GEMSS_kernel": MechanismGEMSSWrapper("classification", "kernel", K, D, kernel_gamma=100.0, **GEMSS_HP),
        "GEMSS_scalefixed": MechanismGEMSSWrapper("classification", "scalefixed", K, D, **{k: v for k, v in GEMSS_HP.items() if k != "var_slab"}),
        "Masking_logistic": MaskingWrapper("logistic", K, D, "classification", alpha=0.05),
        "StabilitySelection": StabilitySelectionWrapper(sparsity=D, task="classification", alpha=0.05),
        "ALFESE_mi": AlfeseWrapper(n_solutions=K, task="classification", selector_type="mi", tau=1.0, k=D),
    }


def diversity(sols):
    sets = [set(s["support"]) for s in sols.values() if s.get("support")]
    n_distinct = len({frozenset(s) for s in sets})
    if len(sets) < 2:
        return len(sets), n_distinct, 0.0
    dis = [1 - _jaccard(sets[a], sets[b]) for a in range(len(sets)) for b in range(a + 1, len(sets))]
    return len(sets), n_distinct, float(np.mean(dis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default=None, help="substring filter (e.g. diabetes)")
    ap.add_argument("-K", type=int, default=6)
    ap.add_argument("-D", type=int, default=5)
    args = ap.parse_args()

    paths = sorted(glob.glob(DATA_GLOB))
    if args.dataset:
        paths = [p for p in paths if args.dataset.lower() in os.path.basename(p).lower()]
    rows = []
    for path in paths:
        name = os.path.basename(path).split("_")[0]
        X, y = load(path)
        print(f"\n=== {name}: n={X.shape[0]} p={X.shape[1]} pos_rate={y.mean():.2f} ===")
        print(f"  {'method':18s} {'#sol':>4s} {'#dist':>5s} {'dissim':>6s} {'predF1_best':>11s} {'predF1_mean':>11s} {'sec':>6s}")
        for mname, model in methods(args.K, args.D).items():
            t = time.time()
            try:
                sols = model.fit(X, y)
            except Exception as e:
                print(f"  {mname:18s} FAIL: {str(e)[:50]}")
                continue
            dur = time.time() - t
            ns, nd, dis = diversity(sols)
            pq = predictive_quality(X, y, sols, "classification")
            print(f"  {mname:18s} {ns:4d} {nd:5d} {dis:6.2f} {pq['pred_best']:11.2f} {pq['pred_mean']:11.2f} {dur:6.1f}")
            rows.append(dict(dataset=name, method=mname, n_sol=ns, n_distinct=nd,
                             dissim=dis, pred_best=pq["pred_best"], pred_mean=pq["pred_mean"],
                             runtime=dur))
    if rows:
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(rows).to_csv("results/real_data_results.csv", index=False)
        print("\nSaved results/real_data_results.csv")


if __name__ == "__main__":
    main()
