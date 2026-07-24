"""Semi-synthetic benchmark for the GEMSS rebuttal (Track T5).

WHY (reviewers #4/#1/#3): GEMSS is never tested on REAL data with KNOWN ground
truth, so correctness claims are limited. This builds the missing bridge: take a
REAL preprocessed design matrix X (real feature correlations), then PLANT known
overlapping sparse supports and SIMULATE the response y from them. Ground truth
is therefore known while the feature correlations are realistic.

DESIGN
------
* Real X: the same preprocessed metabolomics matrices scripts/run_rq3_clean.py
  uses (gemss/data/preprocessed_datasets/*.csv). We reuse its LEAK-FIXED loader
  (drop Sample Name / metadata__* / stratification_groups / response), then
  z-score every column.
* Planted supports: K supports of size D over the REAL columns, sharing a common
  core of `overlap` features plus private features -- the SAME overlap semantics
  as src.hard_data_factory.generate_overlapping_dataset (core + private layout).
* Response: y = sum_k X[:, support_k] @ beta_k + Gaussian noise, betas O(1)
  (regression). Every planted support is a genuine contributor; ground truth =
  the K planted supports (and their union).
* Methods: GEMSS (MechanismGEMSSWrapper mechanism='joint', K in {3,6,12}, then
  clustered to m=3 with ensemble_parity.cluster_supports), the randomised-Lasso
  ENSEMBLE (RandomizedLassoEnsembleWrapper), and ALFESE (AlfeseWrapper). Wrappers
  are imported directly from src.wrappers (NOT via tuned_compare) to avoid a race
  with a concurrent editor.
* Scoring: union-F1 (features recovered anywhere vs the union ground truth) AND
  per-support recovery via optimal Hungarian assignment against the planted
  supports -- recovery = fraction of planted supports matched at Jaccard >= 0.5,
  and matched-F1 = mean F1 of the Hungarian-matched pairs.

Usage (smoke):
  .venv/bin/python rebuttal/semisynth.py -d diabetes --overlaps 0 5 \
      --seeds 0 1 --n_iter 300 --Kgrid 3 6 --max_features 60

Usage (full):
  .venv/bin/python rebuttal/semisynth.py --overlaps 0 3 6 --seeds 0 1 2 3 4
"""

import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# repo root + diagnostics on the path (mirror run_rq3_clean / tuned_compare)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "diagnostics"))
warnings.filterwarnings("ignore")

from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

# method WRAPPERS imported directly (as tuned_compare does) -- NOT tuned_compare
from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper
from src.wrappers.sklearn_wrappers import RandomizedLassoEnsembleWrapper
from src.wrappers.alfese_wrapper import AlfeseWrapper
from src.evaluation import calculate_metrics, _jaccard
from ensemble_parity import cluster_supports

# Same GEMSS hyper-parameters as diagnostics/tuned_compare.py (n_iter overridable
# for smoke tests via --n_iter).
GEMSS_HP = dict(prior="sss", var_spike=0.1, lr=0.01, n_iter=6000, batch_size=16,
                weight_slab=0.9, weight_spike=0.1)

DATA_GLOB = os.path.join(_ROOT, "..", "..", "gemss", "data",
                         "preprocessed_datasets", "*.csv")


# --------------------------------------------------------------------------- #
# Real-data loader (reused from scripts/run_rq3_clean.py -- LEAK FIX)          #
# --------------------------------------------------------------------------- #
def load(path):
    """Return (X_features_df, feature_names). LEAK FIX: drop stratification_groups,
    ids and metadata. The real RESPONSE column is dropped too -- we plant our own.
    Same drop logic as scripts/run_rq3_clean.py:load()."""
    df = pd.read_csv(path)
    drop = [c for c in df.columns
            if c == "Sample Name" or c.startswith("metadata__")
            or c == "stratification_groups" or c == "response"]
    feat = df.drop(columns=drop)
    if feat.isnull().values.any():
        feat = feat.fillna(feat.mean())
    return feat, list(feat.columns)


# --------------------------------------------------------------------------- #
# Planting + simulation                                                        #
# --------------------------------------------------------------------------- #
def plant_supports(p, K, D, overlap, rng):
    """K supports of size D over the p REAL columns, sharing a core of `overlap`
    features plus private features. Mirrors the core+private layout of
    src.hard_data_factory.generate_overlapping_dataset. Returns dict
    {"solution_k": sorted list of column indices} and the union set."""
    if not (0 <= overlap < D):
        raise ValueError(f"overlap ({overlap}) must satisfy 0 <= overlap < D ({D}).")
    private = D - overlap
    union_size = overlap + K * private
    if union_size > p:
        raise ValueError(
            f"union of supports ({union_size}) exceeds available features ({p}); "
            "reduce K/D or overlap, or use a wider matrix.")
    all_signal = rng.choice(p, size=union_size, replace=False)
    core = all_signal[:overlap]
    rest = all_signal[overlap:]
    supports = {}
    for k in range(K):
        priv = rest[k * private:(k + 1) * private]
        supp = np.concatenate([core, priv]).astype(int)
        supports[f"solution_{k}"] = sorted(int(i) for i in supp)
    union = {int(i) for i in all_signal}
    return supports, union


def simulate_y(Xstd, supports, rng, noise_frac=0.3):
    """y = sum_k Xstd[:, support_k] @ beta_k + Gaussian noise. Betas O(1)
    (uniform 0.5..2, random sign). Noise std = noise_frac * std(signal)."""
    n = Xstd.shape[0]
    signal = np.zeros(n)
    for supp in supports.values():
        beta = rng.uniform(0.5, 2.0, size=len(supp)) * rng.choice([-1.0, 1.0], size=len(supp))
        signal = signal + Xstd[:, supp] @ beta
    noise_std = noise_frac * (np.std(signal) + 1e-12)
    y = signal + rng.normal(0.0, noise_std, size=n)
    return y.astype(float)


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def score(sols, planted, union, p, jacc_thresh=0.5):
    """union-F1 + per-support recovery (Hungarian). Returns (union_f1, recovery,
    matched_f1). recovery = fraction of planted supports matched at
    Jaccard>=jacc_thresh; matched_f1 = mean F1 of Hungarian-matched pairs."""
    union_f1 = calculate_metrics(sols, set(union), p)["F1_Score"]

    planted_sets = [set(s) for s in planted.values()]
    pred_sets = [set(s.get("support", [])) for s in sols.values() if s.get("support")]
    n_pl = len(planted_sets)
    if not pred_sets:
        return union_f1, 0.0, 0.0

    sim = np.array([[_jaccard(a, b) for b in pred_sets] for a in planted_sets])
    rows, cols = linear_sum_assignment(-sim)  # maximize total Jaccard
    matched_jacc = np.zeros(n_pl)
    matched_f1 = np.zeros(n_pl)
    for i, j in zip(rows, cols):
        matched_jacc[i] = sim[i, j]
        a, b = planted_sets[i], pred_sets[j]
        inter = len(a & b)
        rec = inter / len(a) if a else 0.0
        prec = inter / len(b) if b else 0.0
        matched_f1[i] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    recovery = float((matched_jacc >= jacc_thresh).mean())
    return union_f1, recovery, float(matched_f1.mean())


# --------------------------------------------------------------------------- #
# Method fits (all return exactly m solutions)                                 #
# --------------------------------------------------------------------------- #
def fit_gemss(Xstd, y, K, D, m, hp):
    """Fit GEMSS joint with K components, cluster down to m -- exactly the
    reduction tuned_compare.fit_sols uses for GEMSSjoint."""
    w = MechanismGEMSSWrapper("regression", "joint", K, D, **hp)
    sols = w.fit(Xstd, y)
    supp = [tuple(sorted(s["support"])) for s in sols.values() if s.get("support")]
    return cluster_supports(supp, Xstd.shape[1], m, D)


def fit_ensemble(Xstd, y, D, m, n_restarts):
    return RandomizedLassoEnsembleWrapper(
        n_solutions=m, sparsity=D, task="regression",
        alpha=0.05, n_restarts=n_restarts).fit(Xstd, y)


def fit_alfese(Xstd, y, D, m, tau):
    return AlfeseWrapper(
        n_solutions=m, task="regression", selector_type="mi", tau=tau, k=D).fit(Xstd, y)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default=None,
                    help="substring filter (e.g. diabetes / arabidopsis); default all non-pcos")
    ap.add_argument("-D", type=int, default=10, help="planted support size")
    ap.add_argument("-K", "--Kplant", type=int, default=3, help="# planted supports")
    ap.add_argument("-m", type=int, default=3, help="# solutions every method returns")
    ap.add_argument("--Kgrid", type=int, nargs="+", default=[3, 6, 12],
                    help="GEMSS component counts to sweep")
    ap.add_argument("--overlaps", type=int, nargs="+", default=[0, 3, 6],
                    help="planted core size (shared features) per overlap level")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--noise_frac", type=float, default=0.3)
    ap.add_argument("--n_iter", type=int, default=None,
                    help="override GEMSS n_iter (smoke tests)")
    ap.add_argument("--n_restarts", type=int, default=300,
                    help="ensemble bootstrap restarts")
    ap.add_argument("--alfese_tau", type=float, default=0.5)
    ap.add_argument("--max_features", type=int, default=None,
                    help="subset X to the first N real columns (smoke tests)")
    ap.add_argument("--methods", nargs="+", default=["GEMSS", "ENS", "ALFESE"],
                    choices=["GEMSS", "ENS", "ALFESE"])
    ap.add_argument("--out", default=os.path.join("results", "semisynth.csv"))
    args = ap.parse_args()

    hp = dict(GEMSS_HP)
    if args.n_iter is not None:
        hp["n_iter"] = args.n_iter

    paths = sorted(glob.glob(DATA_GLOB))
    # skip pcos (worst leak victim in run_rq3_clean; not central) unless asked
    paths = [p for p in paths if "pcos" not in os.path.basename(p).lower()]
    if args.dataset:
        paths = [p for p in paths if args.dataset.lower() in os.path.basename(p).lower()]
    if not paths:
        print(f"[!] no dataset matched under {DATA_GLOB}"); sys.exit(1)

    rows = []
    t_all = time.time()
    for path in paths:
        dname = os.path.basename(path).split("_")[0]
        feat, fnames = load(path)
        X = feat.values.astype(float)
        if args.max_features is not None:
            X = X[:, :args.max_features]
        Xstd = np.nan_to_num(StandardScaler().fit_transform(X))
        n, p = Xstd.shape
        print(f"\n=== {dname}: real X n={n} p={p} "
              f"(plant K={args.Kplant} D={args.D}, m={args.m}) ===")

        for ov in args.overlaps:
            # accumulate per-method metric lists across seeds
            acc = {}  # method_label -> {"uf": [], "rec": [], "mf1": [], "sec": []}

            def push(label, uf, rec, mf1, sec):
                d = acc.setdefault(label, {"uf": [], "rec": [], "mf1": [], "sec": []})
                d["uf"].append(uf); d["rec"].append(rec); d["mf1"].append(mf1); d["sec"].append(sec)

            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                try:
                    planted, union = plant_supports(p, args.Kplant, args.D, ov, rng)
                except ValueError as e:
                    print(f"  overlap={ov} seed={seed} SKIP: {e}"); continue
                y = simulate_y(Xstd, planted, rng, noise_frac=args.noise_frac)

                if "GEMSS" in args.methods:
                    for K in args.Kgrid:
                        t = time.time()
                        try:
                            sols = fit_gemss(Xstd, y, K, args.D, args.m, hp)
                            uf, rec, mf1 = score(sols, planted, union, p)
                        except Exception as e:
                            print(f"  GEMSS K={K} ov={ov} sd={seed} ERR {str(e)[:70]}")
                            continue
                        push(f"GEMSS_K{K}", uf, rec, mf1, time.time() - t)
                if "ENS" in args.methods:
                    t = time.time()
                    try:
                        sols = fit_ensemble(Xstd, y, args.D, args.m, args.n_restarts)
                        uf, rec, mf1 = score(sols, planted, union, p)
                        push("ENS", uf, rec, mf1, time.time() - t)
                    except Exception as e:
                        print(f"  ENS ov={ov} sd={seed} ERR {str(e)[:70]}")
                if "ALFESE" in args.methods:
                    t = time.time()
                    try:
                        sols = fit_alfese(Xstd, y, args.D, args.m, args.alfese_tau)
                        uf, rec, mf1 = score(sols, planted, union, p)
                        push("ALFESE", uf, rec, mf1, time.time() - t)
                    except Exception as e:
                        print(f"  ALFESE ov={ov} sd={seed} ERR {str(e)[:70]}")

            # GEMSS_best = best K by mean union-F1 (method shown at its best, as tuned_compare)
            gk = {k: v for k, v in acc.items() if k.startswith("GEMSS_K")}
            if gk:
                best = max(gk.items(), key=lambda kv: np.mean(kv[1]["uf"]))
                acc["GEMSS_best"] = best[1]

            # print + record a mean row per method
            print(f"  overlap={ov} (core={ov}/{args.D}) over {len(args.seeds)} seeds")
            print(f"    {'method':12s} {'union_F1':>9s} {'recovery':>9s} {'matched_F1':>11s} {'sec/fit':>8s}")
            order = [k for k in ("GEMSS_best",) if k in acc] + \
                    sorted(k for k in acc if k.startswith("GEMSS_K")) + \
                    [k for k in ("ENS", "ALFESE") if k in acc]
            for label in order:
                d = acc[label]
                if not d["uf"]:
                    continue
                muf, mrec, mmf1, msec = (float(np.mean(d["uf"])), float(np.mean(d["rec"])),
                                         float(np.mean(d["mf1"])), float(np.mean(d["sec"])))
                tag = "*" if label == "GEMSS_best" else " "
                print(f"  {tag} {label:12s} {muf:9.3f} {mrec:9.3f} {mmf1:11.3f} {msec:8.1f}")
                rows.append(dict(dataset=dname, n=n, p=p, overlap=ov, core=ov,
                                 D=args.D, Kplant=args.Kplant, m=args.m,
                                 method=label, n_seeds=len(d["uf"]),
                                 union_f1=muf, recovery=mrec, matched_f1=mmf1,
                                 sec_per_fit=msec))

    if rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nSaved {args.out}  (total wall-clock {time.time() - t_all:.1f}s)")


if __name__ == "__main__":
    main()
