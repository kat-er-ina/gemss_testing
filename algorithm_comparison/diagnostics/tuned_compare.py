"""Fully-tuned comparison: run ONE method over its OWN hyperparameter grid on a
synthetic overlap problem and emit sol_f1 for every grid point (one row per
hp-setting). Aggregation later picks each method's best HP by seed-mean -- so
every method, including the ones we expect to lose, is shown at its best.

Per coauthor (KH): tune tau AND the underlying selector for ALFESE (MI expected
best); tune alpha for ensemble/masking/stability; rho for EnumLasso; lambda for
BB-SSL; K (#components) for GEMSS.
"""
import argparse, csv, os, resource, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hard_data_factory import (generate_overlapping_dataset,
                                    generate_overlapping_dataset_equicorr)
from src.evaluation import calculate_structural_metrics, calculate_metrics, _jaccard
from src.wrappers.mechanism_gemss import MechanismGEMSSWrapper
from src.wrappers.sklearn_wrappers import (RandomizedLassoEnsembleWrapper,
                                           MaskingWrapper, StabilitySelectionWrapper)
from src.wrappers.enumlasso_wrapper import EnumLassoWrapper
from src.wrappers.alfese_wrapper import AlfeseWrapper
from bb_ssl_probe import run_bbssl
from ensemble_parity import cluster_supports

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

GEMSS_HP = dict(prior="sss", var_spike=0.1, lr=0.01, n_iter=6000, batch_size=16,
                weight_slab=0.9, weight_spike=0.1)

# methods whose fit exposes an ELBO / free energy (T1); all others log blank elbo/n_iter
GEMSS_METHODS = {"GEMSS", "GEMSSjoint", "GEMSSJ", "GEMSSJHI", "GEMSSJjoint"}


def _peak_mem_mb():
    """Process peak RSS in MB. ru_maxrss is bytes on macOS, kilobytes on Linux."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0


def _is_classification(y):
    """True iff y looks like binary 0/1 labels (classification / binarized target)."""
    u = np.unique(np.asarray(y, float)[~np.isnan(np.asarray(y, float))])
    return u.size <= 2 and set(np.round(u).astype(int).tolist()) <= {0, 1}


def held_out_predictive(X, y, support, seed):
    """Model-agnostic, ground-truth-free predictive score for a feature SUPPORT.

    70/30 train/test split (fixed by seed); OLS + R^2 for regression, logistic +
    ROC-AUC for binary y. Refits on ONLY the given support -> usable for every
    method's solutions, not just GEMSS. NaN if the support is empty or a fold
    degenerates (e.g. a single class in train/test for classification).
    """
    support = sorted({int(i) for i in support})
    if not support:
        return float("nan")
    Xf = np.nan_to_num(np.asarray(X, float), nan=0.0)[:, support]
    yv = np.asarray(y, float).ravel()
    keep = ~np.isnan(yv)
    Xf, yv = Xf[keep], yv[keep]
    if Xf.shape[0] < 5:
        return float("nan")
    try:
        Xtr, Xte, ytr, yte = train_test_split(Xf, yv, test_size=0.30, random_state=int(seed))
    except Exception:
        return float("nan")
    try:
        if _is_classification(yv):
            if np.unique(ytr).size < 2 or np.unique(yte).size < 2:
                return float("nan")
            clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
            return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
        reg = LinearRegression().fit(Xtr, ytr)
        return float(r2_score(yte, reg.predict(Xte)))
    except Exception:
        return float("nan")


def serialize_supports(sols):
    """m support tuples -> one CSV cell: ';'-joined, each a ','-joined sorted index list."""
    parts = []
    for s in sols.values():
        supp = s.get("support")
        if supp:
            parts.append(",".join(str(int(i)) for i in sorted(supp)))
    return ";".join(parts)


def grid(method):
    if method == "GEMSS":  return [{"K": k} for k in (3, 6, 12, 18, 24)]
    if method == "GEMSSjoint":  return [{"K": k} for k in (3, 6, 12, 18, 24)]  # learned variances (KH's default)
    if method == "GEMSSJ": return [{"K": 6, "lj": lj} for lj in (0, 0.3, 1, 3, 10, 100, 1000)]
    if method == "GEMSSJHI": return [{"K": 6, "lj": lj} for lj in (1000, 3000, 10000, 30000, 100000, 1000000)]
    if method == "GEMSSJjoint": return [{"K": 6, "lj": lj} for lj in (0, 0.3, 1, 3, 10, 100, 1000, 3000, 10000, 30000, 100000, 1000000)]  # joint + jaccard, full bracket
    if method == "ENS":    return [{"alpha": a} for a in (0.005, 0.01, 0.02, 0.05)]
    if method == "BBSSL":  return [{"l0": l0, "l1": l1} for l0 in (5, 10, 30) for l1 in (0.1, 0.5)]
    if method == "ENUM":   return [{"rho": r} for r in (0.01, 0.02, 0.05, 0.1)]
    if method == "STAB":   return [{"alpha": a} for a in (0.01, 0.05, 0.1)]
    if method == "ALFESE": return [{"sel": "mi", "tau": t}
                                   for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    if method == "MASK":   return [{"alpha": a} for a in (0.01, 0.05, 0.1)]
    raise ValueError(method)


def _record_gemss_meta(wrapper, meta):
    """Copy the GEMSS wrapper's post-fit diagnostics into the caller's meta dict."""
    if meta is None:
        return
    meta["elbo"] = getattr(wrapper, "elbo_", None)
    meta["n_iter"] = getattr(wrapper, "n_iter_", None)
    meta["n_iter_converged"] = getattr(wrapper, "n_iter_converged_", None)


def fit_sols(method, hp, X, y, D, m, nrestarts, meta=None):
    """Every method returns exactly m final solutions. GEMSS fits K (>= m)
    components as internal capacity, then CLUSTERS them down to m -- the same
    agglomerative-Jaccard consensus the ensemble applies to its restart cloud,
    so the reduction to m is identical across methods (no 'best-m-of-K' credit;
    GEMSS's near-equal component weights make weight-ranking meaningless).

    If ``meta`` (a dict) is passed, GEMSS branches fill it with elbo / n_iter /
    n_iter_converged for T1/T6 logging. Non-GEMSS branches leave it untouched.
    The return value is unchanged (the m-solution dict), so existing callers
    (e.g. diagnostics/kh_repro.py) that omit ``meta`` are unaffected."""
    p = X.shape[1]
    if method == "GEMSS":
        w = MechanismGEMSSWrapper("regression", "scalefixed", hp["K"], D, **GEMSS_HP)
        sols = w.fit(X, y); _record_gemss_meta(w, meta)
        supp = [tuple(sorted(s["support"])) for s in sols.values() if s.get("support")]
        return cluster_supports(supp, p, m, D)
    if method == "GEMSSjoint":                           # KH's default: LEARNED variances
        w = MechanismGEMSSWrapper("regression", "joint", hp["K"], D, **GEMSS_HP)
        sols = w.fit(X, y); _record_gemss_meta(w, meta)
        supp = [tuple(sorted(s["support"])) for s in sols.values() if s.get("support")]
        return cluster_supports(supp, p, m, D)          # K components -> m consensus
    if method in ("GEMSSJ", "GEMSSJHI", "GEMSSJjoint"):  # GEMSS + Jaccard diversity penalty
        mech = "joint" if method == "GEMSSJjoint" else "scalefixed"
        w = MechanismGEMSSWrapper("regression", mech, hp["K"], D,
                                  lambda_jaccard=hp["lj"], **GEMSS_HP)
        sols = w.fit(X, y); _record_gemss_meta(w, meta)
        supp = [tuple(sorted(s["support"])) for s in sols.values() if s.get("support")]
        return cluster_supports(supp, p, m, D)
    if method == "ENS":
        return RandomizedLassoEnsembleWrapper(n_solutions=m, sparsity=D, task="regression",
                                              alpha=hp["alpha"], n_restarts=nrestarts).fit(X, y)
    if method == "BBSSL":
        return run_bbssl(X, y, D, m, hp["l0"], hp["l1"], 200)[0]
    if method == "ENUM":
        return EnumLassoWrapper(n_solutions=m, sparsity=D, task="regression", rho=hp["rho"]).fit(X, y)
    if method == "STAB":
        return StabilitySelectionWrapper(n_solutions=m, sparsity=D, task="regression",
                                         alpha=hp["alpha"]).fit(X, y)
    if method == "ALFESE":
        return AlfeseWrapper(n_solutions=m, task="regression", selector_type=hp["sel"],
                             tau=hp["tau"], k=D).fit(X, y)
    if method == "MASK":
        return MaskingWrapper("lasso", m, D, "regression", alpha=hp["alpha"]).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--D", type=int, default=10)
    ap.add_argument("--m", type=int, default=3, help="# final solutions every method returns")
    ap.add_argument("--nrestarts", type=int, default=3000)
    ap.add_argument("--noise", type=float, default=0.05,
                    help="generator noise_std = Gaussian FEATURE noise added to X (RQ6 noise sweep)")
    ap.add_argument("--nan", type=float, default=0.0,
                    help="generator nan_ratio / missing fraction (RQ6 missingness sweep)")
    ap.add_argument("--structure", choices=["latent", "equicorr"], default="latent",
                    help="T4 covariance mechanism: 'latent' (default, shared low-rank "
                         "factor -- unchanged) or 'equicorr' (per-block equicorrelation).")
    ap.add_argument("--nonlinear", action="store_true",
                    help="T7(a): non-additive response (linear + factor interaction). "
                         "Off by default -> unchanged linear response.")
    ap.add_argument("--noise-dist", dest="noise_dist",
                    choices=["gauss", "t", "laplace"], default="gauss",
                    help="T7(b): distribution of the additive feature noise. "
                         "'gauss' (default) is unchanged; 't'/'laplace' are heavy-tailed.")
    ap.add_argument("--signal-frac", dest="signal_frac", type=float, default=0.6,
                    help="equicorr only: per-support signal fraction (support validity vs block independence)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="run all these seeds (paired across methods); one row per seed/hp")
    ap.add_argument("--torchseed", action="store_true",
                    help="also seed torch per fit (reduces GEMSS run-to-run variance)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def _dissim(sols):
        sets = [set(s["support"]) for s in sols.values() if s.get("support")]
        if len(sets) < 2:
            return 0.0
        ds = [1 - _jaccard(sets[a], sets[b]) for a in range(len(sets)) for b in range(a + 1, len(sets))]
        return float(np.mean(ds))

    f = open(args.out, "w", newline="")
    # New columns are appended AFTER the original ones so existing aggregators keep
    # reading the same fields: supports (T2 actual per-solution feature sets),
    # pred_mean/pred_best (T1 held-out predictive over the m sols), elbo/n_iter
    # (T1 free energy + iters-to-convergence, GEMSS only), fit_seconds/peak_mem_mb (T6).
    # scenario columns (structure/nonlinear/noise_dist) are appended at the VERY
    # END so every existing aggregator that reads fields by name is unaffected;
    # older CSVs simply lack them (DictReader.get -> None), and the defaults
    # (latent / False / gauss) mark rows produced by the unchanged code path.
    w = csv.DictWriter(f, fieldnames=["method", "n", "p", "overlap", "noise", "nan", "seed", "hp",
                                      "sol_f1", "union_f1", "dissim", "overlap_struct_err",
                                      "supports", "pred_mean", "pred_best",
                                      "elbo", "n_iter", "n_iter_converged",
                                      "fit_seconds", "peak_mem_mb",
                                      "structure", "nonlinear", "noise_dist"])
    w.writeheader(); f.flush()
    gen = (generate_overlapping_dataset_equicorr if args.structure == "equicorr"
           else generate_overlapping_dataset)
    extra = {"signal_frac": args.signal_frac} if args.structure == "equicorr" else {}
    for seed in args.seeds:
        X, y, truth, planted = gen(
            n_samples=args.n, n_features=args.p, n_solutions=3, sparsity=args.D,
            latent_rank=2, overlap=args.overlap, noise_std=args.noise, nan_ratio=args.nan,
            binarize=False, nonlinear=args.nonlinear, noise_dist=args.noise_dist, seed=seed, **extra)
        for hp in grid(args.method):
            if args.torchseed:
                import torch; torch.manual_seed(seed)
            meta = {}                         # GEMSS fit fills elbo / n_iter here
            supports_str, pred_mean, pred_best = "", float("nan"), float("nan")
            fit_seconds = float("nan")
            try:
                t_fit = time.perf_counter()
                sols = fit_sols(args.method, hp, X, y, args.D, args.m, args.nrestarts, meta=meta)
                fit_seconds = time.perf_counter() - t_fit
                m = calculate_structural_metrics(sols, planted)
                uf = calculate_metrics(sols, set(truth), X.shape[1])["F1_Score"]
                f1, err, dis = m["sol_f1"], m["overlap_struct_err"], _dissim(sols)
                # T2: actual per-solution supports; T1: model-agnostic held-out predictive
                supports_str = serialize_supports(sols)
                preds = [held_out_predictive(X, y, s["support"], seed)
                         for s in sols.values() if s.get("support")]
                preds = [v for v in preds if v == v]  # drop NaNs
                if preds:
                    pred_mean, pred_best = float(np.mean(preds)), float(np.max(preds))
            except Exception as e:
                print(f"  {args.method} sd={seed} hp={hp} ERR {str(e)[:120]}", flush=True)
                f1, err, uf, dis = float("nan"), float("nan"), float("nan"), float("nan")
            elbo = meta.get("elbo") if args.method in GEMSS_METHODS else ""
            n_iter = meta.get("n_iter") if args.method in GEMSS_METHODS else ""
            n_iter_conv = meta.get("n_iter_converged") if args.method in GEMSS_METHODS else ""
            w.writerow({"method": args.method, "n": args.n, "p": args.p, "overlap": args.overlap,
                        "noise": args.noise, "nan": args.nan,
                        "seed": seed, "hp": str(hp), "sol_f1": f1, "union_f1": uf,
                        "dissim": dis, "overlap_struct_err": err,
                        "supports": supports_str, "pred_mean": pred_mean, "pred_best": pred_best,
                        "elbo": "" if elbo is None else elbo,
                        "n_iter": "" if n_iter is None else n_iter,
                        "n_iter_converged": "" if n_iter_conv is None else n_iter_conv,
                        "fit_seconds": fit_seconds, "peak_mem_mb": _peak_mem_mb(),
                        "structure": args.structure, "nonlinear": int(args.nonlinear),
                        "noise_dist": args.noise_dist})
            f.flush()
            print(f"  {args.method} ov={args.overlap} sd={seed} hp={hp} -> "
                  f"u={uf:.3f} s={f1:.3f}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
