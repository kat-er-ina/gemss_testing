"""Plan sec. 44: the paper's real-data table (RQ2, tab:RQ2) re-evaluated with the feature selection inside the CV.

Published protocol (REPLICATION.md: scripts/run_rq3_clean.py and run_pcos_preterm.py at -K 12 -m 8 -D 10): each method
is fit ONCE on all the standardised data; its supports are clustered to m = 8 solutions (cluster_supports); each
solution is scored by the best of the 11 end-models of gemss.postprocessing.result_modeling, each under stratified
5-fold CV (random_state 0, scaling inside folds, predictions pooled over folds, support-weighted F1 of the pooled
predictions). The selection is outside the CV. It is rerun here in the same job (rows protocol = 'published').

Selection inside the CV (same outer folds, rows protocol = 'inside'): in each outer training fold the method is fit
(X standardised on that fold) and clustered to m solutions; every solution x end-model pair is fit on the training
fold and predicts the held-out fold. Solutions differ between folds, so they are compared by rank: in each fold the
solutions are ranked by their inner-CV score, and the held-out predictions of rank r are pooled over the folds.
  f1_inside_best11 : per rank, F1 of the pooled held-out predictions for each end-model, best of 11 taken on the
                     held-out F1. Removes the feature-selection bias only; the max over end-models stays optimistic.
  f1_honest        : per rank, the end-model chosen in each fold by inner 5-fold CV on the training fold (pooled
                     inner predictions, weighted F1, as the paper's scorer); nothing is chosen on the held-out fold.
Ranks present in every fold are summarised (avg / min / max over ranks); nsol = mean distinct solutions per fold;
dissim = mean within-fold pairwise 1-Jaccard.

Methods: rq3.factories(12, 10) - GEMSS_default (GEMSS's own optimiser, as published), RandLasso_ensemble (RLE),
EnumLasso, Masking_logistic, ALFESE_mi, BBSSL. GEMSS is read out two ways from one fit: top-D by |mu| (the harness
behind the published numbers; GEMSS_default) and the outlier rule the paper's text describes (GEMSS_outlier:
|mu_j - mean(mu)| > 2.5 x mean absolute deviation, gemss.postprocessing.outliers.detect_outlier_features).
Datasets: diabetes, preterm (the table) and pcos (PCOS diagnosis; negative control, no signal per sec. 43).

Usage: rq2_honest.py --dataset diabetes --method GEMSS_default --rep 0 --out x.csv
       (rep r: outer folds StratifiedKFold(5, shuffle, random_state=r); rep 0 = the paper's folds; method seed r)
"""
import argparse, os, sys, time, warnings
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for q in (ROOT, HERE, os.path.join(ROOT, "scripts")):
    sys.path.insert(0, q)
warnings.filterwarnings("ignore")
import run_rq3_clean as rq3                       # noqa: E402
from d_select import real_data                    # noqa: E402
from ensemble_parity import cluster_supports      # noqa: E402
from gemss.postprocessing import result_modeling as rm          # noqa: E402
from gemss.postprocessing.outliers import detect_outlier_features   # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import StratifiedKFold   # noqa: E402
from sklearn.metrics import f1_score              # noqa: E402

K, D, M = 12, 10, 8
OUTLIER_COEFF = 2.5
MODELS = list(rq3.END_MODELS)


def wf1(y, p):
    return float(f1_score(y, p, average="weighted"))


def fit_method(method, X, y, seed):
    """Returns {variant: list of raw supports}."""
    torch.manual_seed(seed); np.random.seed(seed)
    fac = rq3.factories(K, D)[method]
    w = fac()
    sols = w.fit(X, y)
    out = {method: [tuple(sorted(s["support"])) for s in sols.values() if s.get("support")]}
    if method == "GEMSS_default":
        mu = np.asarray(w.mu_, float)
        out["GEMSS_outlier"] = [tuple(sorted(int(j) for j in detect_outlier_features(
            pd.Series(mu[k]), threshold_coeff=OUTLIER_COEFF)["features"])) for k in range(mu.shape[0])]
        out["GEMSS_outlier"] = [s for s in out["GEMSS_outlier"] if s]
    return out


def clustered(raw, p):
    return [sorted(v["support"]) for v in cluster_supports(raw, p, M, D).values() if v.get("support")] if raw else []


def predict(name, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    mdl = rm._create_model(name, "classification", 0)
    mdl.fit(sc.transform(Xtr), ytr)
    return mdl.predict(sc.transform(Xte))


def inner_scores(Xtr, ytr, s):
    """Weighted F1 of pooled inner 5-fold predictions per end-model (None if it fails)."""
    out = {}
    for name in MODELS:
        yt, yp = [], []
        try:
            for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, ytr):
                yp.extend(predict(name, Xtr[a][:, s], ytr[a], Xtr[b][:, s])); yt.extend(ytr[b])
            out[name] = wf1(yt, yp)
        except Exception:
            out[name] = None
    return out


def summarise(f1s):
    f1s = [v for v in f1s if v is not None and not np.isnan(v)]
    return (np.mean(f1s), min(f1s), max(f1s)) if f1s else (np.nan, np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["diabetes", "preterm", "pcos"])
    ap.add_argument("--method", required=True, choices=["GEMSS_default", "RandLasso_ensemble", "EnumLasso",
                                                        "Masking_logistic", "ALFESE_mi", "BBSSL"])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df, y, fnames = real_data(a.dataset)
    Xraw = df.values.astype(float); n, p = Xraw.shape
    rows = []
    # ---- published protocol (selection on all data), same seed
    t0 = time.perf_counter()
    raw_all = fit_method(a.method, np.nan_to_num(StandardScaler().fit_transform(Xraw)), y, a.rep)
    for var, raw in raw_all.items():
        cl = clustered(raw, p)
        named = {f"s{i}": [fnames[j] for j in c] for i, c in enumerate(cl)}
        f1s = [v[0] for v in rq3.best_model_f1(named, df, y, 5).values() if v[0] >= 0] if cl else []
        avg, lo, hi = summarise(f1s)
        rows.append(dict(dataset=a.dataset, method=var, rep=a.rep, protocol="published", nsol=len(cl),
                         f1_avg=avg, f1_min=lo, f1_max=hi, dissim=rq3.dissim_stats(cl)[0],
                         sizes=";".join(str(len(c)) for c in cl), solutions=";".join(",".join(map(str, c)) for c in cl),
                         seconds=time.perf_counter() - t0))
        print(f"  published {a.dataset} {var} rep {a.rep}: nsol {len(cl)} F1 {avg:.3f} [{lo:.3f}-{hi:.3f}]", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    # ---- selection inside the outer folds
    folds = list(StratifiedKFold(5, shuffle=True, random_state=a.rep).split(Xraw, y))
    per = {}        # variant -> list over folds of dict(test_idx, ranked list of (inner best F1, chosen model, preds by model))
    t0 = time.perf_counter()
    for f, (tr, te) in enumerate(folds):
        sc = StandardScaler().fit(Xraw[tr])
        Xtr, Xte = np.nan_to_num(sc.transform(Xraw[tr])), np.nan_to_num(sc.transform(Xraw[te]))
        for var, raw in fit_method(a.method, Xtr, y[tr], 1000 * a.rep + f).items():
            cl = clustered(raw, p)
            sols = []
            for s in cl:
                ins = inner_scores(Xtr, y[tr], s)
                ok = {k: v for k, v in ins.items() if v is not None}
                if not ok:
                    continue
                best = max(ok, key=lambda k: ok[k])      # first in MODELS order on ties
                preds = {}
                for name in ok:
                    try:
                        preds[name] = predict(name, Xtr[:, s], y[tr], Xte[:, s])
                    except Exception:
                        pass
                sols.append(dict(score=ok[best], model=best, preds=preds, support=s))
            sols.sort(key=lambda d: -d["score"])
            per.setdefault(var, []).append(dict(te=te, sols=sols, dissim=rq3.dissim_stats([d["support"] for d in sols])[0]))
        print(f"  inside {a.dataset} {a.method} rep {a.rep} fold {f}: {time.perf_counter() - t0:.0f}s", flush=True)
    for var, fl in per.items():
        R = min(len(d["sols"]) for d in fl)
        f1_honest, f1_b11 = [], []
        for r in range(R):
            yt = np.concatenate([y[d["te"]] for d in fl])
            ph = np.concatenate([d["sols"][r]["preds"].get(d["sols"][r]["model"], np.full(len(d["te"]), -1)) for d in fl])
            f1_honest.append(wf1(yt, ph))
            byM = []
            for name in MODELS:
                if all(name in d["sols"][r]["preds"] for d in fl):
                    byM.append(wf1(yt, np.concatenate([d["sols"][r]["preds"][name] for d in fl])))
            f1_b11.append(max(byM) if byM else np.nan)
        h, b = summarise(f1_honest), summarise(f1_b11)
        rows.append(dict(dataset=a.dataset, method=var, rep=a.rep, protocol="inside",
                         nsol=np.mean([len(d["sols"]) for d in fl]), ranks_all_folds=R,
                         f1_avg=h[0], f1_min=h[1], f1_max=h[2], f1_b11_avg=b[0], f1_b11_min=b[1], f1_b11_max=b[2],
                         f1_honest_by_rank=";".join(f"{v:.3f}" for v in f1_honest),
                         models_rank1=";".join(d["sols"][0]["model"] for d in fl if d["sols"]),
                         dissim=np.mean([d["dissim"] for d in fl]),
                         solutions_by_fold="|".join(";".join(",".join(map(str, s["support"])) for s in d["sols"]) for d in fl),
                         seconds=time.perf_counter() - t0))
        print(f"  inside {a.dataset} {var} rep {a.rep}: nsol {rows[-1]['nsol']:.1f} (ranks in all folds {R}) "
              f"honest F1 {h[0]:.3f} [{h[1]:.3f}-{h[2]:.3f}]  best-of-11 on held-out {b[0]:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
