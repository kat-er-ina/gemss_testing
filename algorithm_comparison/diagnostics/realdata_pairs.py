"""Plan sec. 31 (experiment D): joint fit (F3J) vs ensemble of independent fits (F3Ens) on the paper's real
datasets, with the paper's RQ3 protocol and hyperparameters (scripts/run_rq3_clean.py: leak fix, standardised X,
sss prior, slab s.d. 32, K 12 components, D 10, clustered to m 8 solutions, each solution scored by the best of
the 11 end-models under nested CV). Both arms use the Bernoulli likelihood (LogisticTarget). No ground truth:
descriptive comparison over torch seeds (same data, same seed for both arms).

Usage: realdata_pairs.py --dataset diabetes|arabidopsis|preterm --arm F3J|F3Ens --seeds 1 2 ... --out x.csv
"""
import argparse, glob, os, sys, time
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for q in (ROOT, HERE, os.path.join(ROOT, "scripts")):
    sys.path.insert(0, q)
import run_rq3_clean as rq3                       # noqa: E402  (loader, end-model F1, HP, data location)
import run_pcos_preterm as pt                     # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from ensemble_parity import cluster_supports      # noqa: E402
from src.wrappers.weightfix_gemss import WeightFixGEMSSWrapper   # noqa: E402

ARMS = dict(F3J=("ngvi", {}), F3Ens=("ngvi_dec", dict(strat_draws=16)))


def data(name):
    if name == "preterm":
        return pt.load_preterm(sorted(glob.glob(pt.DATA_GLOB))[0])
    path = [q for q in sorted(glob.glob(rq3.DATA_GLOB)) if os.path.basename(q).lower().startswith(name)]
    return rq3.load(path[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["diabetes", "arabidopsis", "preterm"])
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("-K", type=int, default=12)
    ap.add_argument("-D", type=int, default=10)
    ap.add_argument("-m", type=int, default=8)
    ap.add_argument("--n-iter", dest="n_iter", type=int, default=None, help="override 6000 (smoke tests only)")
    ap.add_argument("--no-endmodel", dest="endmodel", action="store_false", help="skip the end-model F1 (smoke)")
    ap.add_argument("--step-rule", dest="step_rule", choices=["fixed", "invL"], default="fixed", help="plan sec. 37")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df_feat, y, fnames = data(a.dataset)
    n, p = df_feat.shape
    outer_cv = "loo" if n < 40 else 5
    X = np.nan_to_num(StandardScaler().fit_transform(df_feat.values))
    hp = dict(rq3.GEMSS_HP)
    if a.n_iter:
        hp["n_iter"] = a.n_iter
    est, opts = ARMS[a.arm]
    opts = dict(opts, **({"step_rule": a.step_rule} if a.step_rule != "fixed" else {}))
    rows = []
    for seed in a.seeds:
        torch.manual_seed(seed)
        w = WeightFixGEMSSWrapper("classification", est, n_components=a.K, sparsity=a.D, estimator_opts=opts, **hp)
        t0 = time.perf_counter()
        sols = w.fit(X, y)
        secs = time.perf_counter() - t0
        raw = [tuple(sorted(s["support"])) for s in sols.values() if s.get("support")]
        cl = [sorted(v["support"]) for v in cluster_supports(raw, p, a.m, a.D).values() if v.get("support")]
        named = {f"{a.arm}_s{i}": [fnames[j] for j in c] for i, c in enumerate(cl)}
        f1s = []
        if a.endmodel:
            per = rq3.best_model_f1(named, df_feat, y, outer_cv)
            f1s = [v[0] for v in per.values() if v[0] >= 0]
        dmean, dlo, dhi = rq3.dissim_stats(cl)
        al = np.asarray(w.alpha_, float); al = al / al.sum()
        rows.append(dict(dataset=a.dataset, arm=a.arm, seed=seed, n=n, p=p, K=a.K, D=a.D, m=a.m, step_rule=a.step_rule,
                         nsol=len(cl), f1_min=min(f1s) if f1s else np.nan, f1_mean=np.mean(f1s) if f1s else np.nan,
                         f1_best=max(f1s) if f1s else np.nan, dissim=dmean, dissim_lo=dlo, dissim_hi=dhi,
                         n_distinct_raw=len(set(raw)), elbo=w.elbo_, alpha_max=float(al.max()),
                         perplexity=float(np.exp(-(al[al > 0] * np.log(al[al > 0])).sum())), fit_seconds=secs,
                         raw_supports=";".join(",".join(map(str, s)) for s in raw),
                         solutions=";".join(",".join(map(str, c)) for c in cl)))
        print(f"  {a.dataset} {a.arm} seed {seed}: nsol {len(cl)} F1 best {rows[-1]['f1_best']:.3f} "
              f"dissim {dmean:.2f} distinct raw {len(set(raw))} elbo {w.elbo_:.1f} {secs:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
