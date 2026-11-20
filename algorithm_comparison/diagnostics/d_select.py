"""Plan sec. 42: choosing the solution size D from data instead of fixing D = 10.

Selector: K-fold cross-validated held-out loss over a grid of D, one-standard-error rule (the smallest D whose CV loss
is within one standard error of the minimum; Breiman et al. 1984, Hastie et al. ESL sec. 7.10). The rule is applied
by the analysis script (analyze_dselect.py); this script writes per-fold and full-data rows.
Reference selector: SuSiE's own empirical-Bayes size (number of credible sets with L as an upper bound; Wang et al.
2020, sec. 3.3). GEMSS's ELBO is NOT a usable evidence across D: the sss prior's log-density is a Monte-Carlo average
over 100 random supports with Gaussian normalisers that treat the variance as a standard deviation. It is written
(column elbo) for completeness only.

Plan sec. 47 adds --prior exact (the exact sss density, src/wrappers/exact_sss.py; ELBO then comparable across D),
--var-slab, --no-cv (semi: full-data fits only) and --y-scale oracle (semi: y divided by the true noise s.d.).
Plan sec. 50 (the prior as a design choice) adds --prior ss (GEMSS's standard spike-and-slab, slab weight D/p, i.e.
expected support size D) and --prior student (GEMSS's Student-t prior, --t-df, --t-scale), and --var-spike.

Arms: F3J (joint fit) and F3Ens (ensemble of independent fits), as realdata_pairs.py, sss prior, K 12, m 8, the
paper's RQ3 hyperparameters, 1/L step rule (plan sec. 37) throughout.

One fold: X standardised with training-fold statistics (and y, for regression); the arm fit on the training fold;
its K raw top-D supports clustered to m solutions (cluster_supports, as everywhere); one fixed end-model per solution
(L2-logistic, sklearn defaults C 1, for classification; OLS for regression) fit on the training fold; the solution
with the lowest training loss is kept (all solutions have size D, so this is the AIC choice) and its held-out loss
is the fold's loss (log-loss / MSE on the standardised y). Held-out loss of the other solutions is summarised too.

Modes
  real  --dataset diabetes|preterm|pcos|arabidopsis --arm F3J|F3Ens --D 1 2 ... --reps 5 --full-seeds 1 .. 5
        CV rows: reps x 5 stratified folds (fold torch seed 1000*rep + fold). Full rows: the arm on all data per
        seed, scored with the paper's RQ3 protocol (best-of-11 end-model F1, nested CV) and, per solution, the 5-fold
        CV log-loss of its L2-logistic model; n_rashomon = solutions within one standard error of the best one.
        pcos = the PCOS-diagnosis label (response=PCOS file, leak fix of run_rq3_clean.load).
  semi  --xdata diabetes|pcos --S 3 --pve 0.3 --reps 1 .. 10 --arm F3J|F3Ens --D ...
        SuSiE's simulation protocol (Wang et al. 2020, sec. 4) on a real design matrix (standardised): S effect
        variables uniformly at random among non-constant columns, b_j ~ N(0, 0.6^2), sigma^2 set so that the
        sample PVE equals phi, y ~ N(Xb, sigma^2 I). Data seed (xdata, S, phi, rep). One 5-fold CV per replicate plus
        a full-data fit (fold = -1). Truth columns on full rows, with proxy groups G_j = {j} + {k : |r_jk| >= 0.9}:
        per solution strict recall, proxy recall (effects whose G_j the solution hits), share of features inside
        some G_j; n_full_cover (solutions hitting every G_j); n_sig_all / n_sig_full = distinct effect signatures
        (per effect, which members of G_j are chosen; padding outside all G_j ignored) over solutions hitting
        at least one G_j / full-cover
        solutions = alternatives that differ in substance, not in padding.
  susie-real --datasets ... --out   SuSiE (susieR defaults, L 10) on standardised X and the 0/1 label.
  susie-semi --xdata ... --S ... --pve ... --reps ... --out
        SuSiE (L 10 for S <= 5, L 20 for S = 10, as the paper's two scenarios): n_cs, CS power / coverage (strict
        and proxy), sizes, purity; SuSiE-top alternatives (12 most probable one-per-CS supports, susie_compare.py)
        through the same truth columns.
"""
import argparse, glob, os, sys, time
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for q in (ROOT, HERE, os.path.join(ROOT, "scripts")):
    sys.path.insert(0, q)
import run_rq3_clean as rq3                       # noqa: E402  (loader, end-model F1, HP, data location)
import run_pcos_preterm as pt                     # noqa: E402
from realdata_pairs import ARMS                   # noqa: E402
from ensemble_parity import cluster_supports      # noqa: E402
from src.wrappers.weightfix_gemss import WeightFixGEMSSWrapper   # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import LogisticRegression, LinearRegression   # noqa: E402
from sklearn.model_selection import StratifiedKFold, KFold               # noqa: E402
from sklearn.metrics import log_loss, f1_score   # noqa: E402

D_GRID = [1, 2, 3, 4, 5, 7, 10, 15]
XCODE = dict(diabetes=1, pcos=2)
RHO = 0.9


def real_data(name):
    """(df_features, y, feature_names) with the leak fix; pcos = PCOS-diagnosis label."""
    if name == "preterm":
        return pt.load_preterm(sorted(glob.glob(pt.DATA_GLOB))[0])
    paths = sorted(glob.glob(rq3.DATA_GLOB))
    if name == "pcos":
        path = [q for q in paths if os.path.basename(q).startswith("pcos") and "response=PCOS" in q]
    else:
        path = [q for q in paths if os.path.basename(q).lower().startswith(name)]
    assert len(path) == 1, (name, path)
    return rq3.load(path[0])


def std(Xtr, Xte=None):
    sc = StandardScaler().fit(Xtr)
    f = lambda A: np.nan_to_num(sc.transform(A))
    return (f(Xtr), f(Xte)) if Xte is not None else f(Xtr)


def prior_hp(spec, D, p):
    """Plan sec. 50: the prior as a design choice. spec: the parsed arguments (prior, var_slab, var_spike, t_df,
    t_scale). Returns (selector kwargs overriding rq3.GEMSS_HP, exact flag)."""
    hp, pr = {}, getattr(spec, "prior", "mc")
    if getattr(spec, "var_slab", None) is not None:
        hp["var_slab"] = spec.var_slab
    if getattr(spec, "var_spike", None) is not None:
        hp["var_spike"] = spec.var_spike
    if pr == "ss":
        hp.update(prior="ss", weight_slab=D / p, weight_spike=1 - D / p)
    elif pr == "student":
        hp.update(prior="student", student_df=spec.t_df, student_scale=spec.t_scale)
    return hp, pr == "exact"


def fit_arm(X, y, task, arm, D, seed, K, m, n_iter=None, spec=None):
    hp = dict(rq3.GEMSS_HP)
    if n_iter:
        hp["n_iter"] = n_iter
    over, exact = prior_hp(spec, D, X.shape[1])
    hp.update(over)
    est, opts = ARMS[arm]
    torch.manual_seed(seed)
    w = WeightFixGEMSSWrapper(task, est, n_components=K, sparsity=D, estimator_opts=dict(opts, step_rule="invL"),
                              exact_sss=exact, **hp)
    t0 = time.perf_counter()
    sols = w.fit(X, y)
    secs = time.perf_counter() - t0
    raw = [tuple(sorted(s["support"])) for s in sols.values() if s.get("support")]
    cl = [sorted(v["support"]) for v in cluster_supports(raw, X.shape[1], m, D).values() if v.get("support")]
    return raw, cl, w.elbo_, secs


def pcols(a):
    """Row columns recording the prior settings (plan sec. 50)."""
    return dict(var_spike=a.var_spike or 0.1, t_df=a.t_df if a.prior == "student" else None,
                t_scale=a.t_scale if a.prior == "student" else None)


def endmodel(task):
    return LogisticRegression(C=1.0, max_iter=5000) if task == "classification" else LinearRegression()


def scores(task, Xtr, ytr, Xte, yte, sols):
    """Per solution: (train loss, test loss, test F1, test accuracy)."""
    out = []
    for s in sols:
        mdl = endmodel(task).fit(Xtr[:, s], ytr)
        if task == "classification":
            ptr, pte = mdl.predict_proba(Xtr[:, s])[:, 1], mdl.predict_proba(Xte[:, s])[:, 1]
            out.append((log_loss(ytr, ptr, labels=[0, 1]), log_loss(yte, pte, labels=[0, 1]),
                        f1_score(yte, pte > 0.5, zero_division=0), float(np.mean((pte > 0.5) == yte))))
        else:
            out.append((float(np.mean((mdl.predict(Xtr[:, s]) - ytr) ** 2)),
                        float(np.mean((mdl.predict(Xte[:, s]) - yte) ** 2)), np.nan, np.nan))
    return np.array(out)


def cv_row(task, Xraw, y, tr, te, arm, D, seed, a):
    Xtr, Xte = std(Xraw[tr], Xraw[te])
    ytr, yte = y[tr], y[te]
    if task == "regression":
        mu, sd = ytr.mean(), ytr.std()
        ytr, yte = (ytr - mu) / sd, (yte - mu) / sd
    raw, cl, elbo, secs = fit_arm(Xtr, ytr, task, arm, D, seed, a.K, a.m, a.n_iter, a)
    sc = scores(task, Xtr, ytr, Xte, yte, cl)
    i = int(np.argmin(sc[:, 0]))
    return dict(nsol=len(cl), n_distinct_raw=len(set(raw)), train_loss=sc[i, 0], test_loss=sc[i, 1],
                test_f1=sc[i, 2], test_acc=sc[i, 3], test_loss_mean=sc[:, 1].mean(), test_loss_min=sc[:, 1].min(),
                test_loss_max=sc[:, 1].max(), elbo=elbo, fit_seconds=secs)


def cv_logloss(X, y, s, folds=5):
    """5-fold CV log-loss (mean, s.e. over folds) of the L2-logistic model on features s (X already standardised)."""
    ls = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y):
        mdl = endmodel("classification").fit(X[tr][:, s], y[tr])
        ls.append(log_loss(y[te], mdl.predict_proba(X[te][:, s])[:, 1], labels=[0, 1]))
    return float(np.mean(ls)), float(np.std(ls, ddof=1) / np.sqrt(folds))


def core_size(sols):
    return len(set.intersection(*map(set, sols))) if sols else 0


def save(rows, out):
    pd.DataFrame(rows).to_csv(out, index=False)


# ---------------------------------------------------------------- real data
def run_real(a):
    df, y, fnames = real_data(a.dataset)
    Xraw = df.values.astype(float); n, p = Xraw.shape
    outer_cv = "loo" if n < 40 else 5
    base = dict(dataset=a.dataset, arm=a.arm, n=n, p=p, K=a.K, m=a.m, prior=a.prior, var_slab=a.var_slab or 32.0, **pcols(a))
    rows = []
    for D in a.D:
        for rep in range(1, a.reps + 1):
            for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(Xraw, y)):
                r = cv_row("classification", Xraw, y, tr, te, a.arm, D, 1000 * rep + f, a)
                rows.append(dict(base, D=D, kind="cv", rep=rep, fold=f, seed=1000 * rep + f, **r))
            print(f"  {a.dataset} {a.arm} D {D} rep {rep}: CV log-loss "
                  f"{np.mean([q['test_loss'] for q in rows[-5:]]):.3f}", flush=True)
            save(rows, a.out)
        X = std(Xraw)
        for seed in a.full_seeds:
            raw, cl, elbo, secs = fit_arm(X, y, "classification", a.arm, D, seed, a.K, a.m, a.n_iter, a)
            named = {f"s{i}": [fnames[j] for j in c] for i, c in enumerate(cl)}
            f1s = [v[0] for v in rq3.best_model_f1(named, df, y, outer_cv).values() if v[0] >= 0] if a.endmodel else []
            cvl = np.array([cv_logloss(X, y, c) for c in cl])
            b = int(np.argmin(cvl[:, 0])); rash = [c for c, (mu, _) in zip(cl, cvl) if mu <= cvl[b, 0] + cvl[b, 1]]
            dmean, _, _ = rq3.dissim_stats(cl)
            rows.append(dict(base, D=D, kind="full", rep=0, fold=-1, seed=seed, nsol=len(cl),
                             n_distinct_raw=len(set(raw)), f1_best=max(f1s) if f1s else np.nan,
                             f1_mean=np.mean(f1s) if f1s else np.nan, f1_min=min(f1s) if f1s else np.nan,
                             dissim=dmean, cvloss_best=cvl[b, 0], cvloss_best_se=cvl[b, 1],
                             cvloss_mean=cvl[:, 0].mean(), n_rashomon=len(rash), core=core_size(cl),
                             core_rashomon=core_size(rash), elbo=elbo, fit_seconds=secs,
                             solutions=";".join(",".join(map(str, c)) for c in cl),
                             rashomon=";".join(",".join(map(str, c)) for c in rash)))
            print(f"  {a.dataset} {a.arm} D {D} full seed {seed}: nsol {len(cl)} F1 best {rows[-1]['f1_best']:.3f} "
                  f"Rashomon {len(rash)} core {core_size(cl)} {secs:.0f}s", flush=True)
            save(rows, a.out)


# ---------------------------------------------------------------- semi-synthetic (SuSiE protocol)
def semi_data(xname, S, pve, rep, return_s2=False):
    df, _, _ = real_data(xname)
    X = std(df.values.astype(float)); n, p = X.shape
    rng = np.random.default_rng([XCODE[xname], S, int(round(100 * pve)), rep])
    cand = np.flatnonzero(X.std(0) > 0)
    eff = np.sort(rng.choice(cand, S, replace=False))
    b = np.zeros(p); b[eff] = rng.normal(0.0, 0.6, S)
    xb = X @ b
    s2 = xb.var() * (1 - pve) / pve
    y = xb + rng.normal(0.0, np.sqrt(s2), n)
    C = np.nan_to_num(np.abs(np.corrcoef(X.T)))
    G = {int(j): set(np.flatnonzero(C[j] >= RHO).tolist()) | {int(j)} for j in eff}
    return (X, y, [int(j) for j in eff], G, s2) if return_s2 else (X, y, [int(j) for j in eff], G)


def truth(sols, eff, G):
    if not sols:
        return dict(nsol=0)
    E, allg = set(eff), set().union(*G.values())
    per = []
    for s in sols:
        ss = set(s)
        cover = [bool(ss & G[j]) for j in eff]
        per.append((len(ss & E) / len(E), float(np.mean(cover)), len(ss & allg) / len(ss), all(cover),
                    tuple(frozenset(ss & G[j]) for j in eff)))
    U = set().union(*map(set, sols))
    return dict(nsol=len(sols), recall_strict_mean=np.mean([q[0] for q in per]),
                recall_proxy_mean=np.mean([q[1] for q in per]), share_true_mean=np.mean([q[2] for q in per]),
                n_full_cover=sum(q[3] for q in per), n_sig_all=len({q[4] for q in per if any(q[4])}),
                n_sig_full=len({q[4] for q in per if q[3]}),
                union_f1_strict=2 * len(U & E) / (len(U) + len(E)), _per=per)


def run_semi(a):
    rows = []
    for rep in a.reps:
        X, y, eff, G, s2 = semi_data(a.xdata, a.S, a.pve, rep, return_s2=True)
        n, p = X.shape
        base = dict(xdata=a.xdata, S=a.S, pve=a.pve, rep=rep, arm=a.arm, n=n, p=p, K=a.K, m=a.m, prior=a.prior,
                    var_slab=a.var_slab or 32.0, y_scale=a.y_scale, **pcols(a),
                    effects=",".join(map(str, eff)), proxy_sizes=",".join(str(len(G[j])) for j in eff))
        for D in a.D:
            for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=rep).split(X) if a.cv else []):
                r = cv_row("regression", X, y, tr, te, a.arm, D, 1000 * rep + f, a)
                rows.append(dict(base, D=D, kind="cv", fold=f, seed=1000 * rep + f, **r))
            # standardised y (default; GEMSS's unit noise variance then overstates the noise by 1/(1 - PVE)), or
            # 'oracle': y centred and divided by the true noise s.d., so the unit-variance likelihood is correct
            ys = (y - y.mean()) / (np.sqrt(s2) if a.y_scale == "oracle" else y.std())
            raw, cl, elbo, secs = fit_arm(X, ys, "regression", a.arm, D, rep, a.K, a.m, a.n_iter, a)
            t = truth(cl, eff, G)
            if cl:
                sc = scores("regression", X, ys, X, ys, cl); i = int(np.argmin(sc[:, 0])); q = t["_per"][i]
                t.update(sel_recall_strict=q[0], sel_recall_proxy=q[1], sel_share_true=q[2], sel_full_cover=q[3])
            t.pop("_per", None)
            rows.append(dict(base, D=D, kind="full", fold=-1, seed=rep, n_distinct_raw=len(set(raw)), elbo=elbo,
                             fit_seconds=secs, core=core_size(cl), solutions=";".join(",".join(map(str, c)) for c in cl),
                             **t))
            print(f"  {a.xdata} S {a.S} pve {a.pve} rep {rep} {a.arm} D {D} ({a.prior}): "
                  f"ELBO {elbo:.1f}  full: sel proxy recall "
                  f"{t.get('sel_recall_proxy', np.nan):.2f} n_sig_full {t.get('n_sig_full', 0)} {secs:.0f}s", flush=True)
            save(rows, a.out)


# ---------------------------------------------------------------- SuSiE references
def run_susie_real(a):
    from susie_compare import run_susie
    rows = []
    for ds in a.datasets:
        df, y, fnames = real_data(ds)
        res = run_susie([(std(df.values.astype(float)), y)], 10, a.rscript)[0]
        cs = res["cs"]
        rows.append(dict(dataset=ds, L=10, n_cs=len(cs), cs_sizes=";".join(str(len(c[2])) for c in cs),
                         cs_purity_min=min((c[1] for c in cs), default=np.nan),
                         cs_members="|".join(",".join(fnames[j] for j in c[2]) for c in cs),
                         converged=res["info"].get("converged"), V=res["info"].get("V")))
        print(f"  SuSiE {ds}: n_cs {len(cs)} sizes {rows[-1]['cs_sizes']}", flush=True)
    save(rows, a.out)


def run_susie_semi(a):
    from susie_compare import run_susie, cs_dists, top_supports
    rows = []
    for xd in a.xdata:
        for S in a.S:
            L = 10 if S <= 5 else 20
            for pve in a.pve:
                data = [semi_data(xd, S, pve, rep) for rep in a.reps]
                res = run_susie([(X, y) for X, y, _, _ in data], L, a.rscript)
                for rep, (X, y, eff, G), r in zip(a.reps, data, res):
                    cs = [c[2] for c in r["cs"]]; allg = set().union(*G.values())
                    hit_s = [any(j in c for c in cs) for j in eff]
                    hit_p = [any(G[j] & set(c) for c in cs) for j in eff]
                    top = [tuple(sorted(s)) for s in top_supports(cs_dists(r), 12)] if cs else []
                    cl = [sorted(v["support"]) for v in cluster_supports(top, X.shape[1], a.m, len(cs)).values()
                          if v.get("support")] if top else []
                    t = truth(cl, eff, G); t.pop("_per", None)
                    rows.append(dict(xdata=xd, S=S, pve=pve, rep=rep, arm="SuSiE", L=L, n_cs=len(cs),
                                     power_strict=np.mean(hit_s), power_proxy=np.mean(hit_p),
                                     coverage_strict=np.mean([bool(set(c) & set(eff)) for c in cs]) if cs else np.nan,
                                     coverage_proxy=np.mean([bool(set(c) & allg) for c in cs]) if cs else np.nan,
                                     cs_median_size=np.median([len(c) for c in cs]) if cs else np.nan,
                                     cs_purity_min=min((c[1] for c in r["cs"]), default=np.nan),
                                     converged=r["info"].get("converged"), effects=",".join(map(str, eff)),
                                     cs_members="|".join(",".join(map(str, c)) for c in cs), **t))
                print(f"  SuSiE {xd} S {S} pve {pve}: mean n_cs {np.mean([q['n_cs'] for q in rows[-len(a.reps):]]):.2f}",
                      flush=True)
                save(rows, a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "semi", "susie-real", "susie-semi"])
    ap.add_argument("--dataset", choices=["diabetes", "preterm", "pcos", "arabidopsis"])
    ap.add_argument("--datasets", nargs="+", default=["diabetes", "preterm", "pcos", "arabidopsis"])
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--D", type=int, nargs="+", default=D_GRID)
    ap.add_argument("--reps", type=int, nargs="+", default=[5], help="real: number of CV repeats; semi: replicate ids")
    ap.add_argument("--full-seeds", dest="full_seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--xdata", nargs="+", default=["diabetes"])
    ap.add_argument("--S", type=int, nargs="+", default=[3])
    ap.add_argument("--pve", type=float, nargs="+", default=[0.3])
    ap.add_argument("-K", type=int, default=12)
    ap.add_argument("-m", type=int, default=8)
    ap.add_argument("--n-iter", dest="n_iter", type=int, default=None, help="override 6000 (smoke tests only)")
    ap.add_argument("--no-endmodel", dest="endmodel", action="store_false", help="skip the RQ3 end-model F1 (smoke)")
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--prior", choices=["mc", "exact", "ss", "student"], default="mc",
                    help="sss with GEMSS's Monte Carlo estimator (default) or the exact density (sec. 47); standard "
                         "spike-and-slab with slab weight D/p or Student-t (sec. 50)")
    ap.add_argument("--var-spike", dest="var_spike", type=float, default=None, help="spike scale (default 0.1, GEMSS)")
    ap.add_argument("--t-df", dest="t_df", type=float, default=3.0, help="Student-t degrees of freedom (GEMSS: 3)")
    ap.add_argument("--t-scale", dest="t_scale", type=float, default=1.0, help="Student-t scale (GEMSS: 1)")
    ap.add_argument("--var-slab", dest="var_slab", type=float, default=None, help="slab scale (default 32, GEMSS)")
    ap.add_argument("--no-cv", dest="cv", action="store_false", help="semi: full-data fits only")
    ap.add_argument("--y-scale", dest="y_scale", choices=["std", "oracle"], default="std",
                    help="semi, full fits: standardise y (default) or divide by the true noise s.d.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.mode == "real":
        a.reps = a.reps[0]
        run_real(a)
    elif a.mode == "semi":
        a.xdata, a.S, a.pve = a.xdata[0], a.S[0], a.pve[0]
        run_semi(a)
    elif a.mode == "susie-real":
        run_susie_real(a)
    else:
        run_susie_semi(a)


if __name__ == "__main__":
    main()
