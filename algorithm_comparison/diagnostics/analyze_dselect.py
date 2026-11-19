"""Plan sec. 42: analysis of d_select.py results.

Usage: analyze_dselect.py real <dir>      (R_*.csv, U_real.csv)
       analyze_dselect.py semi <dir>      (S_*.csv, U_semi.csv)
One-standard-error rule: D_min = argmin of the mean CV loss; D_1se = smallest D whose mean CV loss <= mean(D_min) +
se(D_min), se = s.d. over folds / sqrt(#folds). Real: pooled over 5 repeats x 5 folds, and per repeat. Semi: per
replicate (5 folds) and pooled per cell. '*' = 95 % CI of a paired difference excludes 0; 'e' = CI inside +-0.02.
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
GRID = [1, 2, 3, 4, 5, 7, 10, 15]


def short(x, band=0.02):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return "n/a"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); m = x.mean()
    return f"{m:+.3f}{'e' if (m - h > -band and m + h < band) else ('*' if (m - h > 0 or m + h < 0) else ' ')}"


def load(d, g):
    fs = [f for f in glob.glob(os.path.join(d, g)) if os.path.getsize(f) > 0]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def one_se(cv, loss="test_loss"):
    """cv: rows of one curve (all D). Returns (D_min, D_1se, table)."""
    t = cv.groupby("D")[loss].agg(["mean", "std", "count"])
    t["se"] = t["std"] / np.sqrt(t["count"])
    dmin = int(t["mean"].idxmin())
    d1 = int(min(D for D in t.index if t.loc[D, "mean"] <= t.loc[dmin, "mean"] + t.loc[dmin, "se"]))
    return dmin, d1, t


def susie_grid(k):
    """Grid value used for SuSiE's size: the smallest grid D >= n_cs (n_cs 0 -> 1)."""
    return min(D for D in GRID if D >= max(k, 1))


def real(d):
    g = load(d, "R_*.csv"); u = load(d, "U_real.csv")
    cv, full = g[g.kind == "cv"], g[g.kind == "full"]
    print(f"rows: cv {len(cv)}, full {len(full)}; SuSiE {len(u)}")
    chosen = {}
    for ds in sorted(cv.dataset.unique()):
        print(f"\n================ {ds}")
        if len(u) and ds in set(u.dataset):
            r = u[u.dataset == ds].iloc[0]
            print(f"SuSiE (L 10): n_cs {r.n_cs}, sizes {r.cs_sizes}, min purity {r.cs_purity_min}")
        cur = {}
        for arm in ["F3J", "F3Ens"]:
            c = cv[(cv.dataset == ds) & (cv.arm == arm)]
            if c.empty:
                continue
            dmin, d1, t = one_se(c)
            per = [one_se(c[c.rep == rp])[1] for rp in sorted(c.rep.unique())]
            f1 = c.groupby("D").test_f1.mean()
            cur[arm] = t.assign(f1=f1)
            chosen[(ds, arm)] = d1
            print(f"{arm}: D_min {dmin}, D_1se {d1}; per-repeat D_1se {per}")
        if cur:
            tab = pd.concat({a: v[["mean", "se", "f1"]] for a, v in cur.items()}, axis=1).round(3)
            print("CV held-out log-loss (mean, se over folds) and F1 of the training-chosen solution:")
            print(tab.to_string())
            w = cv[cv.dataset == ds].pivot_table(index=["D", "rep", "fold"], columns="arm", values="test_loss")
            if {"F3J", "F3Ens"} <= set(w.columns):
                print("paired F3J - F3Ens CV log-loss per D:", (w.F3J - w.F3Ens).groupby("D").apply(short).to_dict())
        f = full[full.dataset == ds]
        if not f.empty:
            M = ["nsol", "f1_best", "f1_mean", "f1_min", "dissim", "n_rashomon", "core", "core_rashomon"]
            print("full-data fits (mean over seeds; RQ3 best-of-11 F1; Rashomon = solutions within 1 se of the best):")
            print(f.groupby(["D", "arm"])[M].mean().unstack("arm").round(3).to_string())
    print("\n================ chosen D (pooled one-se rule)")
    print(pd.Series(chosen).unstack().to_string())


def semi(d):
    g = load(d, "S_*.csv"); u = load(d, "U_semi.csv")
    cv, full = g[g.kind == "cv"], g[g.kind == "full"]
    cell = ["xdata", "S", "pve"]
    print(f"rows: cv {len(cv)}, full {len(full)}; SuSiE {len(u)}")
    ch = []
    for key, c in cv.groupby(cell + ["arm", "rep"]):
        if c.D.nunique() < 2:
            continue
        dmin, d1, _ = one_se(c)
        ch.append(dict(zip(cell + ["arm", "rep"], key), D_min=dmin, D_1se=d1))
    ch = pd.DataFrame(ch)
    if ch.empty:
        print("no complete curves yet"); return
    T = ["sel_recall_proxy", "sel_share_true", "sel_full_cover", "recall_proxy_mean", "share_true_mean",
         "n_full_cover", "n_sig_full", "n_sig_all", "nsol", "core"]
    fr = full.copy(); fr["sel_full_cover"] = fr.sel_full_cover.astype(str).eq("True").astype(float)
    at1 = ch.merge(fr, left_on=cell + ["arm", "rep", "D_1se"], right_on=cell + ["arm", "rep", "D"])
    atS = fr[fr.D == fr.S]; at10 = fr[fr.D == 10]
    print("\n== chosen D per replicate (one-se rule on 5 folds) vs the true S; SuSiE n_cs")
    s1 = ch.groupby(cell + ["arm"]).agg(D_1se_mean=("D_1se", "mean"), D_1se_med=("D_1se", "median"),
                                        exact=("D_1se", lambda v: 0), n=("D_1se", "size"))
    s1["exact"] = ch.assign(e=ch.D_1se == ch.S).groupby(cell + ["arm"]).e.mean()
    s1["D_min_mean"] = ch.groupby(cell + ["arm"]).D_min.mean()
    print(s1.unstack("arm").round(2).to_string())
    if len(u):
        su = u.groupby(cell).agg(n_cs=("n_cs", "mean"), exact=("n_cs", lambda v: 0), power_proxy=("power_proxy", "mean"),
                                 coverage_proxy=("coverage_proxy", "mean"), cs_size=("cs_median_size", "mean"),
                                 n_sig_full=("n_sig_full", "mean"), n_sig_all=("n_sig_all", "mean"))
        su["exact"] = u.assign(e=u.n_cs == u.S).groupby(cell).e.mean()
        print("SuSiE:"); print(su.round(2).to_string())
    for lab, x in [("at the chosen D (D_1se per replicate)", at1), ("at D = S (oracle size)", atS), ("at D = 10", at10)]:
        print(f"\n== truth {lab}: mean over replicates")
        print(x.groupby(cell + ["arm"])[T].mean().unstack("arm").round(2).to_string())
        w = x.pivot_table(index=cell + ["rep"], columns="arm", values=["sel_recall_proxy", "n_sig_full", "n_sig_all"])
        out = {}
        for m in ["sel_recall_proxy", "n_sig_full", "n_sig_all"]:
            if (m, "F3J") in w and (m, "F3Ens") in w:
                out[f"{m} J-Ens"] = (w[(m, "F3J")] - w[(m, "F3Ens")]).groupby(cell).apply(short)
        print(pd.DataFrame(out).to_string())
    print("\n== pooled per-cell CV curve (MSE on standardised y), one-se choice")
    for key, c in cv.groupby(cell + ["arm"]):
        dmin, d1, t = one_se(c)
        print(dict(zip(cell + ["arm"], key)), f"D_min {dmin} D_1se {d1}", t["mean"].round(3).to_dict())


if __name__ == "__main__":
    {"real": real, "semi": semi}[sys.argv[1]](sys.argv[2])
