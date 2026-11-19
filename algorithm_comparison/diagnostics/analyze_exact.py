"""Plan sec. 47: exact structured spike-and-slab density vs GEMSS's Monte Carlo estimator.

Usage: analyze_exact.py semi <exact dir> <dselect dir>     (X32/X1/X1o_*.csv vs the MC full rows S_*.csv, U_semi.csv)
       analyze_exact.py real <exact dir> <dselect dir>     (R_*.csv in both)
'*' = 95 % CI of the paired difference excludes 0; 'e' = CI inside +-0.02.
ELBO choice = argmax over the D grid of the full-data fit's ELBO, per replicate / seed.
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_dselect import one_se    # noqa: E402

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)


def short(x, band=0.02):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return "n/a"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); m = x.mean()
    return f"{m:+.3f}{'e' if (m - h > -band and m + h < band) else ('*' if (m - h > 0 or m + h < 0) else ' ')}"


def load(d, g):
    fs = [f for f in glob.glob(os.path.join(d, g)) if os.path.getsize(f) > 0]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def semi(ex, mc):
    cell = ["xdata", "S", "pve"]
    T = ["sel_recall_proxy", "share_true_mean", "n_sig_full", "n_sig_all", "nsol", "core"]
    m = load(mc, "S_*.csv"); m = m[m.kind == "full"].assign(variant="mc32")
    e32, e1, e1o = load(ex, "X32_*.csv"), load(ex, "X1_*.csv"), load(ex, "X1o_*.csv")
    parts = [m] + [x.assign(variant=v) for x, v in [(e32, "ex32"), (e1, "ex1"), (e1o, "ex1o")] if len(x)]
    d = pd.concat(parts, ignore_index=True)
    u = load(mc, "U_semi.csv")
    print("rows per variant/arm:", d.groupby(["variant", "arm"]).size().to_dict())
    for D_lab, sel in [("D = S", d.D == d.S), ("D = 10", d.D == 10)]:
        x = d[sel]
        print(f"\n== truth at {D_lab}: mean over replicates, Monte Carlo (mc32) vs exact (ex32), slab 32")
        print(x[x.variant.isin(["mc32", "ex32"])].groupby(cell + ["arm", "variant"])[T].mean().unstack(["arm", "variant"])
              .round(2).to_string())
        w = x.pivot_table(index=cell + ["rep"], columns=["variant", "arm"], values=T)
        out = {}
        for t in ["sel_recall_proxy", "share_true_mean", "n_sig_full", "nsol"]:
            for arm in ["F3J", "F3Ens"]:
                if (t, "ex32", arm) in w and (t, "mc32", arm) in w:
                    out[f"{t} ex-mc {arm}"] = (w[(t, "ex32", arm)] - w[(t, "mc32", arm)]).groupby(cell).apply(short)
            if (t, "ex32", "F3J") in w and (t, "ex32", "F3Ens") in w:
                out[f"{t} J-Ens exact"] = (w[(t, "ex32", "F3J")] - w[(t, "ex32", "F3Ens")]).groupby(cell).apply(short)
        print("paired differences:"); print(pd.DataFrame(out).to_string())
    print("\n== ELBO choice of D (argmax over the grid, per replicate) vs S; SuSiE n_cs; mc32 = GEMSS's estimator")
    ch = d.loc[d.groupby(cell + ["variant", "arm", "rep"]).elbo.idxmax()]
    tab = ch.groupby(cell + ["variant", "arm"]).D.agg(["mean", "median", lambda v: 0]).rename(columns={"<lambda_0>": "exact"})
    tab["exact"] = ch.assign(e=ch.D == ch.S).groupby(cell + ["variant", "arm"]).e.mean()
    print(tab.unstack(["variant", "arm"]).round(2).to_string())
    if len(u):
        print("SuSiE n_cs:", u.groupby(cell).n_cs.mean().round(2).to_dict())
    print("\n== truth at the ELBO-chosen D (per replicate)")
    print(ch.groupby(cell + ["variant", "arm"])[["sel_recall_proxy", "share_true_mean", "n_sig_full"]].mean()
          .unstack(["variant", "arm"]).round(2).to_string())
    print("\n== mean ELBO by D (diabetes X, S 3, PVE 0.6), per variant/arm")
    q = d[(d.xdata == "diabetes") & (d.S == 3) & (d.pve == 0.6)]
    print(q.groupby(["variant", "arm", "D"]).elbo.mean().unstack("D").round(1).to_string())


def real(ex, mc):
    e, m = load(ex, "R_*.csv").assign(prior="exact"), load(mc, "R_*.csv").assign(prior="mc")
    d = pd.concat([e, m], ignore_index=True)
    d = d[d.dataset.isin(set(e.dataset))]
    cv, full = d[d.kind == "cv"], d[d.kind == "full"]
    for ds in sorted(set(e.dataset)):
        print(f"\n================ {ds}")
        for arm in ["F3J", "F3Ens"]:
            for pr in ["mc", "exact"]:
                c = cv[(cv.dataset == ds) & (cv.arm == arm) & (cv.prior == pr)]
                if c.D.nunique() < 2:
                    continue
                dmin, d1, t = one_se(c)
                per = [one_se(c[c.rep == r])[1] for r in sorted(c.rep.unique())]
                print(f"{arm} {pr:5s}: D_min {dmin} D_1se {d1} per-repeat {per}; CV log-loss by D {t['mean'].round(3).to_dict()}")
        w = cv[cv.dataset == ds].pivot_table(index=["arm", "D", "rep", "fold"], columns="prior", values="test_loss")
        if {"mc", "exact"} <= set(w.columns):
            print("paired exact - mc CV log-loss:", (w.exact - w.mc).groupby(["arm", "D"]).apply(short).unstack("D").to_string())
        f = full[full.dataset == ds]
        M = ["nsol", "n_rashomon", "core", "core_rashomon", "f1_mean", "elbo"]
        print("full-data fits (mean over seeds):")
        print(f.groupby(["D", "arm", "prior"])[M].mean().unstack(["arm", "prior"]).round(2).to_string())
        fe = f[f.prior == "exact"]
        if len(fe):
            ch = fe.loc[fe.groupby(["arm", "seed"]).elbo.idxmax()]
            print("ELBO choice of D (exact prior, per seed):", ch.groupby("arm").D.apply(list).to_dict())


if __name__ == "__main__":
    {"semi": semi, "real": real}[sys.argv[1]](sys.argv[2], sys.argv[3])
