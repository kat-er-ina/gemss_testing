"""Plan sec. 50: the prior as a design choice — semi-synthetic screen (F3J, full-data fits, D in {S, 10}).

Usage: analyze_prior.py <prior dir> <dselect dir> <exact dir> [pilot|stage1|rq1|bench|diag]
  pilot   Student-t pilot (T_*.csv; diabetes X, S 1/3/5, replicates 1-10) vs the sss estimator (sec. 46 S_* rows) and
          the exact sss (sec. 47 X32_* rows) on the same replicates; applies the reading fixed in sec. 50.
  stage1  stage 1a (P_*.csv, Student-t grid PT_*.csv) plus the two spike-0.1 sss variants from secs. 46-47.
  rq1     stage 1b (B_*.csv, prior_rq1.py): union-F1, planted recovery, planted found; paired by seed vs the shipped
          default.
  bench   stage 1c (C_*.csv): per design (generator, rank, task, p), the same metrics plus planted found at Jaccard
          >= 0.5 and the share of signal features per solution; paired by seed vs the shipped default.
  diag    sec. 50.8 (D_*.csv): prior x start x noise; the readings stable / preferred / search vs model failure.
Variant = prior / spike s.d. (/ df, scale for Student-t). Truth metrics as in d_select.py: sel_recall_proxy (proxy
recall of the selected solution), share_true_mean, n_sig_full (fully correct near-optimal solutions), nsol.
Paired differences against the shipped default (sss estimator, spike 0.1): mean, '*' = 95 % CI excludes 0, 'e' = CI
inside +-0.02.
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
CELL = ["xdata", "S", "pve"]
T = ["sel_recall_proxy", "share_true_mean", "n_sig_full", "nsol"]
REF = "mc/0.1"


def short(x, band=0.02):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return "n/a"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); m = x.mean()
    return f"{m:+.3f}{'e' if (m - h > -band and m + h < band) else ('*' if (m - h > 0 or m + h < 0) else ' ')}"


def load(d, g):
    fs = [f for f in glob.glob(os.path.join(d, g)) if os.path.getsize(f) > 0]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def label(r):
    sp = f"{r.var_spike:g}"
    if r.prior == "student":
        eb = r.get("t_eb") if hasattr(r, "get") else None
        if isinstance(eb, str) and eb != "none":
            return f"t/eb-{eb}/df{r.t_df:g}/start{r.t_scale:g}"
        return f"t/df{r.t_df:g}/sc{r.t_scale:g}"
    return f"{r.prior}/{sp}"


def assemble(pdir, mcdir, exdir, pattern):
    mc = load(mcdir, "S_*.csv"); mc = mc[(mc.kind == "full") & (mc.arm == "F3J")].assign(prior="mc", var_spike=0.1)
    ex = load(exdir, "X32_*_F3J.csv"); ex = ex[ex.kind == "full"].assign(var_spike=0.1)
    new = pd.concat([load(pdir, p) for p in pattern], ignore_index=True)
    d = pd.concat([mc, ex, new], ignore_index=True)
    d = d[(d.D == d.S) | (d.D == 10)].copy()
    d["Dlab"] = np.where(d.D == d.S, "D=S", "D=10")
    d["variant"] = d.apply(label, axis=1)
    return d


def table(d, Dlab, cells):
    x = d[d.Dlab == Dlab]
    print(f"\n== {Dlab}: mean over replicates per cell")
    print(x.groupby(CELL + ["variant"])[["sel_recall_proxy", "share_true_mean", "n_sig_full"]].mean()
          .unstack("variant").round(2).to_string())
    w = x.pivot_table(index=CELL + ["rep"], columns="variant", values=T)
    print(f"\n== {Dlab}: mean over cells (each cell weighted equally)")
    m = x.groupby(CELL + ["variant"])[T].mean().groupby("variant").mean()
    print(m.sort_values("sel_recall_proxy", ascending=False).round(3).to_string())
    print(f"\n== {Dlab}: paired difference vs {REF}, per cell (proxy recall | share true | n_sig_full)")
    out = {}
    for v in sorted(set(x.variant) - {REF}):
        for t, ab in [("sel_recall_proxy", "rec"), ("share_true_mean", "share"), ("n_sig_full", "nsig")]:
            if (t, v) in w and (t, REF) in w:
                out[f"{v} {ab}"] = (w[(t, v)] - w[(t, REF)]).groupby(CELL).apply(short)
    print(pd.DataFrame(out).T.to_string())
    return m


def pilot(pdir, mcdir, exdir):
    d = assemble(pdir, mcdir, exdir, ["T_*.csv"])
    reps = sorted(set(d[d.prior == "student"].rep))
    d = d[(d.xdata == "diabetes") & d.S.isin([1, 3, 5]) & d.rep.isin(reps)]
    print("rows per variant:", d.groupby("variant").size().to_dict(), "replicates", reps)
    mS = table(d, "D=S", None)
    table(d, "D=10", None)
    t = mS[mS.index.str.startswith("t/")]
    best = t.sel_recall_proxy.idxmax()
    rec_ok = t.loc[best, "sel_recall_proxy"] >= mS.loc[REF, "sel_recall_proxy"] - 0.02
    sh_ok = t.loc[best, "share_true_mean"] >= mS.loc[REF, "share_true_mean"] - 0.02
    print(f"\n== Reading (sec. 50): best Student-t variant at D = S: {best}")
    print(f"   proxy recall {t.loc[best, 'sel_recall_proxy']:.3f} vs {REF} {mS.loc[REF, 'sel_recall_proxy']:.3f}"
          f" -> {'ok' if rec_ok else 'FAIL'};  share true {t.loc[best, 'share_true_mean']:.3f} vs"
          f" {mS.loc[REF, 'share_true_mean']:.3f} -> {'ok' if sh_ok else 'FAIL'}")
    print("   PASS: Student-t joins stage 1" if (rec_ok and sh_ok) else "   DROP Student-t")


def stage1(pdir, mcdir, exdir):
    d = assemble(pdir, mcdir, exdir, ["P_*.csv", "PT_*.csv", "PE_*.csv"])
    print("rows per variant:", d.groupby("variant").size().to_dict())
    for Dlab in ["D=S", "D=10"]:
        table(d, Dlab, None)


def rq1(pdir, *_):
    d = load(pdir, "B_*.csv")
    d["variant"] = d.apply(label, axis=1)
    print("seeds per variant:", d.groupby("variant").size().to_dict())
    M = ["union_f1", "planted_recovery", "n_planted_found", "n_distinct_raw", "fit_seconds"]
    out = d.groupby("variant")[M].mean()
    w = d.pivot_table(index="seed", columns="variant", values=["union_f1", "planted_recovery", "n_planted_found"])
    for t, ab in [("union_f1", "dUF1"), ("planted_recovery", "dplanted"), ("n_planted_found", "dfound")]:
        out[ab] = {v: short(w[(t, v)] - w[(t, REF)]) for v in out.index if v != REF and (t, v) in w}
    print(out.sort_values("union_f1", ascending=False).round(3).to_string())


def bench(pdir, *_):
    d = load(pdir, "C_*.csv")
    d["variant"] = d.apply(label, axis=1)
    d["design"] = d.apply(lambda r: f"{r.generator}" + (f" r{r['rank']}" if r.generator == "latent" else "")
                          + (" cls" if r.task == "classification" else "") + (f" p{r.p}" if r.p != 200 else ""), axis=1)
    M = ["union_f1", "planted_recovery", "n_planted_found", "n_planted_found50", "share_signal", "fit_seconds"]
    for ds_, x in d.groupby("design"):
        out = x.groupby("variant")[M].mean()
        out["seeds"] = x.groupby("variant").size()
        w = x.pivot_table(index="seed", columns="variant", values=["union_f1", "planted_recovery", "n_planted_found50"])
        for t, ab in [("union_f1", "dUF1"), ("planted_recovery", "dplanted"), ("n_planted_found50", "dfound50")]:
            out[ab] = {v: short(w[(t, v)] - w[(t, REF)]) for v in out.index if v != REF and (t, v) in w and (t, REF) in w}
        print(f"\n== {ds_}")
        print(out.sort_values("planted_recovery", ascending=False).round(3).to_string())


def diag(pdir, *_):
    d = load(pdir, "D_*.csv")
    d["variant"] = d.apply(label, axis=1)
    M = ["planted_recovery_raw", "n_planted_found_raw", "planted_recovery", "n_planted_found", "union_f1",
         "noise_var_final", "oracle_var", "elbo_model"]
    print(d.groupby(["variant", "init", "noise"])[M].mean().round(3).to_string())
    print("\n== readings per prior and noise: kept from planted start (raw, >= 0.8, of 3); found from default start;"
          " objective(planted) - objective(default), paired")
    for (v, nz), x in d.groupby(["variant", "noise"]):
        w = x.pivot_table(index="seed", columns="init", values=["n_planted_found_raw", "elbo_model"])
        kept = w[("n_planted_found_raw", "planted")].mean() if ("n_planted_found_raw", "planted") in w else np.nan
        found = w[("n_planted_found_raw", "default")].mean() if ("n_planted_found_raw", "default") in w else np.nan
        dobj = (short(w[("elbo_model", "planted")] - w[("elbo_model", "default")], band=0.0)
                if v.startswith("t/") is False and ("elbo_model", "planted") in w and ("elbo_model", "default") in w
                else "n/a (EB-scale t)")
        stable = kept >= 2
        pref = dobj.endswith("*") and dobj.startswith("+")
        verdict = ("search failure" if stable and pref and found < 0.5 else
                   "model failure" if not stable or (not dobj.startswith("n/a") and not pref) else "stable; preference n/a")
        print(f"  {v:22s} {nz:6s} kept {kept:.2f}  found {found:.2f}  d_objective {dobj:>12s}  -> {verdict}")


def temper(pdir, *_):
    """Sec. 50.10: random start, distinct design; F_* (lasso, temper-lasso, temper-oracle) with the D_* default-start
    partners (unit, oracle = fixed at the oracle). Readings fixed in sec. 50.10 before the results."""
    f = load(pdir, "F_*.csv")
    dd = load(pdir, "D_*.csv")
    dd = dd[(dd.init == "default") & dd.noise.isin(["unit", "oracle"]) & (dd.prior != "mc")]
    d = pd.concat([dd, f], ignore_index=True)
    d["variant"] = d.apply(label, axis=1)
    M = ["n_planted_found_raw", "planted_recovery_raw", "n_planted_found", "union_f1", "noise_var_final", "elbo_model"]
    order = ["unit", "lasso", "temper-lasso", "oracle", "temper-oracle"]
    g = d.groupby(["variant", "noise"])[M].mean().round(3)
    print(g.reindex([(v, n) for v in sorted(set(d.variant)) for n in order if (v, n) in g.index]).to_string())
    print("\nlasso / oracle noise variance (per seed): median %.2f, range %.2f-%.2f"
          % tuple(np.percentile(f.lasso_var / f.oracle_var, [50, 0, 100])) if len(f) else "")
    C = [("endpoint", "lasso", "unit"), ("path", "temper-lasso", "lasso"), ("estimate", "temper-oracle", "temper-lasso"),
         ("path@oracle", "temper-oracle", "oracle"), ("vs GEMSS", "temper-lasso", "unit")]
    print("\n== paired differences (seeds), '*' = 95 % CI excludes 0, 'e' = inside +-0.02: found_raw | recovery_raw | union-F1")
    for v, x in d.groupby("variant"):
        w = x.pivot_table(index="seed", columns="noise", values=M)
        for name, a, b in C:
            if (M[0], a) in w and (M[0], b) in w:
                cells = [short(w[(m, a)] - w[(m, b)]) for m in ("n_planted_found_raw", "planted_recovery_raw", "union_f1")]
                print(f"  {v:22s} {name:12s} {a:>13s} - {b:13s} " + "  ".join(f"{c:>8s}" for c in cells))
    print("\n== reading (sec. 50.10): a practical arm (lasso or temper-lasso) fixes the search if it finds >= 1.0 of 3"
          " planted solutions (raw, Jaccard >= 0.8) from the random start on average, without losing union-F1 vs unit"
          " (paired CI lower bound > -0.05)")
    for v, x in d.groupby("variant"):
        w = x.pivot_table(index="seed", columns="noise", values=M)
        for a in ("lasso", "temper-lasso"):
            if ("n_planted_found_raw", a) not in w or ("union_f1", "unit") not in w:
                continue
            fnd = w[("n_planted_found_raw", a)].mean()
            du = (w[("union_f1", a)] - w[("union_f1", "unit")]).dropna()
            lo = du.mean() - stats.t.ppf(0.975, len(du) - 1) * du.std(ddof=1) / np.sqrt(len(du))
            ok = fnd >= 1.0 and lo > -0.05
            print(f"  {v:22s} {a:13s} found {fnd:.2f}  union-F1 vs unit {du.mean():+.3f} (lower {lo:+.3f})"
                  f" -> {'FIXES the search' if ok else 'no fix'}")


if __name__ == "__main__":
    mode = sys.argv[4] if len(sys.argv) > 4 else "pilot"
    {"pilot": pilot, "stage1": stage1, "rq1": rq1, "bench": bench, "diag": diag, "temper": temper}[mode](sys.argv[1], sys.argv[2], sys.argv[3])
