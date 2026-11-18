"""Plan sec. 39: SuSiE vs GEMSS joint fit (F3J) and ensemble (F3Ens), same data and seeds.

Usage:
  analyze_susie.py synth <susie dir> <gemss dir> <glob prefix E|B> <cells, e.g. p,n | noise,n> [prior sss] [step 6000]
  analyze_susie.py real <susie dir> <gemss real dir> [<gemss real invL dir>]
'*' = 95 % CI of the paired difference excludes 0; 'e' = CI inside +-0.02.
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)


def short(x, band=0.02):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return "n/a"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); m = x.mean()
    return f"{m:+.3f}{'e' if (m - h > -band and m + h < band) else ('*' if (m - h > 0 or m + h < 0) else ' ')}"


def load(d, g):
    fs = [f for f in glob.glob(os.path.join(d, g)) if os.path.getsize(f) > 0]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


if sys.argv[1] == "synth":
    sdir, gdir, pre, cells = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(",")
    prior = sys.argv[6] if len(sys.argv) > 6 else "sss"; step = int(sys.argv[7]) if len(sys.argv) > 7 else 6000
    s = load(sdir, f"{pre}_susie_*.csv")
    g = load(gdir, f"{pre}_*.csv"); g = g[(g.prior == prior) & (g.step == step) & (g.K == 12)]
    key = ["overlap"] + cells + ["seed"]
    d = pd.concat([s[key + ["arm", "union_f1", "sol_f1", "n_distinct"]], g[key + ["arm", "union_f1", "sol_f1", "n_distinct"]]])
    grp = ["overlap"] + cells
    print(f"SuSiE rows {len(s)}, GEMSS rows ({prior}, step {step}, K 12) {len(g)}")
    for m in ["union_f1", "sol_f1", "n_distinct"]:
        print(f"\n== {m}: mean per arm"); print(d.groupby(grp + ["arm"])[m].mean().unstack("arm").round(3).to_string())
        w = d.pivot_table(index=key, columns="arm", values=m)
        out = {}
        for a, b in [("SuSiE-top", "F3J"), ("SuSiE-top", "F3Ens"), ("SuSiE-samp", "F3J")]:
            if a in w and b in w:
                out[f"{a} - {b}"] = (w[a] - w[b]).groupby(grp).apply(short)
        print(f"-- {m}: paired differences"); print(pd.DataFrame(out).to_string())
    x = s[s.arm == "SuSiE-top"].copy()
    x["cs_mean_size"] = x.cs_sizes.fillna("").astype(str).map(lambda t: np.mean([int(v) for v in t.split(";") if v]) if t else 0)
    print("\n== SuSiE fit: credible sets and SuSiE-native union-F1 (mean over seeds)")
    print(x.groupby(grp)[["n_cs", "cs_mean_size", "cs_purity_min", "union_f1_csunion", "union_f1_pipD", "converged",
                          "susie_seconds"]].mean().round(3).to_string())
else:
    s = load(sys.argv[2], "D_*_susie.csv")
    M = ["nsol", "f1_best", "f1_mean", "f1_min", "dissim", "n_distinct_raw"]
    print("== SuSiE"); print(s.groupby(["dataset", "arm"])[M + ["n_cs"]].mean().round(3).to_string())
    print(s[s.arm == "SuSiE-top"][["dataset", "n_cs", "cs_sizes", "cs_purity_min", "cs_members"]].to_string())
    for lab, dd in [("fixed", sys.argv[3])] + ([("invL", sys.argv[4])] if len(sys.argv) > 4 else []):
        g = load(dd, "D_*.csv")
        print(f"\n== GEMSS ({lab})"); print(g.groupby(["dataset", "arm"])[M].mean().round(3).to_string())
