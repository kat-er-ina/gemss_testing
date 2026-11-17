"""Plan sec. 19, ensemble-fairness check on A: do diverse starts (oracle = top exact patterns; patdiv = random
distinct supports) let independent fits (F3Dec, F3Ens) match the joint fit (F3J)?
Usage: analyze_ensemble_starts.py <dir with I_*.csv>
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(sys.argv[1], "I_*.csv"))], ignore_index=True)
print("rows:", len(d), d.groupby(["init", "arm"]).size().to_dict())


def short(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); m = x.mean()
    return f"{m:+.3f}{'*' if (m - h > 0 or m + h < 0) else ' '}"


idx = ["design", "rho", "n", "rep", "K"]
for col in ("tv_post", "tv_fit", "n_groups", "elbo"):
    print(f"\n== {col}: mean by init x arm, per (design, rho, n), pooled over K")
    print(d.groupby(["design", "rho", "n", "init", "arm"])[col].mean().unstack(["init", "arm"]).round(3).to_string())

print("\n== F3J (default start) minus independent arm under each start, tv_post (negative = joint better)")
pv = d.pivot_table(index=idx, columns=["init", "arm"], values="tv_post")
for init in ("default", "oracle", "patdiv"):
    for arm in ("F3Dec", "F3Ens"):
        g = (pv[("default", "F3J")] - pv[(init, arm)]).groupby(["design", "rho", "n"]).apply(short)
        print(f"   J(default) - {arm}({init}):", " ".join(f"{k[0]}/{k[1]}/{k[2]}:{v}" for k, v in g.items()))
print("\n== F3J under oracle start minus F3J default (does the joint fit also profit?)")
g = (pv[("oracle", "F3J")] - pv[("default", "F3J")]).groupby(["design", "rho", "n"]).apply(short)
print("  ", " ".join(f"{k[0]}/{k[1]}/{k[2]}:{v}" for k, v in g.items()))
print("\n== do oracle-started independent fits stay where they started? n_groups at the end vs K")
print(d[d.init == "oracle"].groupby(["arm", "K"]).n_groups.describe()[["mean", "min", "max"]].round(2).to_string())
