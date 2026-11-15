"""R10 (plan sec. 11): choose the GEMSS spike width without ground truth, by ELBO.

Usage: analyze_r10.py <results/amvi dir>
Data: R3 (r2/R3), R4 (r2/R4, default spike 0.1), R6 (r3/R6).
Rules (fixed in the brief):
  (a) per seed, (spike, K) jointly by max ELBO
  (b) per seed and K, spike by max ELBO; then K by the paper rule (best seed-mean union-F1)
Reference: paper rule over (spike, K) (truth-chosen, = R6 table), and default spike 0.1.
"""
import glob, os, re, sys
import numpy as np, pandas as pd
from scipy import stats

root = sys.argv[1]
files = (glob.glob(os.path.join(root, "r2/R3/GEMSSjoint*.csv"))
         + glob.glob(os.path.join(root, "r2/R4/GEMSSjoint*_r[25]_s*.csv"))
         + glob.glob(os.path.join(root, "r3/R6/GEMSSjoint*.csv")))
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
d["spike"] = d["gemss_overrides"].fillna("").map(
    lambda s: float(re.search(r"var_spike=([0-9.e-]+)", s).group(1)) if "var_spike" in s else 0.1)
d["K"] = d["hp"].map(lambda s: int(re.search(r"\d+", s).group()))
d["prior"] = d["method"].map({"GEMSSjoint": "structured", "GEMSSjointSS": "standard"})
d = d.drop_duplicates(["prior", "latent_rank", "spike", "seed", "K"])


def ci(x):
    x = np.asarray(x, float); m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:.2f} ± {h:.2f}"


for (prior, r), g in d.groupby(["prior", "latent_rank"]):
    spikes = sorted(g.spike.unique())
    n_cells = g.groupby(["spike", "K"]).seed.nunique()
    seeds = sorted(set.intersection(*[set(s.seed) for _, s in g.groupby(["spike", "K"])]))
    g = g[g.seed.isin(seeds)]
    # reference: truth-chosen (spike, K) by seed-mean union-F1, and default spike 0.1
    cell = g.groupby(["spike", "K"]).union_f1.mean()
    sb, kb = cell.idxmax()
    ref = g[(g.spike == sb) & (g.K == kb)].union_f1
    d01 = g[g.spike == 0.1].groupby("K").union_f1.mean()
    kd = d01.idxmax(); dflt = g[(g.spike == 0.1) & (g.K == kd)].union_f1
    # (a) joint ELBO argmax per seed
    a = g.loc[g.groupby("seed").elbo.idxmax()]
    # (b) per seed and K spike by ELBO, then K by paper rule
    b = g.loc[g.groupby(["seed", "K"]).elbo.idxmax()]
    kb_b = b.groupby("K").union_f1.mean().idxmax()
    bk = b[b.K == kb_b]
    print(f"\n== {prior} prior, r = {r}: spikes {spikes}, {len(seeds)} seeds")
    print(f"  truth-chosen (paper rule): spike {sb}, K {kb}: {ci(ref)}")
    print(f"  default spike 0.1, K by truth ({kd}): {ci(dflt)}")
    print(f"  (a) spike+K by ELBO: {ci(a.union_f1)}; spikes chosen {a.spike.value_counts().sort_index().to_dict()}; "
          f"K chosen {a.K.value_counts().sort_index().to_dict()}")
    print(f"  (b) spike by ELBO per K, K by truth ({kb_b}): {ci(bk.union_f1)}; spikes chosen "
          f"{bk.spike.value_counts().sort_index().to_dict()}")
    # how ELBO ranks spikes on average (per seed, K): mean ELBO per spike
    print("  mean ELBO by spike:", g.groupby("spike").elbo.mean().round(1).to_dict())
    print("  mean union-F1 by spike (K by truth per spike):",
          {s: round(gs.groupby("K").union_f1.mean().max(), 2) for s, gs in g.groupby("spike")})
