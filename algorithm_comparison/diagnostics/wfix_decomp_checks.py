"""F3 decomposition checks on the RQ1 trajectory dumps (plan sec. 16).

(1) Do the learned weights change the components?  F3 (ngvi) vs F3 frozen-w (ngvi_frozen) at the
    same seed / K / step: fraction of components with the same top-D set, max relative |mu| difference.
(2) Do components interact at all?  'Leak' = 1 - own responsibility of each component for its own
    draws under uniform weights, at 6k steps (0 = no overlap: the mixture entropy decouples and the
    components are independent fits at that point). frozen (Adam) vs ngvi_frozen (NGD).
Usage: wfix_decomp_checks.py <results/amvi dir>
"""
import itertools, os, sys
import numpy as np, pandas as pd
from scipy.special import logsumexp

root = sys.argv[1]
dd = lambda a, p: os.path.join(root, f"wfix/dumps/R5w_{a}_{p}")
rows = []
for p, s, K, step in itertools.product(("sss", "ss"), range(1, 26), (3, 6, 12, 24), (6000, 48000)):
    a = np.load(os.path.join(dd("ngvi", p), f"{p}_r2_sd{s}_K{K}.npz"))
    b = np.load(os.path.join(dd("ngvi_frozen", p), f"{p}_r2_sd{s}_K{K}.npz"))
    ma, mb = a[f"mu_{step}"], b[f"mu_{step}"]
    same = np.mean([set(np.argsort(-np.abs(x))[:10]) == set(np.argsort(-np.abs(y))[:10]) for x, y in zip(ma, mb)])
    rows.append((p, step, K, same, np.abs(ma - mb).max() / np.abs(ma).max()))
d = pd.DataFrame(rows, columns=["prior", "step", "K", "same_topD", "rel"])
print("== (1) F3 vs F3 frozen-w components")
print(d.groupby(["prior", "step", "K"]).agg(same_topD=("same_topD", "mean"), med_rel=("rel", "median"),
                                            max_rel=("rel", "max")).round(4).to_string())

print("\n== (2) responsibility leak at 6k, uniform weights")
rng = np.random.default_rng(0)
for p in ("sss", "ss"):
    for arm in ("frozen", "ngvi_frozen"):
        out = []
        for K in (6, 12, 24):
            leak = []
            for s in range(1, 26):
                z = np.load(os.path.join(dd(arm, p), f"{p}_r2_sd{s}_K{K}.npz")); mu, var = z["mu_6000"], z["var_6000"]
                for c in range(K):
                    x = mu[c] + rng.standard_normal((100, mu.shape[1])) * np.sqrt(var[c])
                    lp = -0.5 * (((x[:, None] - mu[None]) ** 2) / var[None] + np.log(2 * np.pi * var[None])).sum(-1)
                    leak.append(np.mean(1 - np.exp(lp[:, c] - logsumexp(lp, 1))))
            out.append(f"K{K}: mean {np.mean(leak):.4f}, comps > 1% {np.mean(np.array(leak) > 0.01):.2f}")
        print(f"  {p:3s} {arm:12s} " + "  ".join(out))
