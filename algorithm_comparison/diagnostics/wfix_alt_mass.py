"""Weight mass on each planted alternative in the RQ1 trajectory dumps (plan sec. 15).

For every fit, each component is assigned to the planted support whose PRIVATE features (not shared
by all solutions) its top-D set overlaps most (ties -> lower index; < half of them -> 'none'). Reported per arm, prior, K, step: mean over
seeds of the weight on the best-weighted / 2nd / 3rd planted support and on 'none', plus how
many planted supports carry >= 1/(2m) of the weight.
Usage: wfix_alt_mass.py <worktree algorithm_comparison> <results/amvi dir>
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
from src.hard_data_factory import generate_overlapping_dataset

root, D, m = sys.argv[2], 10, 3
sup = {}
for s in range(1, 26):
    *_, ss = generate_overlapping_dataset(n_samples=100, n_features=200, n_solutions=3, sparsity=10,
        latent_rank=2, overlap=4, noise_std=0.05, nan_ratio=0.0, binarize=False, nonlinear=False,
        noise_dist="gauss", seed=s)
    sets = [set(x) for x in ss.values()]
    shared = set.intersection(*sets)
    sup[s] = [x - shared for x in sets]          # private features of each planted solution
DIRS = {"current": "r3/dumps/R5_{p}", **{a: f"wfix/dumps/R5w_{a}_{{p}}" for a in ("frozen", "strat", "score", "ngvi", "fixedpoint", "ngvi_frozen")}}
rows = []
for arm, d in DIRS.items():
    for p in ("sss", "ss"):
        for K in (6, 12, 24):
            for s in range(1, 26):
                f = os.path.join(root, d.format(p=p), f"{p}_r2_sd{s}_K{K}.npz")
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                for step in (6000, 48000):
                    if f"mu_{step}" not in z:
                        continue
                    mu, al = z[f"mu_{step}"], z[f"alpha_{step}"]
                    w = np.zeros(m + 1)
                    for k in range(K):
                        top = set(np.argsort(np.abs(mu[k]))[::-1][:D].tolist())
                        ov = [len(top & t) for t in sup[s]]
                        j = int(np.argmax(ov)) if max(ov) >= len(sup[s][int(np.argmax(ov))]) / 2 else m
                        w[j] += al[k]
                    ws = np.sort(w[:m])[::-1]
                    rows.append(dict(arm=arm, prior=p, K=K, seed=s, step=step, w1=ws[0], w2=ws[1], w3=ws[2], none=w[m],
                                     n_alt=int((w[:m] >= 1 / (2 * m)).sum()), n_found=int((w[:m] > 0).sum())))
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
order = ["current", "frozen", "strat", "score", "ngvi", "fixedpoint", "ngvi_frozen"]
for p in ("sss", "ss"):
    for step in (6000, 48000):
        t = r[(r.prior == p) & (r.step == step)].groupby(["K", "arm"])[["w1", "w2", "w3", "none", "n_alt", "n_found"]].mean()
        t = t.reindex(pd.MultiIndex.from_product([[6, 12, 24], order], names=["K", "arm"]))
        print(f"\n== {p}, {step} steps: weight on best / 2nd / 3rd planted support, on none; "
              f"n_alt = #supports with weight >= 1/6; n_found = #supports hit by any component")
        print(t.round(2).to_string())
