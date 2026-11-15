"""Why does longer training hurt RQ1 recovery? Compare GEMSS parameter dumps at 6k (R4, r=2) and
24k steps (R2), same seeds/K/torch seed. Per component: data fit, coordinates outside the spike,
signal share of the big coordinates and of the top-D fill, mass on noise features; and the
relation between a component's weight alpha_k and the noise in its top-D.
Usage: long_training_internals.py <worktree algorithm_comparison> <results/amvi dir>"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
from src.hard_data_factory import generate_overlapping_dataset
root = sys.argv[2]; D = 10
data = {}
for s in range(1, 26):
    X, y, t, _ = generate_overlapping_dataset(n_samples=100, n_features=200, n_solutions=3, sparsity=D,
        latent_rank=2, overlap=4, noise_std=0.05, nan_ratio=0.0, binarize=False, nonlinear=False,
        noise_dist="gauss", seed=s)
    data[s] = (X, y, np.array(sorted(t)))
arms = {"sss": ("GEMSSjoint", "r2/dumps/R4_GEMSSjoint_r2", "r2/dumps/R2_GEMSSjoint"),
        "ss": ("GEMSSjointSS", "r2/dumps/R4_GEMSSjointSS_r2", "r2/dumps/R2_GEMSSjointSS")}
rows = []
for prior, (meth, d6, d24) in arms.items():
    for steps, dd in ((6000, d6), (24000, d24)):
        for s in range(1, 26):
            X, y, tr = data[s]; sig = np.zeros(200, bool); sig[tr] = True
            for K in (3, 6, 12, 18, 24):
                f = os.path.join(root, dd, f"{meth}_r2_sd{s}_K{K}.npz")
                if not os.path.exists(f): continue
                z = np.load(f); mu, var, al = z["mu"], z["var"], z["alpha"]
                for k in range(K):
                    m = mu[k]; big = np.abs(m) > 0.3
                    top = np.argsort(-np.abs(m))[:D]; fill = [j for j in top if not big[j]]
                    r = y - X @ m
                    rows.append(dict(prior=prior, steps=steps, seed=s, K=K, k=k, alpha_rel=al[k] * K,
                        rss=float(r @ r), n_big=int(big.sum()), big_sig=float(sig[big].mean()) if big.any() else np.nan,
                        n_fill=len(fill), fill_sig=float(sig[fill].mean()) if fill else np.nan,
                        topD_sig=float(sig[top].mean()),
                        l1_noise=float(np.abs(m[~sig]).sum()), l1_sigsmall=float(np.abs(m[sig & ~big]).sum()),
                        max_noise=float(np.abs(m[~sig]).max()),
                        fill_min=float(np.abs(m[top]).min()), sd_q=float(np.sqrt(var[k]).mean())))
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
cols = ["rss", "n_big", "big_sig", "n_fill", "fill_sig", "topD_sig", "l1_noise", "l1_sigsmall", "max_noise", "fill_min", "sd_q"]
print("== per-component means by prior, K, steps")
print(d.groupby(["prior", "K", "steps"])[cols].mean().round(3).to_string())
print("\n== 24k: does a component's weight predict noise in its top-D? (Spearman alpha_rel vs topD_sig, per prior/K)")
for (pr, K), g in d[d.steps == 24000].groupby(["prior", "K"]):
    lo = g[g.alpha_rel < 0.5]; hi = g[g.alpha_rel >= 1.0]
    print(f"  {pr} K={K}: rho={g[['alpha_rel','topD_sig']].corr('spearman').iloc[0,1]:+.2f}; topD_sig alpha<0.5K^-1: "
          f"{lo.topD_sig.mean():.2f} (n={len(lo)}), alpha>=K^-1: {hi.topD_sig.mean():.2f} (n={len(hi)})")
print("\n== alpha spread (sd of log alpha) by prior, K, steps")
d["la"] = np.log(d.alpha_rel)
print(d.groupby(["prior", "K", "steps"]).la.std().unstack().round(2).to_string())
