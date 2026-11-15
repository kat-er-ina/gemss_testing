"""R8 internals at K = 12: coordinates outside the spike and signal share of the top-D, per setting.
Usage: r8_internals.py <worktree algorithm_comparison> <results/amvi dir>"""
import os, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from src.hard_data_factory import generate_overlapping_dataset
root = sys.argv[2]
truth = {}
for s in range(1, 26):
    _, _, t, _ = generate_overlapping_dataset(n_samples=100, n_features=200, n_solutions=3, sparsity=10,
        latent_rank=2, overlap=4, noise_std=0.05, nan_ratio=0.0, binarize=False, nonlinear=False,
        noise_dist="gauss", seed=s)
    truth[s] = set(t)
arms = {"sss": ("GEMSSjoint", 0.1, {"default": "r2/dumps/R4_GEMSSjoint_r2"}),
        "ss01": ("GEMSSjointSS", 0.01, {"default": "r2/dumps/R3_GEMSSjointSS_v0.01"})}
for prior, (meth, spike, extra) in arms.items():
    print(f"== {prior} (K = 12): setting | coords outside spike per comp (of which signal) | signal share top-D | top-2 share of top-D |mu|")
    for st in ["lik0.02", "lik0.05", "default", "lik0.3", "lik1.0", "yn0.3_def", "yn0.3_lik", "yn1.0_def", "yn1.0_lik"]:
        dd = os.path.join(root, extra.get(st, f"r4/dumps/R8_{prior}_{st}"))
        o, os_, sh, t2 = [], [], [], []
        for s in range(1, 26):
            f = os.path.join(dd, f"{meth}_r2_sd{s}_K12.npz")
            if not os.path.exists(f): continue
            z = np.load(f); mu, al = z["mu"], z["alpha"]
            for m, a in zip(mu, al):
                out = np.flatnonzero(np.abs(m) > 3 * spike)
                top = np.argsort(-np.abs(m))[:10]
                o.append(len(out)); os_.append(len(set(out) & truth[s]))
                sh.append(len(set(top) & truth[s]) / 10)
                v = np.sort(np.abs(m[top]))[::-1]; t2.append(v[:2].sum() / v.sum())
        print(f"  {st:10s} | {np.mean(o):6.1f} ({np.mean(os_):.1f}) | {np.mean(sh):.2f} | {np.mean(t2):.2f}")
