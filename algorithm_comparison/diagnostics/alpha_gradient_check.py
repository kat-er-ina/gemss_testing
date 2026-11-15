"""Does GEMSS's ELBO estimator give the mixture weights a data-driven gradient?
Loads a dumped GEMSS fit (mu, var, alpha), rebuilds the GEMSS mixture + prior + likelihood, and compares
  (i)  the gradient GEMSS actually uses: autograd of mean h(z) over z ~ mixture.sample(16) w.r.t. _log_alpha,
       averaged over many batches (component index drawn by torch.multinomial, not differentiable);
  (ii) the exact gradient of the ELBO w.r.t. the same logits: alpha_i (E_{q_i}[h] - sum_k alpha_k E_{q_k}[h]).
Usage: alpha_gradient_check.py <worktree algorithm_comparison> <results/amvi dir> <prior ss|sss> <dump subdir> <seed> <K>"""
import os, sys
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from src.hard_data_factory import generate_overlapping_dataset
from gemss.feature_selection.models import GaussianMixture, SpikeAndSlabPrior, StructuredSpikeAndSlabPrior
root, prior_name, sub, seed, K = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
X, y, _, _ = generate_overlapping_dataset(n_samples=100, n_features=200, n_solutions=3, sparsity=10, latent_rank=2,
    overlap=4, noise_std=0.05, nan_ratio=0.0, binarize=False, nonlinear=False, noise_dist="gauss", seed=seed)
X, y = torch.tensor(X), torch.tensor(y)
meth = "GEMSSjoint" if prior_name == "sss" else "GEMSSjointSS"
z = np.load(os.path.join(root, sub, f"{meth}_r2_sd{seed}_K{K}.npz"))
mix = GaussianMixture(K, 200)
with torch.no_grad():
    mix.mu.copy_(torch.tensor(z["mu"]))
    mix._log_var.copy_(torch.log(torch.expm1(torch.tensor(z["var"]))))
    a = torch.tensor(z["alpha"]); mix._log_alpha.copy_(torch.log(a[:-1] / a[-1]))
prior = (SpikeAndSlabPrior(100.0, 0.1, 10 / 200, 1 - 10 / 200) if prior_name == "ss"
         else StructuredSpikeAndSlabPrior(200, 10, var_slab=100.0, var_spike=0.1))
def h(zz): return prior.log_prob(zz) - 0.5 * ((zz @ X.T - y) ** 2).sum(-1) - mix.log_prob(zz)
# (i) GEMSS's estimator
N = 3000; g = np.zeros((N, K - 1))
for t in range(N):
    mix.zero_grad(); zz, _ = mix.sample(16); h(zz).mean().backward(); g[t] = mix._log_alpha.grad.numpy()
# (ii) exact: per-component expectations
with torch.no_grad():
    al = mix.get_alpha(); var = mix.get_variance(); E = torch.zeros(K)
    for k in range(K):
        zz = mix.mu[k] + torch.randn(4000, 200) * var[k].sqrt(); E[k] = h(zz).mean()
    ex = (al * (E - (al * E).sum()))[:-1].numpy()
print(f"{prior_name} {sub} seed {seed} K {K}: alpha*K = {np.round(al.numpy() * K, 2)}")
print(f"  per-component E_q_k[h] minus mixture mean (nats): {np.round((E - (al * E).sum()).numpy(), 1)}")
print(f"  GEMSS gradient, mean over {N} batches: {np.round(g.mean(0), 3)}  (SE {np.round(g.std(0) / np.sqrt(N), 3)}; per-batch sd {np.round(g.std(0), 2)})")
print(f"  exact ELBO gradient:                   {np.round(ex, 3)}")
