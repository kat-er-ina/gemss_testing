"""Validate the repulsion principle: SVGD covers the modes of a 2-D mixture.

Bug-free discipline for mechanism B (kernel repulsion). We reproduce the published
Stein Variational Gradient Descent behaviour (Liu & Wang 2016): with the kernel
repulsion term, particles spread to cover ALL modes of a multimodal target; with
the repulsion removed (plain gradient ascent), they collapse onto one mode. This
confirms our kernel/repulsion implementation is correct before it is used inside
GEMSS (where `_kernel_repulsion` is the potential-based analogue of this force).

Target: equal-weight mixture of 3 Gaussians at (-3,-3), (3,3), (0,4), unit cov.
Pass: SVGD covers all 3 modes; no-repulsion control covers fewer.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CENTERS = torch.tensor([[-3.0, -3.0], [3.0, 3.0], [0.0, 4.0]])


def grad_logp(X):
    """Gradient of log p(x) for the 3-Gaussian mixture (autograd)."""
    Xr = X.detach().clone().requires_grad_(True)
    # log p = logsumexp_k -0.5||x-c_k||^2  (+const)
    d2 = ((Xr.unsqueeze(1) - CENTERS.unsqueeze(0)) ** 2).sum(-1)  # [n,K]
    logp = torch.logsumexp(-0.5 * d2, dim=1).sum()
    (g,) = torch.autograd.grad(logp, Xr)
    return g


def _bandwidth(X):
    pd2 = torch.cdist(X, X) ** 2
    med = pd2[pd2 > 0].median() if (pd2 > 0).any() else torch.tensor(1.0)
    return (med / np.log(X.shape[0] + 1)).clamp(min=1e-3)


def svgd(X, steps=600, eps=0.3, mode="svgd"):
    """mode='svgd' = driving + kernel repulsion; mode='plain' = independent
    gradient ascent (no kernel) -- the collapse control."""
    X = X.clone()
    for _ in range(steps):
        g = grad_logp(X)              # [n,d]
        if mode == "plain":
            X = X + eps * g           # each particle climbs independently
            continue
        pd2 = torch.cdist(X, X) ** 2
        h = _bandwidth(X)
        K = torch.exp(-pd2 / h)       # [n,n]
        driving = K @ g
        grad_k = (2.0 / h) * (K.sum(1, keepdim=True) * X - K @ X)
        phi = (driving + grad_k) / X.shape[0]
        X = X + eps * phi
    return X


def modes_covered(X, radius=2.0):
    d = torch.cdist(X, CENTERS)            # [n,K]
    nearest = d.argmin(1)
    covered = sum(int(((nearest == k) & (d[:, k] < radius)).any()) for k in range(len(CENTERS)))
    per_mode = [int(((nearest == k) & (d[:, k] < radius)).sum()) for k in range(len(CENTERS))]
    return covered, per_mode


def main():
    torch.manual_seed(0)
    X0 = torch.randn(60, 2) * 0.1        # tightly clustered at the origin
    Xs = svgd(X0, mode="svgd")
    Xn = svgd(X0, mode="plain")
    cs, ps = modes_covered(Xs)
    cn, pn = modes_covered(Xn)
    print("SVGD on 3-mode 2-D mixture; 60 particles started tightly at the origin.")
    print(f"  SVGD (driving+repulsion): modes_covered={cs}/3  per_mode={ps}")
    print(f"  plain gradient ascent:    modes_covered={cn}/3  per_mode={pn}")
    # SVGD should cover all modes ~evenly (equal-weight target); plain GA collapses.
    even = all(8 <= c <= 32 for c in ps)
    ok = (cs == 3) and even and (cn < 3)
    print("PASS" if ok else "FAIL (SVGD should cover all modes ~evenly; plain GA should collapse)")


if __name__ == "__main__":
    main()
