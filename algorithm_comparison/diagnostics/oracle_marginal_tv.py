"""Reference for R9: grouped-support TV of the product of the EXACT marginal PIPs (independent
Bernoullis), i.e. what perfect marginals alone achieve with an independence readout.
Usage (PYTHONPATH=<amvi>/src): oracle_marginal_tv.py <amvi root>"""
import json, os, sys
from itertools import product
import numpy as np
from amvi import RegressionModel
from amvi.data import make_one_triplet, make_two_triplets
from amvi.exact import exact_reference
root = sys.argv[1]
cfgs = [c for c in json.load(open(os.path.join(root, "results/datasets.json"))) if c["p"] == 10]
out = {}
for c in cfgs:
    two = c.get("groups") == 2
    if two:
        d = make_two_triplets(c); x, y, cols = d["x"], d["y"], sorted(int(i) for i in d["groups"].ravel())
    else:
        x, y, beta, test, block = make_one_triplet(c); cols = sorted(int(i) for i in block)
    if (not two) and c["rho"] not in (0.7, 0.9): continue
    ref = exact_reference(RegressionModel(x, y, c["sigma"], c["tau"], c["omega"]))
    g = len(cols); pats = np.array(list(product([0, 1], repeat=g)))
    pe = np.bincount(ref["masks"][:, cols] @ (2 ** np.arange(g)[::-1]), weights=ref["support"], minlength=2 ** g)
    pc = ref["pip"][cols]
    pind = np.prod(np.where(pats == 1, pc, 1 - pc), axis=1)
    key = ("two" if two else "one", c["rho"])
    out.setdefault(key, []).append((0.5 * np.abs(pind - pe).sum(), 1 - pe.max()))
for k in sorted(out):
    a = np.array(out[k])
    print(f"{k}: n={len(a)}  TV(exact marginals, independent) mean {a[:,0].mean():.3f}  | TV of best single pattern {a[:,1].mean():.3f}")
