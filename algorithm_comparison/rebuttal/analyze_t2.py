"""T2 - a recovery metric that verifies INDIVIDUAL supports (reviewer #3/#1).

union-F1 only asks whether planted features appear *somewhere* among the returned
solutions; it never checks that a returned support matches a planted one. Here we
build the missing metric: optimal (Hungarian) assignment between the m returned
supports and the 3 planted supports, then report
  * recovery = fraction of planted supports matched at Jaccard >= 0.5
  * matchF1  = mean per-support F1 after optimal assignment
  * union_f1 = the existing metric, for contrast.
The question: does GEMSS's union-F1 lead survive the stricter, per-support metric?

Selection is the ground-truth oracle (best hp by union_f1 per overlap), i.e. we ask
whether the union-F1 winner is also the support-recovery winner.

Usage: .venv/bin/python rebuttal/analyze_t2.py
"""
import csv, glob, sys, statistics as st
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, ".")
from src.hard_data_factory import generate_overlapping_dataset

ROWS = [r for f in glob.glob("results/instr_rq2/pt_*.csv") for r in csv.DictReader(open(f))]
OVS = [0, 2, 4, 6, 8]
TAU = 0.5
_PLANT = {}


def fnum(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def parse_supports(s):
    return [set(int(x) for x in grp.split(",") if x != "") for grp in s.split(";") if grp]


def planted(ov, seed):
    key = (ov, seed)
    if key not in _PLANT:
        _, _, _, pl = generate_overlapping_dataset(
            n_samples=100, n_features=200, n_solutions=3, sparsity=10, latent_rank=2,
            overlap=ov, noise_std=0.05, nan_ratio=0.0, binarize=False, seed=seed)
        _PLANT[key] = [set(v) for v in pl.values()]
    return _PLANT[key]


def jacc(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def f1(a, b):
    return 2 * len(a & b) / (len(a) + len(b)) if (len(a) + len(b)) else 0.0


def match_metrics(returned, plant):
    P = len(plant)
    if not returned:
        return 0.0, 0.0
    C = np.zeros((len(returned), P))
    for i, a in enumerate(returned):
        for j, b in enumerate(plant):
            C[i, j] = jacc(a, b)
    ri, ci = linear_sum_assignment(-C)
    pairs = [(returned[i], plant[j]) for i, j in zip(ri, ci)]
    recovery = sum(1 for a, b in pairs if jacc(a, b) >= TAU) / P
    matchf1 = st.mean([f1(a, b) for a, b in pairs])
    return recovery, matchf1


def best_hp(method, ov):
    acc = defaultdict(list)
    for r in ROWS:
        if r["method"] == method and int(r["overlap"]) == ov:
            u = fnum(r.get("union_f1"))
            if u is not None:
                acc[r["hp"]].append(u)
    return max(acc, key=lambda h: st.mean(acc[h])) if acc else None


def main():
    methods = sorted({r["method"] for r in ROWS})
    print(f"per-support recovery at oracle-best hp (matched Jaccard>= {TAU}), mean over overlaps\n")
    print(f"{'method':11s} {'union_f1':>8s} {'recovery':>8s} {'matchF1':>8s}")
    summary = {}
    for m in methods:
        U, R, F = [], [], []
        for ov in OVS:
            hp = best_hp(m, ov)
            if hp is None:
                continue
            for r in ROWS:
                if r["method"] != m or int(r["overlap"]) != ov or r["hp"] != hp:
                    continue
                sup = parse_supports(r.get("supports", ""))
                u = fnum(r.get("union_f1"))
                if u is None:
                    continue
                rec, mf1 = match_metrics(sup, planted(ov, int(r["seed"])))
                U.append(u); R.append(rec); F.append(mf1)
        if U:
            summary[m] = (st.mean(U), st.mean(R), st.mean(F))
            print(f"{m:11s} {summary[m][0]:>8.3f} {summary[m][1]:>8.3f} {summary[m][2]:>8.3f}")

    # does the union-F1 ranking match the recovery ranking?
    by_u = sorted(summary, key=lambda k: -summary[k][0])
    by_r = sorted(summary, key=lambda k: -summary[k][1])
    print(f"\nunion_f1 ranking: {' > '.join(by_u)}")
    print(f"recovery ranking: {' > '.join(by_r)}")
    print(f"GEMSS wins union_f1: {by_u[0]=='GEMSSjoint'};  GEMSS wins recovery: {by_r[0]=='GEMSSjoint'}")


if __name__ == "__main__":
    main()
