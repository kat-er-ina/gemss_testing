"""T4 - does GEMSS's high-overlap edge replicate under a DIFFERENT covariance?

best-hp union-F1 (seed-mean) per method x overlap, on the equicorrelation generator,
at two signal_frac values (0.6 harder, 0.85 near-equivalent supports).

Usage: .venv/bin/python rebuttal/analyze_t4.py
"""
import csv, glob, statistics as st
from collections import defaultdict

OVS = [0, 2, 4, 6, 8]
METHODS = ["GEMSSjoint", "ENS", "ALFESE"]


def fnum(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def best_uf(rows, method, ov):
    byhp = defaultdict(list)
    for r in rows:
        if r["method"] == method and int(r["overlap"]) == ov:
            u = fnum(r.get("union_f1"))
            if u is not None:
                byhp[r["hp"]].append(u)
    return max((st.mean(v) for v in byhp.values()), default=None)


def main():
    for sf, tag in [("sf60", "signal_frac=0.6 (harder, R^2~0.45 supports)"),
                    ("sf85", "signal_frac=0.85 (near-equivalent supports)")]:
        rows = [r for f in glob.glob(f"results/t4_equicorr/{sf}/pt_*.csv") for r in csv.DictReader(open(f))]
        print(f"\n=== equicorr {tag} === union-F1 (best hp, seed-mean)")
        print(f"{'method':11s} " + "  ".join(f"ov{o:>2d}" for o in OVS))
        for m in METHODS:
            cells = [best_uf(rows, m, o) for o in OVS]
            print(f"{m:11s} " + "  ".join(f"{c:.3f}" if c is not None else " -- " for c in cells))
        marg = []
        for o in OVS:
            g = best_uf(rows, "GEMSSjoint", o)
            b = max([x for x in (best_uf(rows, "ENS", o), best_uf(rows, "ALFESE", o)) if x is not None], default=None)
            marg.append(g - b if (g is not None and b is not None) else None)
        print("GEMSS-margin " + "  ".join(f"{c:+.3f}" if c is not None else " -- " for c in marg))


if __name__ == "__main__":
    main()
