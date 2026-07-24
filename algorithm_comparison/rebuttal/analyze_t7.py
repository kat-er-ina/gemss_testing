"""T7 - does GEMSS's advantage survive misspecification?

best-hp union-F1 (seed-mean) per scenario x method x overlap. Scenario is read from
the (nonlinear, noise_dist) columns: nonlin (nonlinear=1), tnoise (noise_dist=t),
laplace (noise_dist=laplace). Compares GEMSS(joint) / ensemble / ALFESE.

Usage: .venv/bin/python rebuttal/analyze_t7.py
"""
import csv, glob, statistics as st
from collections import defaultdict

ROWS = [r for f in glob.glob("results/t7_robust/pt_*.csv") for r in csv.DictReader(open(f))]
OVS = [0, 4, 8]
METHODS = ["GEMSSjoint", "ENS", "ALFESE"]


def fnum(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def scenario(r):
    if r.get("nonlinear") in ("1", "1.0"):
        return "nonlinear"
    return {"t": "t-noise", "laplace": "laplace"}.get(r.get("noise_dist"), None)


def best_uf(rows, method, ov):
    byhp = defaultdict(list)
    for r in rows:
        if r["method"] == method and int(r["overlap"]) == ov:
            u = fnum(r.get("union_f1"))
            if u is not None:
                byhp[r["hp"]].append(u)
    return max((st.mean(v) for v in byhp.values()), default=None)


def main():
    for scen in ["nonlinear", "t-noise", "laplace"]:
        rows = [r for r in ROWS if scenario(r) == scen]
        print(f"\n=== {scen} === union-F1 (best hp, seed-mean)")
        print(f"{'method':11s} " + "  ".join(f"ov{o:>2d}" for o in OVS))
        for m in METHODS:
            cells = [best_uf(rows, m, o) for o in OVS]
            print(f"{m:11s} " + "  ".join(f"{c:.3f}" if c is not None else " -- " for c in cells))
        # GEMSS margin over best baseline per overlap
        marg = []
        for o in OVS:
            g = best_uf(rows, "GEMSSjoint", o)
            b = max([x for x in (best_uf(rows, "ENS", o), best_uf(rows, "ALFESE", o)) if x is not None], default=None)
            marg.append(g - b if (g is not None and b is not None) else None)
        print("GEMSS-margin " + "  ".join(f"{c:+.3f}" if c is not None else " -- " for c in marg))


if __name__ == "__main__":
    main()
