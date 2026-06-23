"""RQ5 dimensionality: mean rank over the 18 (n,p,overlap) cells (tab:RQ5_lown).

Within each cell (n in {30,50,100} x p in {1000,2000,5000} x overlap in {2,6}) the three
methods are ranked by best-hp seed-mean union-F1 (rank 1 = best). We report each
method's mean rank and number of rank-1 cells. Paper: GEMSS 1.17 (15/18), ALFESE 2.06
(3/18), ensemble 2.78 (0/18).

GEMSS is the joint mechanism: p in {1000,5000} from mechsweep, p=2000 from jointrq6
(both method GEMSSjoint). ensemble (ENS) and ALFESE come from rq6grid.

Usage: .venv/bin/python scripts/make_rq5_meanrank.py [--results-dir results/paper]
"""
import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

NS, PS, OVS = [30, 50, 100], [1000, 2000, 5000], [2, 6]


def load(*dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.csv")):
            rows += list(csv.DictReader(open(f)))
    return rows


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def best_unionf1(rows, method, n, p, ov):
    by_hp = defaultdict(list)
    for r in rows:
        if (r.get("method") == method and int(r["n"]) == n and int(r["p"]) == p
                and int(r["overlap"]) == ov):
            u = fnum(r.get("union_f1"))
            if u is not None:
                by_hp[r["hp"]].append(u)
    if not by_hp:
        return None
    return max(st.mean(v) for v in by_hp.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/paper")
    args = ap.parse_args()
    R = args.results_dir
    gemss_rows = load(os.path.join(R, "mechsweep"), os.path.join(R, "jointrq6"))
    grid_rows = load(os.path.join(R, "rq6grid"))

    ranks = defaultdict(list)
    rank1 = defaultdict(int)
    cells = 0
    for n in NS:
        for p in PS:
            for ov in OVS:
                vals = {
                    "GEMSS": best_unionf1(gemss_rows, "GEMSSjoint", n, p, ov),
                    "ALFESE": best_unionf1(grid_rows, "ALFESE", n, p, ov),
                    "ensemble": best_unionf1(grid_rows, "ENS", n, p, ov),
                }
                if any(v is None for v in vals.values()):
                    print(f"  [skip cell n={n} p={p} ov={ov}: missing {[k for k,v in vals.items() if v is None]}]")
                    continue
                cells += 1
                order = sorted(vals, key=lambda k: vals[k], reverse=True)
                for rk, k in enumerate(order, 1):
                    ranks[k].append(rk)
                rank1[order[0]] += 1

    print(f"\nmean rank over {cells} cells:")
    print(f"{'method':10s} {'mean rank':>9s} {'rank-1':>8s}")
    for k in ["GEMSS", "ALFESE", "ensemble"]:
        print(f"{k:10s} {st.mean(ranks[k]):>9.2f} {rank1[k]:>5d}/{cells}")

    paper = {"GEMSS": (1.17, 15), "ALFESE": (2.06, 3), "ensemble": (2.78, 0)}
    ok = all(abs(st.mean(ranks[k]) - paper[k][0]) <= 0.02 and rank1[k] == paper[k][1] for k in paper)
    print("\nMATCHES PAPER" if ok and cells == 18 else "\nCHECK: differs from paper / cells != 18")


if __name__ == "__main__":
    main()
