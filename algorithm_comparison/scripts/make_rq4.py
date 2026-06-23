"""RQ4 ensemble budget curve (tab:rq4).

Ensemble (RLE) union-F1 at the best alpha per (overlap, restart budget R), from
rq4stat, vs the GEMSS(joint) level (jointrq2). Overlaps {0,4,8}, R {50,300,3000,30000}.

Usage: .venv/bin/python scripts/make_rq4.py [--results-dir results/paper]
"""
import argparse, csv, glob, os, statistics as st
from collections import defaultdict

OVS = [0, 4, 8]
RS = [50, 300, 3000, 30000]


def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, "*.csv")):
        rows += list(csv.DictReader(open(f)))
    return rows


def fnum(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/paper")
    args = ap.parse_args()
    ens = load(os.path.join(args.results_dir, "rq4stat"))
    g = load(os.path.join(args.results_dir, "jointrq2"))

    def ens_best(ov, R):
        byA = defaultdict(list)
        for r in ens:
            if int(float(r["overlap"])) == ov and int(float(r["n_restarts"])) == R:
                u = fnum(r["union_f1"])
                if u is not None:
                    byA[r["alpha"]].append(u)
        return max((st.mean(v) for v in byA.values()), default=None)

    def gemss(ov):
        byhp = defaultdict(list)
        for r in g:
            if r.get("method") == "GEMSSjoint" and int(r["overlap"]) == ov:
                u = fnum(r["union_f1"])
                if u is not None:
                    byhp[r["hp"]].append(u)
        return max((st.mean(v) for v in byhp.values()), default=None)

    paper = {0: [.48, .51, .51, .51, .84], 4: [.49, .52, .55, .52, .91], 8: [.49, .56, .53, .55, .90]}
    print(f"{'overlap':9s} | " + "  ".join(f"R={R}" for R in RS) + " | GEMSS")
    ok = True
    for ov in OVS:
        row = [ens_best(ov, R) for R in RS] + [gemss(ov)]
        print(f"ov={ov:<6d} | " + "  ".join(f"{v:.2f}" if v is not None else "--" for v in row))
        ok &= all(abs(round(a, 2) - b) <= 0.02 for a, b in zip(row, paper[ov]))
    print("\nMATCHES PAPER" if ok else "\nsome cells differ from paper")


if __name__ == "__main__":
    main()
