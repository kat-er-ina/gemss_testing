"""Jaccard-penalty sweep (the lambda paragraph in the GEMSS section).

From jjsweep (GEMSSJjoint, p=200, D=10, overlaps {0,2,4,6,8}, lambda in
{0,0.3,...,1e6}, 15 seeds), reports per (lambda, overlap): mean union-F1, mean
recovered dissimilarity, and the gap |dissim - GT| to the true dissimilarity
d* = 1 - ov/(2D-ov). Reproduces the paper's claim that the penalty trades fidelity:
at high overlap (ov8, d*=0.33) raising lambda drives dissimilarity away from the
truth (gap 0.27 -> 0.62) with union-F1 collapsing (0.88 -> 0.37); only when disjoint
(ov0) does it improve fidelity (gap 0.19 -> 0.03).

Usage: .venv/bin/python scripts/make_jaccard.py [--results-dir results/paper]
"""
import argparse
import csv
import glob
import os
import re
import statistics as st
from collections import defaultdict

D = 10


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/paper")
    args = ap.parse_args()
    rows = []
    for f in glob.glob(os.path.join(args.results_dir, "jjsweep", "*.csv")):
        rows += list(csv.DictReader(open(f)))

    # (overlap, lambda) -> {union, dissim}
    agg = defaultdict(lambda: {"u": [], "d": []})
    for r in rows:
        m = re.search(r"'lj':\s*([0-9.eE+-]+)", r.get("hp", ""))
        if not m:
            continue
        lj = float(m.group(1))
        ov = int(r["overlap"])
        u, d = fnum(r.get("union_f1")), fnum(r.get("dissim"))
        if u is not None:
            agg[(ov, lj)]["u"].append(u)
        if d is not None:
            agg[(ov, lj)]["d"].append(d)

    ovs = sorted({k[0] for k in agg})
    ljs = sorted({k[1] for k in agg})
    gt = {ov: 1 - ov / (2 * D - ov) for ov in ovs}
    print("per (overlap, lambda): union-F1 / dissim / gap=|dissim-d*|   (d* per overlap shown)")
    for ov in ovs:
        print(f"\noverlap {ov}  (d*={gt[ov]:.2f}):")
        print("   lambda        uF1   dissim   gap")
        for lj in ljs:
            a = agg[(ov, lj)]
            if not a["u"]:
                continue
            u = st.mean(a["u"]); d = st.mean(a["d"]) if a["d"] else float("nan")
            print(f"   {lj:>9.4g}   {u:.2f}    {d:.2f}    {abs(d-gt[ov]):.2f}")

    # ---- verify key points vs paper ----
    def gap(ov, lj):
        a = agg[(ov, lj)]; return abs(st.mean(a["d"]) - gt[ov])
    def uf(ov, lj):
        return st.mean(agg[(ov, lj)]["u"])
    checks = [
        ("ov8 gap @lambda=0  ~0.27", abs(gap(8, 0.0) - 0.27) <= 0.03),
        ("ov8 gap @lambda=1e6 ~0.62", abs(gap(8, 1e6) - 0.62) <= 0.03),
        ("ov8 uF1 @0 ~0.88",  abs(uf(8, 0.0) - 0.88) <= 0.03),
        ("ov8 uF1 @1e6 ~0.37", abs(uf(8, 1e6) - 0.37) <= 0.03),
        ("ov0 gap @0 ~0.19",  abs(gap(0, 0.0) - 0.19) <= 0.04),
        ("ov0 gap @1e6 ~0.03", abs(gap(0, 1e6) - 0.03) <= 0.04),
    ]
    print("\n=== check vs paper ===")
    for name, passed in checks:
        print(f"  [{'OK' if passed else 'XX'}] {name}")
    print("\nALL MATCH PAPER" if all(p for _, p in checks) else "\nsome differ")


if __name__ == "__main__":
    main()
