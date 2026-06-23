"""Table 2 (overlap sweep): union-F1 and recovered dissimilarity per overlap.

Reproduces tab:rq2_fair. For each (method, overlap) it picks the hyperparameter with
the best seed-mean union-F1 (the "best configuration per method" the paper reports),
then emits union-F1 and recovered dissimilarity at that hp, plus the analytic
ground-truth dissimilarity d* = 1 - ov/(2D-ov).

GEMSS is the joint mechanism (results/paper/jointrq2, method GEMSSjoint); the baselines
come from results/paper/rq2stat. Overlaps 0/2/4/8 (ov6 omitted in the paper table).

Usage: .venv/bin/python scripts/make_table2.py [--results-dir results/paper]
"""
import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

D = 10
OVS = [0, 2, 4, 8]
# raw method key -> (source subdir, paper label)
METHODS = [
    ("GEMSSjoint", "jointrq2", "GEMSS"),
    ("ALFESE", "rq2stat", "ALFESE"),
    ("MASK", "rq2stat", "Masking"),
    ("BBSSL", "rq2stat", "BB-SSL"),
    ("ENS", "rq2stat", "RLE"),
    ("ENUM", "rq2stat", "EnumLASSO"),
]


def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, "*.csv")):
        rows += list(csv.DictReader(open(f)))
    return rows


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def best_hp_values(rows, method, overlap):
    """Pick hp with best seed-mean union-F1; return its (union_f1, dissim) seed-means."""
    by_hp = defaultdict(lambda: {"u": [], "d": []})
    for r in rows:
        if r["method"] != method or int(r["overlap"]) != overlap:
            continue
        u, d = fnum(r.get("union_f1")), fnum(r.get("dissim"))
        if u is not None:
            by_hp[r["hp"]]["u"].append(u)
            if d is not None:
                by_hp[r["hp"]]["d"].append(d)
    if not by_hp:
        return None, None
    best = max(by_hp, key=lambda h: st.mean(by_hp[h]["u"]))
    u = st.mean(by_hp[best]["u"])
    d = st.mean(by_hp[best]["d"]) if by_hp[best]["d"] else None
    return u, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/paper")
    args = ap.parse_args()
    cache = {sub: load(os.path.join(args.results_dir, sub)) for sub in {m[1] for m in METHODS}}

    print("=== union-F1 (best config per method) ===")
    hdr = "  ".join(f"ov{o}" for o in OVS)
    print(f"{'method':10s} | {hdr}")
    uf, di = {}, {}
    for key, sub, label in METHODS:
        uf[label] = [best_hp_values(cache[sub], key, o)[0] for o in OVS]
        di[label] = [best_hp_values(cache[sub], key, o)[1] for o in OVS]
        print(f"{label:10s} | " + "  ".join(f"{v:.2f}" if v is not None else " -- " for v in uf[label]))
    print("\n=== recovered dissimilarity (same config) ===")
    print(f"{'method':10s} | {hdr}")
    for _, _, label in METHODS:
        print(f"{label:10s} | " + "  ".join(f"{v:.2f}" if v is not None else " -- " for v in di[label]))
    print(f"{'GT (d*)':10s} | " + "  ".join(f"{1 - o/(2*D-o):.2f}" for o in OVS))

    # ---- verify against the paper ----
    paper_uf = {"GEMSS": [.84, .89, .91, .90], "ALFESE": [.75, .72, .71, .69],
                "Masking": [.69, .64, .62, .54], "BB-SSL": [.51, .53, .56, .61],
                "RLE": [.50, .51, .54, .53], "EnumLASSO": [.52, .51, .50, .48]}
    paper_di = {"GEMSS": [.83, .82, .80, .59], "ALFESE": [.89, .67, .46, .00],
                "Masking": [1.0, 1.0, 1.0, 1.0], "BB-SSL": [.98, 1.0, 1.0, 1.0],
                "RLE": [.90, .87, .85, .84], "EnumLASSO": [.43, .47, .47, .39]}
    print("\n=== check vs paper (max abs diff per method; tol .01) ===")
    ok = True
    for label in paper_uf:
        du = max(abs(round(a, 2) - b) for a, b in zip(uf[label], paper_uf[label]))
        dd = max(abs(round(a, 2) - b) for a, b in zip(di[label], paper_di[label]))
        flag = "" if du <= .01 and dd <= .01 else "  <-- MISMATCH"
        ok &= (du <= .01 and dd <= .01)
        print(f"  {label:10s} union-F1 Δ={du:.2f}  dissim Δ={dd:.2f}{flag}")
    print("\nALL MATCH PAPER" if ok else "\nMISMATCHES present")


if __name__ == "__main__":
    main()
