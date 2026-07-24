"""RQ6 robustness under feature noise and missingness (tab:RQ6).

Note: the "noise" sweep is Gaussian noise added to the design matrix X (features),
not to the labels y -- see src/hard_data_factory.py (X += noise). It is feature /
input noise, not label noise.

union-F1 (best hp) and recovered dissimilarity for GEMSS(joint) / ensemble / ALFESE,
across a noise sweep (nan=0, noise in {0.05,0.2,0.5,1.0} = clean/x4/x10/x20) and a
missing sweep (noise=0.05, nan in {0.1,0.25,0.5}). p=200, overlap 4, D=10.

GEMSS(joint) from jointrq7; ensemble (ENS) and ALFESE from rq7grid. ALFESE has no
missing-data path (shown as None).

Usage: .venv/bin/python scripts/make_rq6.py [--results-dir results/paper]
"""
import argparse, csv, glob, os, statistics as st
from collections import defaultdict

NOISE = [("clean", 0.05), ("x4", 0.2), ("x10", 0.5), ("x20", 1.0)]   # nan=0
MISS = [("10%", 0.1), ("25%", 0.25), ("50%", 0.5)]                   # noise=0.05


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


def cell(rows, method, noise, nan):
    """best-hp seed-mean union-F1 and its dissim, for (method, noise, nan)."""
    byhp = defaultdict(lambda: {"u": [], "d": []})
    for r in rows:
        if (r.get("method") == method and abs(float(r["noise"]) - noise) < 1e-9
                and abs(float(r["nan"]) - nan) < 1e-9):
            u, d = fnum(r.get("union_f1")), fnum(r.get("dissim"))
            if u is not None:
                byhp[r["hp"]]["u"].append(u)
                if d is not None:
                    byhp[r["hp"]]["d"].append(d)
    if not byhp:
        return None, None
    best = max(byhp, key=lambda h: st.mean(byhp[h]["u"]))
    d = st.mean(byhp[best]["d"]) if byhp[best]["d"] else None
    return st.mean(byhp[best]["u"]), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/paper")
    args = ap.parse_args()
    R = args.results_dir
    g = load(os.path.join(R, "jointrq7"))     # GEMSSjoint
    grid = load(os.path.join(R, "rq7grid"))   # ENS, ALFESE
    src = {"GEMSS": (g, "GEMSSjoint"), "ens": (grid, "ENS"), "ALF": (grid, "ALFESE")}

    def get(method, noise, nan):
        rows, key = src[method]
        return cell(rows, key, noise, nan)

    # paper targets: union-F1 then dissim
    pu = {"GEMSS": [.89, .83, .77, .69, .78, .76, .73], "ens": [.53, .55, .56, .55, .56, .62, .59],
          "ALF": [.72, .73, .65, .54, None, None, None]}
    pd = {"GEMSS": [.79, .66, .55, .45, .59, .46, .37], "ens": [.84, .89, .85, .84, .88, .82, .84],
          "ALF": [.46, .46, .46, .46, None, None, None]}
    levels = NOISE + MISS

    print(f"{'level':8s} | {'GEMSS uF1/dis':>14s} {'ens uF1/dis':>14s} {'ALF uF1/dis':>14s}")
    got_u = {m: [] for m in src}; got_d = {m: [] for m in src}
    for lbl, val in levels:
        nan = val if (lbl, val) in MISS else 0.0
        noise = 0.05 if (lbl, val) in MISS else val
        line = f"{lbl:8s} |"
        for m in ["GEMSS", "ens", "ALF"]:
            u, d = get(m, noise, nan)
            got_u[m].append(u); got_d[m].append(d)
            line += f"  {('--' if u is None else f'{u:.2f}')}/{('--' if d is None else f'{d:.2f}')}".rjust(15)
        print(line)

    def close(a, b):
        if b is None:
            return a is None
        return a is not None and abs(round(a, 2) - b) <= 0.02
    ok = all(close(got_u[m][i], pu[m][i]) and close(got_d[m][i], pd[m][i])
             for m in src for i in range(len(levels)))
    print("\nMATCHES PAPER" if ok else "\nsome cells differ from paper")


if __name__ == "__main__":
    main()
