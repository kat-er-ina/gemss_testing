"""Aggregate tuned_compare outputs into mean +/- 95% CI and paired win-rates.

Reads pt_*.csv (cols: method,n,p,overlap,seed,hp,sol_f1,union_f1,dissim,...).
For each (method, overlap) it selects the best hp by MEAN of the chosen metric
over seeds, then reports mean +/- 95% CI over seeds at that hp, and paired
per-seed win-rates for the decisive comparisons. CI = 1.96*sd/sqrt(N) (normal
approx; N>=15)."""
import argparse, csv, glob, math
from collections import defaultdict


def load(d):
    rows = []
    for f in glob.glob(f"{d}/*.csv"):
        for r in csv.DictReader(open(f)):
            rows.append(r)
    return rows


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def per_seed_at_best(rows, method, overlap, metric):
    """Return {seed: value} at the hp with the best seed-mean, for this method/overlap."""
    byhp = defaultdict(dict)  # hp -> {seed: val}
    for r in rows:
        if r["method"] == method and r["overlap"] == overlap:
            v = fnum(r.get(metric))
            if v is not None:
                byhp[r["hp"]][int(r["seed"])] = v
    if not byhp:
        return {}
    best = max(byhp, key=lambda h: sum(byhp[h].values()) / len(byhp[h]))
    return byhp[best]


def mean_ci(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = sum(vals) / n
    if n < 2:
        return mu, float("nan"), n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))
    return mu, 1.96 * sd / math.sqrt(n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--metric", default="union_f1")
    ap.add_argument("--overlaps", nargs="+", default=["0", "2", "4", "6", "8"])
    ap.add_argument("--methods", nargs="+",
                    default=["GEMSS", "ALFESE", "ENS", "BBSSL", "ENUM", "STAB", "MASK"])
    ap.add_argument("--vs", nargs="+", default=["ALFESE", "ENS"],
                    help="win-rate of GEMSS vs each of these")
    args = ap.parse_args()
    rows = load(args.dir)
    print(f"loaded {len(rows)} rows from {args.dir}; metric={args.metric}\n")
    print(f"{'method':8s} " + "  ".join(f"ov{o:>10s}" for o in args.overlaps))
    perseed = {}  # (method,ov)->{seed:val}
    for me in args.methods:
        cells = []
        for ov in args.overlaps:
            ps = per_seed_at_best(rows, me, ov, args.metric)
            perseed[(me, ov)] = ps
            mu, ci, n = mean_ci(list(ps.values()))
            cells.append(f"{mu:.2f}+-{ci:.2f}(n{n})" if mu == mu else "  --  ")
        print(f"{me:8s} " + "  ".join(f"{c:>12s}" for c in cells))
    # paired win-rates: GEMSS vs each --vs
    for opp in args.vs:
        print(f"\nGEMSS vs {opp}: paired per-seed win-rate ({args.metric}); +mean-diff [95% CI]")
        for ov in args.overlaps:
            g = perseed.get(("GEMSS", ov), {}); o = perseed.get((opp, ov), {})
            common = sorted(set(g) & set(o))
            if not common:
                print(f"  ov{ov}: --"); continue
            diffs = [g[s] - o[s] for s in common]
            wins = sum(1 for d in diffs if d > 0)
            mu, ci, n = mean_ci(diffs)
            sig = "sig" if (mu - ci > 0 or mu + ci < 0) else "n.s."
            print(f"  ov{ov}: GEMSS wins {wins}/{len(common)}  diff {mu:+.3f}+-{ci:.3f} [{sig}]")


if __name__ == "__main__":
    main()
