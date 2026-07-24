"""T1 - can we pick the hyperparameter WITHOUT the ground-truth oracle?

For each method x overlap we average metrics over seeds per hp, then compare the
union-F1 obtained by three selection rules:
  * ORACLE   : argmax mean union_f1               (uses ground truth; upper bound)
  * PREDICT  : argmax mean pred_mean (held-out)   (GT-free, all methods)
  * ELBO     : argmax mean elbo                    (GT-free, GEMSS only)
The GAP = union_f1(oracle) - union_f1(rule) is how much recovery we lose selecting
without the oracle. Small gap => the GT-free criterion is a viable selector.

Usage: .venv/bin/python rebuttal/analyze_t1.py
"""
import csv, glob, statistics as st
from collections import defaultdict

ROWS = [r for f in glob.glob("results/instr_rq2/pt_*.csv") for r in csv.DictReader(open(f))]
OVS = [0, 2, 4, 6, 8]


def fnum(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def hp_means(method, ov):
    """hp -> dict of seed-mean metrics for (method, overlap)."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in ROWS:
        if r["method"] != method or int(r["overlap"]) != ov:
            continue
        for k in ("union_f1", "pred_mean", "pred_best", "elbo"):
            v = fnum(r.get(k))
            if v is not None:
                acc[r["hp"]][k].append(v)
    return {hp: {k: st.mean(vs) for k, vs in d.items()} for hp, d in acc.items()}


def pick(hpm, by):
    cand = {hp: m for hp, m in hpm.items() if by in m}
    if not cand:
        return None, None
    hp = max(cand, key=lambda h: cand[h][by])
    return hp, hpm[hp].get("union_f1")


def main():
    methods = sorted({r["method"] for r in ROWS})
    print("union-F1 under each selection rule (mean over overlaps), and the oracle gap\n")
    print(f"{'method':11s} {'oracle':>7s} {'predict':>8s} {'gap_p':>6s} {'elbo':>6s} {'gap_e':>6s}")
    for m in methods:
        orc, prd, elb = [], [], []
        for ov in OVS:
            hpm = hp_means(m, ov)
            if not hpm:
                continue
            _, uo = pick(hpm, "union_f1")
            _, up = pick(hpm, "pred_mean")
            _, ue = pick(hpm, "elbo")
            if uo is not None:
                orc.append(uo)
            if up is not None:
                prd.append(up)
            if ue is not None:
                elb.append(ue)
        o = st.mean(orc) if orc else float("nan")
        p = st.mean(prd) if prd else float("nan")
        e = st.mean(elb) if elb else None
        line = f"{m:11s} {o:>7.3f} {p:>8.3f} {o-p:>6.3f}"
        line += f" {e:>6.3f} {o-e:>6.3f}" if e is not None else f" {'--':>6s} {'--':>6s}"
        print(line)
    print("\ngap_p = union-F1 lost by selecting hp on held-out PREDICTIVE instead of oracle")
    print("gap_e = same for ELBO selection (GEMSS only)")


if __name__ == "__main__":
    main()
