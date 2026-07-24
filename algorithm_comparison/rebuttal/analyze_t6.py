"""T6 - compute comparison from the instrumented instr_rq2 sweep.

Reports, per method: number of fits, wall-clock per fit (median + p10/p90 across
the hp grid and seeds), and for GEMSS the ELBO-plateau iteration. Memory is the
whole-process peak RSS (monotonic, dominated by imports) so it is reported once as
an envelope, not per-method.

Usage: .venv/bin/python rebuttal/analyze_t6.py
"""
import csv, glob, statistics as st
from collections import defaultdict

ROWS = [r for f in glob.glob("results/instr_rq2/pt_*.csv") for r in csv.DictReader(open(f))]


def fnum(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def pct(v, q):
    v = sorted(v); i = min(len(v) - 1, max(0, int(q * (len(v) - 1))))
    return v[i]


def main():
    t = defaultdict(list); nic = defaultdict(list); mem = []
    for r in ROWS:
        s = fnum(r.get("fit_seconds"))
        if s is not None:
            t[r["method"]].append(s)
        c = fnum(r.get("n_iter_converged"))
        if c is not None:
            nic[r["method"]].append(c)
        m = fnum(r.get("peak_mem_mb"))
        if m is not None:
            mem.append(m)

    print(f"{'method':11s} {'#fits':>6s} {'median s':>9s} {'p10':>7s} {'p90':>7s} {'conv.iter':>9s}")
    for m in sorted(t, key=lambda k: st.median(t[k])):
        row = f"{m:11s} {len(t[m]):>6d} {st.median(t[m]):>9.1f} {pct(t[m],.1):>7.1f} {pct(t[m],.9):>7.1f}"
        row += f" {st.median(nic[m]):>9.0f}" if nic[m] else f" {'--':>9s}"
        print(row)
    print(f"\nprocess peak RSS envelope: median {st.median(mem):.0f} MB "
          f"(monotonic whole-process, import-dominated; not a per-fit delta)")
    # headline contrast for the compute paragraph
    g = st.median(t.get("GEMSSjoint", [float('nan')]))
    e = st.median(t.get("ENS", [float('nan')]))
    print(f"\nheadline: one GEMSS fit (returns m solutions) ~= {g:.0f}s vs "
          f"ensemble at 3000 restarts ~= {e:.0f}s  ({e/g:.1f}x)")


if __name__ == "__main__":
    main()
