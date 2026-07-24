"""Semi-synthetic RASHOMON benchmark for the GEMSS rebuttal (Track T5, refined).

WHY THIS EXISTS (vs rebuttal/semisynth.py)
------------------------------------------
`semisynth.py` plants K supports and builds y = sum_k X[:,support_k] @ beta_k.
That makes the K supports ADDITIVE COMPONENTS of a single mechanism, NOT
alternative/equivalent (Rashomon) solutions -- so per-support recovery is ~0 by
construction and it does not test GEMSS's headline claim (recovering multiple
GENUINELY-EQUIVALENT sparse supports). This script fixes the design.

DESIGN (genuine alternative solutions on REAL correlated X)
----------------------------------------------------------
1) On the real, z-scored X we find naturally COLLINEAR feature GROUPS by greedy
   single-linkage-to-seed clustering: seed a group with a feature, absorb every
   still-unused feature whose |corr| with the seed is >= `group_thresh`. A group
   is kept only if it has >= K members (so it holds >= K near-substitutable
   features). The seed is the group REPRESENTATIVE.

   Why single-linkage-to-the-seed is the RIGHT criterion here: the signal is
   generated from the representative only, so any member j with |corr(X_j,X_rep)|
   >= thresh has X_j ~ +/- X_rep and therefore reconstructs the same contribution
   after an OLS refit (the sign is absorbed by the coefficient). Members are thus
   genuine substitutes for the representative.

2) `sparsity` D = number of signal GROUPS. Signal from ONE representative per
   group:  y = sum_{g in signal_groups} beta_g * X[:, rep_g] + Gaussian noise.

3) The K planted ALTERNATIVE supports are K distinct "one-member-per-group"
   selections. `overlap` = number of SHARED groups: in a shared group all K
   supports use the representative (member 0); in a differing group support k
   uses member k (needs group size >= K). More shared groups => more overlap.
   Because every member of a group is ~substitutable, all K supports reconstruct
   y about equally well -> genuine Rashomon alternatives with controllable
   overlap.

GROUND TRUTH / SCORING
----------------------
* union-F1  : features anywhere-recovered vs the UNION of the K planted
              alternative supports (reuses semisynth.score).
* per-support recovery + matched-F1 : optimal Hungarian assignment of returned
              solutions to the K planted supports, recovery = fraction matched
              at Jaccard >= 0.5 (reuses semisynth.score).
* group-recovery : fraction of the D signal GROUPS covered by >= 1 returned
              feature (the natural "union" metric for the Rashomon design -- a
              method that finds a DIFFERENT valid member still gets credit).

VERIFICATION (printed once per dataset)
---------------------------------------
* mean within-group |corr| (are the groups actually collinear?)
* per-support OLS R^2 spread at overlap=0 (are the K planted supports really
  equivalent? small spread == yes).

METHODS: GEMSS (joint, K in Kgrid, clustered to m), randomised-Lasso ENSEMBLE,
ALFESE -- all via rebuttal.semisynth helpers, which import wrappers directly from
src.wrappers (NOT tuned_compare).

NOTE: y depends only on the representatives, NOT on `overlap`; overlap only
relabels the ground-truth alternative supports. So for a given seed the methods
are FIT ONCE and re-scored at every overlap level (large speedup, identical
solutions).

Usage (smoke):
  ../.venv/bin/python rebuttal/semisynth_rashomon.py -d diabetes \
      --overlaps 0 4 --seeds 0 1 --n_iter 300 --Kgrid 3 6

Usage (full):
  ../.venv/bin/python rebuttal/semisynth_rashomon.py --overlaps 0 3 6 \
      --seeds 0 1 2 3 4
"""

import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "diagnostics"))
sys.path.insert(0, _HERE)
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Reuse the loader, scoring, method-callers and GEMSS hyper-params verbatim.
from semisynth import (load, score, fit_gemss, fit_ensemble, fit_alfese,
                       GEMSS_HP, DATA_GLOB)


# --------------------------------------------------------------------------- #
# Collinear-group construction on the REAL X                                   #
# --------------------------------------------------------------------------- #
def find_collinear_groups(Xstd, min_size, thresh, rng):
    """Greedy single-linkage-to-seed grouping. Returns (groups, absC) where
    `groups` is a list of member-index lists, members[0] is the representative
    (seed), and every other member has |corr| with the seed >= thresh. Only
    groups with >= min_size members are kept. Groups are disjoint."""
    p = Xstd.shape[1]
    absC = np.abs(np.corrcoef(Xstd, rowvar=False))
    absC = np.nan_to_num(absC)
    np.fill_diagonal(absC, 0.0)
    used = np.zeros(p, dtype=bool)
    groups = []
    for seed in rng.permutation(p):
        if used[seed]:
            continue
        used[seed] = True
        # candidate substitutes: still-free features correlated with the seed
        cand = [j for j in range(p) if (not used[j]) and absC[seed, j] >= thresh]
        cand.sort(key=lambda j: -absC[seed, j])
        members = [int(seed)]
        for j in cand:
            if not used[j]:
                members.append(int(j))
                used[j] = True
        if len(members) >= min_size:
            groups.append(members)
        else:
            # release the private members (keep the seed consumed) so they may
            # still seed / join a larger group later; guarantees termination.
            for j in members[1:]:
                used[j] = False
    return groups, absC


def within_group_corr(groups, absC):
    """Mean |corr| of each non-rep member with its representative (member 0)."""
    vals = []
    for g in groups:
        if len(g) > 1:
            vals.extend(absC[g[0], g[1:]].tolist())
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Signal + planted alternative supports                                        #
# --------------------------------------------------------------------------- #
def simulate_y_from_reps(Xstd, signal_groups, rng, noise_frac=0.3):
    """y = sum_g beta_g * X[:, rep_g] + Gaussian noise. beta ~ U(0.5,2)*sign.
    Depends ONLY on the representatives (independent of overlap)."""
    reps = [g[0] for g in signal_groups]
    beta = rng.uniform(0.5, 2.0, size=len(reps)) * rng.choice([-1.0, 1.0], size=len(reps))
    signal = Xstd[:, reps] @ beta
    noise_std = noise_frac * (np.std(signal) + 1e-12)
    y = signal + rng.normal(0.0, noise_std, size=Xstd.shape[0])
    return y.astype(float)


def plant_alternatives(signal_groups, K, overlap, rng):
    """K distinct one-member-per-group supports. `overlap` groups are SHARED
    (all supports use member 0); the remaining groups DIFFER (support k uses
    member k, requires group size >= K). Returns (supports_dict, union_set,
    shared_group_indices)."""
    D = len(signal_groups)
    if not (0 <= overlap <= D):
        raise ValueError(f"overlap ({overlap}) must satisfy 0 <= overlap <= D ({D}).")
    order = rng.permutation(D)
    shared = set(int(i) for i in order[:overlap])
    supports = {}
    for k in range(K):
        supp = []
        for gi, g in enumerate(signal_groups):
            supp.append(g[0] if gi in shared else g[k])
        supports[f"solution_{k}"] = sorted(int(i) for i in supp)
    union = set()
    for s in supports.values():
        union |= set(s)
    return supports, union, shared


def group_recovery(sols, signal_groups):
    """Fraction of signal GROUPS covered by >= 1 returned feature (across all
    returned solutions)."""
    returned = set()
    for s in sols.values():
        returned |= set(s.get("support", []))
    hits = sum(1 for g in signal_groups if returned & set(g))
    return hits / len(signal_groups) if signal_groups else 0.0


def support_r2(Xstd, y, supp):
    lr = LinearRegression().fit(Xstd[:, supp], y)
    return float(lr.score(Xstd[:, supp], y))


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", default=None,
                    help="substring filter (e.g. diabetes / arabidopsis); default all non-pcos")
    ap.add_argument("-D", "--n_groups", type=int, default=8,
                    help="sparsity = number of signal GROUPS = planted support size")
    ap.add_argument("-K", "--Kplant", type=int, default=3, help="# planted alternative supports")
    ap.add_argument("-m", type=int, default=3, help="# solutions every method returns")
    ap.add_argument("--Kgrid", type=int, nargs="+", default=[3, 6, 12],
                    help="GEMSS component counts to sweep")
    ap.add_argument("--group_thresh", type=float, default=0.7,
                    help="|corr| threshold for group membership (near-substitutability)")
    ap.add_argument("--overlaps", type=int, nargs="+", default=[0, 3, 6],
                    help="number of SHARED groups per overlap level (0..D)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--noise_frac", type=float, default=0.3)
    ap.add_argument("--n_iter", type=int, default=None, help="override GEMSS n_iter (smoke)")
    ap.add_argument("--n_restarts", type=int, default=300, help="ensemble bootstrap restarts")
    ap.add_argument("--alfese_tau", type=float, default=0.5)
    ap.add_argument("--max_features", type=int, default=None,
                    help="subset X to the first N real columns (smoke)")
    ap.add_argument("--methods", nargs="+", default=["GEMSS", "ENS", "ALFESE"],
                    choices=["GEMSS", "ENS", "ALFESE"])
    ap.add_argument("--out", default=os.path.join("results", "semisynth_rashomon.csv"))
    args = ap.parse_args()

    hp = dict(GEMSS_HP)
    if args.n_iter is not None:
        hp["n_iter"] = args.n_iter
    D = args.n_groups

    paths = sorted(glob.glob(DATA_GLOB))
    paths = [p for p in paths if "pcos" not in os.path.basename(p).lower()]
    if args.dataset:
        paths = [p for p in paths if args.dataset.lower() in os.path.basename(p).lower()]
    if not paths:
        print(f"[!] no dataset matched under {DATA_GLOB}"); sys.exit(1)

    rows = []
    t_all = time.time()
    for path in paths:
        dname = os.path.basename(path).split("_")[0]
        feat, fnames = load(path)
        X = feat.values.astype(float)
        if args.max_features is not None:
            X = X[:, :args.max_features]
        Xstd = np.nan_to_num(StandardScaler().fit_transform(X))
        n, p = Xstd.shape
        print(f"\n=== {dname}: real X n={n} p={p} "
              f"(D={D} signal groups, K={args.Kplant} alternatives, m={args.m}, "
              f"thresh={args.group_thresh}) ===")

        # acc keyed by (overlap, method_label) -> metric lists across seeds
        acc = {}
        def push(ov, label, uf, rec, mf1, grp, sec):
            d = acc.setdefault((ov, label),
                               {"uf": [], "rec": [], "mf1": [], "grp": [], "sec": []})
            d["uf"].append(uf); d["rec"].append(rec); d["mf1"].append(mf1)
            d["grp"].append(grp); d["sec"].append(sec)

        # verification accumulators (per dataset)
        wg_corrs, r2_all, ngroups_seen, sizes_seen = [], [], [], []
        seeds_ok = 0

        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            groups, absC = find_collinear_groups(Xstd, args.Kplant, args.group_thresh, rng)
            big = [g for g in groups if len(g) >= args.Kplant]
            if len(big) < D:
                print(f"  seed={seed} SKIP: only {len(big)} groups of size>={args.Kplant} "
                      f"found (need D={D}); lower --group_thresh or --D.")
                continue
            seeds_ok += 1
            sel = rng.choice(len(big), size=D, replace=False)
            signal_groups = [big[i] for i in sel]
            ngroups_seen.append(len(big))
            sizes_seen.extend(len(g) for g in signal_groups)
            wg_corrs.append(within_group_corr(signal_groups, absC))

            y = simulate_y_from_reps(Xstd, signal_groups, rng, noise_frac=args.noise_frac)

            # equivalence check: R^2 of each planted alternative at overlap=0
            planted0, _, _ = plant_alternatives(signal_groups, args.Kplant, 0, rng)
            r2s = [support_r2(Xstd, y, s) for s in planted0.values()]
            r2_all.append(r2s)

            # ---- fit each method ONCE per seed (y is overlap-independent) ----
            method_sols = {}   # label -> (sols_dict, sec)
            if "GEMSS" in args.methods:
                for Kc in args.Kgrid:
                    t = time.time()
                    try:
                        sols = fit_gemss(Xstd, y, Kc, D, args.m, hp)
                        method_sols[f"GEMSS_K{Kc}"] = (sols, time.time() - t)
                    except Exception as e:
                        print(f"  GEMSS K={Kc} sd={seed} ERR {str(e)[:70]}")
            if "ENS" in args.methods:
                t = time.time()
                try:
                    sols = fit_ensemble(Xstd, y, D, args.m, args.n_restarts)
                    method_sols["ENS"] = (sols, time.time() - t)
                except Exception as e:
                    print(f"  ENS sd={seed} ERR {str(e)[:70]}")
            if "ALFESE" in args.methods:
                t = time.time()
                try:
                    sols = fit_alfese(Xstd, y, D, args.m, args.alfese_tau)
                    method_sols["ALFESE"] = (sols, time.time() - t)
                except Exception as e:
                    print(f"  ALFESE sd={seed} ERR {str(e)[:70]}")

            # ---- re-score at every overlap level ----
            for ov in args.overlaps:
                if ov > D:
                    continue
                planted, union, _ = plant_alternatives(signal_groups, args.Kplant, ov, rng)
                for label, (sols, sec) in method_sols.items():
                    uf, rec, mf1 = score(sols, planted, union, p)
                    grp = group_recovery(sols, signal_groups)
                    push(ov, label, uf, rec, mf1, grp, sec)

        # ---------------- verification report ---------------- #
        if seeds_ok == 0:
            print("  [!] no usable seed for this dataset; skipping.")
            continue
        r2_flat = np.array([r for seedr2 in r2_all for r in seedr2])
        print(f"  [verify] groups>=K found: mean {np.mean(ngroups_seen):.1f} "
              f"(selected D={D}); group size mean {np.mean(sizes_seen):.1f} "
              f"[{min(sizes_seen)}..{max(sizes_seen)}]")
        print(f"  [verify] mean within-group |corr| = {np.mean(wg_corrs):.3f} "
              f"(threshold was {args.group_thresh})")
        print(f"  [verify] per-support OLS R^2 (overlap=0): "
              f"mean {r2_flat.mean():.3f}  spread [{r2_flat.min():.3f}..{r2_flat.max():.3f}]  "
              f"std {r2_flat.std():.4f}  -> small spread == genuine alternatives")

        # ---------------- results tables ---------------- #
        for ov in args.overlaps:
            if ov > D:
                continue
            present = {lab: d for (o, lab), d in acc.items() if o == ov and d["uf"]}
            if not present:
                continue
            # GEMSS_best = best K by mean union-F1 at THIS overlap
            gk = {k: v for k, v in present.items() if k.startswith("GEMSS_K")}
            if gk:
                present["GEMSS_best"] = max(gk.items(),
                                            key=lambda kv: np.mean(kv[1]["uf"]))[1]
            print(f"\n  overlap={ov} shared / D={D} groups  ({seeds_ok} seeds)")
            print(f"    {'method':12s} {'union_F1':>9s} {'grp_recov':>9s} "
                  f"{'recovery':>9s} {'matched_F1':>11s} {'sec/fit':>8s}")
            order = [k for k in ("GEMSS_best",) if k in present] + \
                    sorted(k for k in present if k.startswith("GEMSS_K")) + \
                    [k for k in ("ENS", "ALFESE") if k in present]
            for label in order:
                d = present[label]
                muf, mrec, mmf1 = (float(np.mean(d["uf"])), float(np.mean(d["rec"])),
                                   float(np.mean(d["mf1"])))
                mgrp, msec = float(np.mean(d["grp"])), float(np.mean(d["sec"]))
                tag = "*" if label == "GEMSS_best" else " "
                print(f"  {tag} {label:12s} {muf:9.3f} {mgrp:9.3f} {mrec:9.3f} "
                      f"{mmf1:11.3f} {msec:8.1f}")
                rows.append(dict(dataset=dname, n=n, p=p, overlap=ov, D=D,
                                 Kplant=args.Kplant, m=args.m, group_thresh=args.group_thresh,
                                 method=label, n_seeds=len(d["uf"]),
                                 within_group_corr=float(np.mean(wg_corrs)),
                                 r2_mean=float(r2_flat.mean()), r2_std=float(r2_flat.std()),
                                 union_f1=muf, group_recovery=mgrp, recovery=mrec,
                                 matched_f1=mmf1, sec_per_fit=msec))

    if rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nSaved {args.out}  (total wall-clock {time.time() - t_all:.1f}s)")
    else:
        print("\n[!] no rows produced.")


if __name__ == "__main__":
    main()
