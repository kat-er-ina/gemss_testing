"""Plan sec. 23 (experiment G-amvi): does the component family of Li, Golub & Lu (2026, arXiv:2609.38656)
-- Bernoulli x Gaussian components with exact zeros, i.e. a per-coordinate inclusion probability inside every
component -- keep its mixture gain on a multimodal block as orthogonal nuisance dimensions are added, where
GEMSS's Gaussian components lose it (plan sec. 22)?

Same datasets as experiment G (overlap_map.make_data + embed_diag, same keys): an exact block (p = 10, A designs)
embedded in P dimensions with mutually orthogonal nuisance; prior weight fixed (pure dimension). Their model:
point-mass spike, slab N(0, tau^2), inclusion omega, KNOWN noise s.d. sigma = 1 (= GEMSS's unit-noise
likelihood), tau = G's slab s.d., omega = G's w. Their posterior also factorises over orthogonal nuisance, so
the exact block posterior is the block-only enumeration (amvi.exact.exact_posterior) and the nuisance PIPs are
closed form (both checked in --selftest).

Their package amvi (commit b8341d8, no license -> run only, never vendored): PYTHONPATH=<repo>/src.
Fit as in their scripts/run_dataset.py: strong_meanfield baseline, then fit(model, Settings(max_components=10,
max_seconds=...), seed, baseline). Their budget is 120 s at p > 10; default here 600 s (status is recorded).

Per (dataset, P), for their mixture (mix) and their mean-field baseline (mf):
  tv_*            total variation between sum_k w_k prod_j Bern(a_kj) over block patterns and the exact one
  pip_mae_*       block PIP error;  nuis_pip_mae_*: nuisance PIP error vs closed form
Arms (same mean-field baseline): asis = their fit unchanged; oracle = their fit unchanged except that the
covariance-split proposal (amvi.solver._split, which builds its split direction from the 20 coordinates with the
largest inclusion probability) is told the block: nuisance inclusion probabilities are masked while the split
picks its coordinates and restored in the proposal. Their code is wrapped at run time, not copied. asis tests
their procedure; oracle isolates their component family from their proposal heuristic.
Split diagnostics: split_block_sel = block coordinates among the 20 the unmodified split would use (from the
mean-field start), max_nuis_a = largest nuisance inclusion probability there.
Mixture structure: K (returned), K_cand (before the final check), status, fallback, gain over mean field
(nats, positive = better), n_groups_block / n_groups_all (distinct rounded inclusion patterns among components,
block / all coordinates), a_diff_block / a_diff_nuis (weight-paired |a_kj - a_lj| summed over block / nuisance).
Usage: amvi_dimdist.py --design M --rho 0.9 --n 25 --P 0 30 100 300 1000 --reps 20 --rep0 2 --out x.csv
       amvi_dimdist.py --selftest
"""
import argparse, csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlap_map import make_data, embed_diag, exact_nuisance_pip

import amvi.solver as AS
from amvi import Mixture, RegressionModel, Settings, fit
from amvi.initialization import strong_meanfield
from amvi.exact import exact_posterior, Model


def pattern_probs(weights, a, masks):
    a = np.clip(np.asarray(a, float), 1e-12, 1 - 1e-12)
    return np.exp(masks @ np.log(a).T + (1 - masks) @ np.log1p(-a).T) @ np.asarray(weights, float)


def score(q, g, ex, npip):
    w, a = np.asarray(q.weights, float), np.asarray(q.a, float)
    pip = w @ a
    out = dict(tv=0.5 * np.abs(pattern_probs(w, a[:, :g], ex["masks"]) - ex["support"]).sum(),
               pip_mae=float(np.abs(pip[:g] - ex["pip"]).mean()))
    out["nuis_pip"] = float(pip[g:].mean()) if npip is not None else np.nan
    out["nuis_pip_mae"] = float(np.abs(pip[g:] - npip).mean()) if npip is not None else np.nan
    return out


def structure(q, g):
    w, a = np.asarray(q.weights, float), np.asarray(q.a, float)
    r = (a > 0.5).astype(int)
    db = dn = 0.0
    for k in range(len(w)):
        for l in range(k + 1, len(w)):
            db += w[k] * w[l] * np.abs(a[k, :g] - a[l, :g]).sum()
            dn += w[k] * w[l] * np.abs(a[k, g:] - a[l, g:]).sum()
    return dict(n_groups_block=len({tuple(x) for x in r[:, :g]}), n_groups_all=len({tuple(x) for x in r}),
                a_diff_block=db, a_diff_nuis=dn)


_SPLIT = AS._split


def block_split(g):
    """their covariance split, with the coordinate choice restricted to the block (oracle)"""
    def split(model, current, index):
        k = int(np.argmax(current.weights))
        am = current.a.copy(); am[:, g:] = 1e-9                  # rank block coordinates first
        prop, det = _SPLIT(model, Mixture(current.weights, am, current.mu, current.var), index)
        a = np.vstack([current.a, current.a[k]])                  # restore nuisance inclusion probabilities
        return Mixture(prop.weights, a, prop.mu, prop.var), dict(det, kind=det["kind"] + "_blockoracle")
    return split


def bern_entropy(q):
    return float(-(q * np.log(q + 1e-300) + (1 - q) * np.log1p(-q + 1e-300)).sum())


def selftest():
    tau, w = 2.0, 0.2
    for design, rho, n in (("M", 0.9, 25), ("U", 0.9, 25)):
        Xb, y, _ = make_data(design, rho, n, 0, 10)
        X, yy = embed_diag(Xb, y, 12, [1, 2, 3])
        full = exact_posterior(Model(X, yy, sigma=1.0, tau=tau, omega=w))
        blk = exact_posterior(Model(X[:, :10], yy, sigma=1.0, tau=tau, omega=w))
        blk0 = exact_posterior(Model(Xb, y, sigma=1.0, tau=tau, omega=w))
        code = full["masks"][:, :10] @ (2 ** np.arange(10)[::-1])
        marg = np.bincount(code, weights=full["support"], minlength=2 ** 10)
        npip = exact_nuisance_pip(X, yy, 10, w, tau, 0.0)
        e1 = np.abs(marg - blk["support"]).max(); e2 = np.abs(blk["support"] - blk0["support"]).max()
        e3 = np.abs(full["pip"][10:] - npip).max()
        print(f"{design}: block marginal of full enumeration vs block-only {e1:.1e}; embedded vs raw block {e2:.1e};"
              f" nuisance PIP vs closed form {e3:.1e}")
        assert e1 < 1e-8 and e2 < 1e-8 and e3 < 1e-8
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--design", choices=["M", "U"])
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--P", type=int, nargs="+", default=[0])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rep0", type=int, default=2)
    ap.add_argument("--slab", type=float, default=2.0, help="their tau (slab s.d.)")
    ap.add_argument("--w", type=float, default=0.2, help="their omega (prior inclusion), fixed in P")
    ap.add_argument("--max-seconds", dest="max_seconds", type=float, default=600.0)
    ap.add_argument("--max-components", dest="max_components", type=int, default=10)
    ap.add_argument("--arms", nargs="+", default=["asis", "oracle"], choices=["asis", "oracle"])
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    f = open(a.out, "w", newline=""); wr = None
    for rep in range(a.rep0, a.rep0 + a.reps):
        Xb, y, _ = make_data(a.design, a.rho, a.n, rep, a.p)
        g = Xb.shape[1]
        key = [ord(a.design), int(round(a.rho * 1000)), 0, a.n, rep, 77]      # as experiment G
        for P in a.P:
            X, yy = embed_diag(Xb, y, P, key)
            Pe = X.shape[1]
            ex = exact_posterior(Model(X[:, :g], yy, sigma=1.0, tau=a.slab, omega=a.w))
            npip = exact_nuisance_pip(X, yy, g, a.w, a.slab, 0.0) if Pe > g else None
            top = np.sort(ex["support"])[::-1]
            row = dict(design=a.design, rho=a.rho, n=a.n, rep=rep, P=Pe, g=g, slab=a.slab, w=a.w,
                       max_seconds=a.max_seconds, ex_maxpat=float(top[0]),
                       ex_n90=int(np.searchsorted(np.cumsum(top), 0.9) + 1),
                       ex_nuis_ent=bern_entropy(npip) if npip is not None else np.nan,
                       ex_nuis_pip=float(npip.mean()) if npip is not None else np.nan)
            model = RegressionModel(X, yy, 1.0, a.slab, a.w)
            try:
                t0 = time.perf_counter()
                _, base, _ = strong_meanfield(model, seed=rep)
                mf_s = time.perf_counter() - t0
                a0 = np.asarray(base["mixture"].a, float)[0]
                sel20 = np.argsort(-a0, kind="stable")[:min(20, Pe)]
                row.update(mf_seconds=mf_s, split_block_sel=int((sel20 < g).sum()),
                           max_nuis_a=float(a0[g:].max()) if Pe > g else np.nan)
            except Exception as e:
                row.update(error="meanfield: " + repr(e)[:180]); base = None
            for arm in a.arms:
                r = dict(row, arm=arm)
                if base is not None:
                    try:
                        AS._split = block_split(g) if arm == "oracle" else _SPLIT
                        res = fit(model, Settings(max_components=a.max_components, max_seconds=a.max_seconds),
                                  seed=rep, baseline=base)
                        r.update(fit_seconds=res["fit_seconds"], K=res["components"],
                                 K_cand=res["candidate_components"], status=res["status"],
                                 fallback=res["final_fallback"], gain_mf=-res["returned_delta_mf"], error="")
                        r.update({f"{k}_mix": v for k, v in score(res["mixture"], g, ex, npip).items()})
                        r.update({f"{k}_mf": v for k, v in score(res["baseline"], g, ex, npip).items()})
                        r.update(structure(res["mixture"], g))
                    except Exception as e:                                   # record, keep going
                        r.update(error=repr(e)[:200])
                    finally:
                        AS._split = _SPLIT
                if wr is None:
                    cols = ["arm"] + list(row) + ["fit_seconds", "K", "K_cand", "status", "fallback", "gain_mf", "error",
                            "tv_mix", "pip_mae_mix", "nuis_pip_mix", "nuis_pip_mae_mix", "tv_mf", "pip_mae_mf",
                            "nuis_pip_mf", "nuis_pip_mae_mf", "n_groups_block", "n_groups_all", "a_diff_block",
                            "a_diff_nuis"]
                    cols = list(dict.fromkeys(cols))
                    wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval=""); wr.writeheader()
                wr.writerow(r); f.flush()
                print(f"  {a.design} rho={a.rho} n={a.n} rep={rep} P={Pe} {arm}: K={r.get('K')} {r.get('status')} "
                      f"tv mix/mf={r.get('tv_mix', np.nan):.3f}/{r.get('tv_mf', np.nan):.3f} "
                      f"t={r.get('mf_seconds', 0) + r.get('fit_seconds', 0):.0f}s {r.get('error', '')}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
