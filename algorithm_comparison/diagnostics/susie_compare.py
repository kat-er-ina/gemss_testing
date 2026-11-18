"""Plan sec. 39: SuSiE (Wang et al. 2020, reference implementation susieR via Rscript, defaults, L = D) on the
designs where GEMSS's joint fit and the ensemble of independent fits were compared (secs. 33, 35, 38).

SuSiE's posterior is a product over L single effects; each credible set (CS) is a set of interchangeable features
for one effect. Alternatives are read from it as supports with one feature per CS (CS members only, probabilities
alpha_l renormalised on the CS), then passed through the SAME readout as GEMSS (cluster_supports to m solutions,
same metrics):
  SuSiE-top   the K most probable distinct supports (primary readout)
  SuSiE-samp  K independent draws (seeded)
Extra columns (one fit, written on both rows): n_cs, cs_sizes, cs_purity_min, union_f1_csunion (union-F1 of
all CS members as one set), union_f1_pipD (union-F1 of the top-D features by PIP as one set), sigma2, elbo,
niter, converged, susie_seconds.

Usage:
  susie_compare.py synth --n 30 --p 1000 --overlap 2 --noise 0.05 --seeds 1 .. 15 --out x.csv   [same generator
      call as trajectory.py: 3 planted solutions, latent rank 2, D 10; K 12, m 3]
  susie_compare.py real --dataset diabetes|arabidopsis|preterm --seeds 1 .. 20 --out x.csv      [RQ3 protocol as
      realdata_pairs.py: standardised X, linear SuSiE on the 0/1 label, K 12, D 10, m 8, best-of-11 end-model F1]
  susie_compare.py selftest
"""
import argparse, glob, heapq, os, shutil, subprocess, sys, tempfile, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for q in (ROOT, HERE, os.path.join(ROOT, "scripts")):
    sys.path.insert(0, q)
from src.hard_data_factory import generate_overlapping_dataset          # noqa: E402
from src.evaluation import calculate_structural_metrics, calculate_metrics, _jaccard   # noqa: E402
from ensemble_parity import cluster_supports                           # noqa: E402

RSCRIPT_FIT = os.path.join(HERE, "susie_fit.R")


def run_susie(jobs, L, rscript="Rscript", keep=None):
    """jobs: list of (X, y). One Rscript call for all. Returns list of dicts (alpha L x p, pip, cs, info)."""
    tmp = keep or tempfile.mkdtemp(prefix="susie_")
    dirs = []
    for i, (X, y) in enumerate(jobs):
        d = os.path.join(tmp, f"job{i}"); os.makedirs(d, exist_ok=True)
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        with open(os.path.join(d, "dims.txt"), "w") as f:
            f.write(f"{X.shape[0]} {X.shape[1]} {L}\n")
        X.astype("<f8").ravel(order="F").tofile(os.path.join(d, "X.bin"))
        y.astype("<f8").tofile(os.path.join(d, "y.bin"))
        dirs.append(d)
    subprocess.run([rscript, RSCRIPT_FIT] + dirs, check=True)
    out = []
    for (X, _), d in zip(jobs, dirs):
        p = X.shape[1]
        alpha = np.fromfile(os.path.join(d, "alpha.bin"), "<f8").reshape((L, p), order="F")
        pip = np.fromfile(os.path.join(d, "pip.bin"), "<f8")
        cs = []
        for line in open(os.path.join(d, "cs.txt")).read().splitlines():
            if line.strip():
                l, pur, mem = line.split(" ")
                cs.append((int(l), float(pur), [int(j) for j in mem.split(",")]))
        info = {}
        for line in open(os.path.join(d, "info.txt")).read().splitlines():
            k, v = line.split(" ", 1)
            info[k] = v
        out.append(dict(alpha=alpha, pip=pip, cs=cs, info=info))
    if keep is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def cs_dists(res):
    """Per CS: (members, probabilities) with alpha_l renormalised on the CS, members sorted by probability."""
    dists = []
    for l, _, mem in res["cs"]:
        pr = res["alpha"][l, mem]; pr = pr / pr.sum()
        o = np.argsort(-pr)
        dists.append((np.asarray(mem)[o], pr[o]))
    return dists


def top_supports(dists, K, n_configs=2000):
    """K most probable distinct supports (one feature per CS; a support's probability sums over the
    configurations that give the same feature set). Best-first enumeration of the product distribution."""
    if not dists:
        return []
    start = (0,) * len(dists)
    lp = lambda c: sum(np.log(dists[i][1][j]) for i, j in enumerate(c))
    heap, seen, agg = [(-lp(start), start)], {start}, {}
    while heap and len(seen) <= n_configs:
        nlp, c = heapq.heappop(heap)
        s = tuple(sorted({int(dists[i][0][j]) for i, j in enumerate(c)}))
        agg[s] = agg.get(s, 0.0) + np.exp(-nlp)
        for i in range(len(c)):
            if c[i] + 1 < len(dists[i][0]):
                nc = c[:i] + (c[i] + 1,) + c[i + 1:]
                if nc not in seen:
                    seen.add(nc); heapq.heappush(heap, (-lp(nc), nc))
    return [s for s, _ in sorted(agg.items(), key=lambda kv: -kv[1])[:K]]


def sample_supports(dists, K, rng):
    if not dists:
        return []
    return [tuple(sorted({int(rng.choice(m, p=pr)) for m, pr in dists})) for _ in range(K)]


def mean_dissim(sets):
    sets = [set(s) for s in sets if s]
    if len(sets) < 2:
        return 0.0
    return float(np.mean([1 - _jaccard(sets[a], sets[b]) for a in range(len(sets)) for b in range(a + 1, len(sets))]))


def union_f1(sets, truth, p):
    sets = [list(s) for s in sets if s]
    if not sets:
        return 0.0
    return calculate_metrics({f"s{i}": {"support": s} for i, s in enumerate(sets)}, truth, p)["F1_Score"]


def fit_columns(res, truth, p, D):
    cs_union = sorted({j for _, _, mem in res["cs"] for j in mem})
    row = dict(n_cs=len(res["cs"]), cs_sizes=";".join(str(len(m)) for _, _, m in res["cs"]),
               cs_purity_min=min((pu for _, pu, _ in res["cs"]), default=np.nan),
               sigma2=float(res["info"]["sigma2"]), elbo=float(res["info"]["elbo"]),
               niter=int(res["info"]["niter"]), converged=int(res["info"]["converged"]),
               susie_seconds=float(res["info"]["seconds"]), susie_version=res["info"]["version"])
    if truth is not None:
        row["union_f1_csunion"] = union_f1([cs_union], truth, p)
        row["union_f1_pipD"] = union_f1([np.argsort(-res["pip"])[:D].tolist()], truth, p)
    return row


def synth(a):
    jobs, meta = [], []
    for seed in a.seeds:
        X, y, truth, planted = generate_overlapping_dataset(
            n_samples=a.n, n_features=a.p, n_solutions=3, sparsity=a.D, latent_rank=a.rank, overlap=a.overlap,
            noise_std=a.noise, nan_ratio=0.0, binarize=False, nonlinear=False, noise_dist="gauss", seed=seed)
        jobs.append((X, y)); meta.append((seed, set(truth), planted))
    t0 = time.perf_counter()
    fits = run_susie(jobs, a.D, a.rscript)
    wall = (time.perf_counter() - t0) / len(jobs)
    rows = []
    for (seed, truth, planted), res in zip(meta, fits):
        base = fit_columns(res, truth, a.p, a.D)
        dists = cs_dists(res)
        for arm, raw in (("SuSiE-top", top_supports(dists, a.K)),
                         ("SuSiE-samp", sample_supports(dists, a.K, np.random.default_rng(seed)))):
            sols = cluster_supports(raw, a.p, a.m, a.D) if raw else {}
            sol_f1 = calculate_structural_metrics(sols, planted)["sol_f1"] if sols else 0.0
            rows.append(dict(prior="susie", arm=arm, seed=seed, K=a.K, step=0, n=a.n, p=a.p, overlap=a.overlap,
                             noise=a.noise, D=a.D, m=a.m, rank=a.rank,
                             union_f1=union_f1([s["support"] for s in sols.values()], truth, a.p), sol_f1=sol_f1,
                             dissim=mean_dissim(s["support"] for s in sols.values()), n_distinct=len(set(raw)),
                             raw_dissim=mean_dissim(raw), seconds=wall,
                             supports=";".join(",".join(map(str, sorted(s["support"]))) for s in sols.values()),
                             **base))
        print(f"  susie n={a.n} p={a.p} ov={a.overlap} nz={a.noise} sd={seed}: n_cs {base['n_cs']} sizes "
              f"[{base['cs_sizes']}] u_top {rows[-2]['union_f1']:.3f} sol_top {rows[-2]['sol_f1']:.3f} "
              f"u_cs {base['union_f1_csunion']:.3f} u_pipD {base['union_f1_pipD']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


def real(a):
    import run_rq3_clean as rq3
    from realdata_pairs import data
    from sklearn.preprocessing import StandardScaler
    df_feat, y, fnames = data(a.dataset)
    n, p = df_feat.shape
    outer_cv = "loo" if n < 40 else 5
    X = np.nan_to_num(StandardScaler().fit_transform(df_feat.values))
    t0 = time.perf_counter()
    res = run_susie([(X, np.asarray(y, float))], a.D, a.rscript)[0]
    secs = time.perf_counter() - t0
    base = fit_columns(res, None, p, a.D)
    dists = cs_dists(res)
    reads = [("SuSiE-top", 0, top_supports(dists, a.K))] + \
            [("SuSiE-samp", s, sample_supports(dists, a.K, np.random.default_rng(s))) for s in a.seeds]
    rows = []
    for arm, seed, raw in reads:
        cl = [sorted(v["support"]) for v in cluster_supports(raw, p, a.m, a.D).values() if v.get("support")] if raw else []
        named = {f"{arm}_s{i}": [fnames[j] for j in c] for i, c in enumerate(cl)}
        f1s = []
        if a.endmodel and named:
            per = rq3.best_model_f1(named, df_feat, y, outer_cv)
            f1s = [v[0] for v in per.values() if v[0] >= 0]
        dmean, dlo, dhi = rq3.dissim_stats(cl)
        rows.append(dict(dataset=a.dataset, arm=arm, seed=seed, n=n, p=p, K=a.K, D=a.D, m=a.m, nsol=len(cl),
                         f1_min=min(f1s) if f1s else np.nan, f1_mean=np.mean(f1s) if f1s else np.nan,
                         f1_best=max(f1s) if f1s else np.nan, dissim=dmean, dissim_lo=dlo, dissim_hi=dhi,
                         n_distinct_raw=len(set(raw)), fit_seconds=secs,
                         raw_supports=";".join(",".join(map(str, s)) for s in raw),
                         solutions=";".join(",".join(map(str, c)) for c in cl),
                         cs_members=";".join(",".join(fnames[j] for j in m) for _, _, m in res["cs"]), **base))
        print(f"  {a.dataset} {arm} seed {seed}: n_cs {base['n_cs']} sizes [{base['cs_sizes']}] nsol {len(cl)} "
              f"F1 best {rows[-1]['f1_best']:.3f} distinct raw {len(set(raw))}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


def selftest():
    rng = np.random.default_rng(0)
    # top_supports equals brute-force enumeration of the product distribution (with a shared member)
    dists = []
    for size in (3, 2, 4):
        pr = rng.dirichlet(np.ones(size)); o = np.argsort(-pr)
        dists.append((rng.choice(8, size, replace=False)[o], pr[o]))
    import itertools
    agg = {}
    for c in itertools.product(*[range(len(m)) for m, _ in dists]):
        s = tuple(sorted({int(dists[i][0][j]) for i, j in enumerate(c)}))
        agg[s] = agg.get(s, 0.0) + np.prod([dists[i][1][j] for i, j in enumerate(c)])
    brute = [s for s, _ in sorted(agg.items(), key=lambda kv: -kv[1])]
    got = top_supports(dists, 6)
    assert got == brute[:6], (got, brute[:6])
    # sampling frequencies match the product distribution
    smp = sample_supports(dists, 20000, np.random.default_rng(1))
    f = {s: smp.count(s) / len(smp) for s in set(smp)}
    err = max(abs(f.get(s, 0) - agg[s]) for s in agg)
    assert err < 0.02, err
    assert top_supports([], 5) == [] and sample_supports([], 5, rng) == []
    print(f"selftest ok: top-K = brute force; sampling max freq error {err:.4f}")


def rcheck(rscript):
    """Needs susieR: 3 effects, each with 2 near-duplicate proxies (corr ~0.99), n 200, p 300. The PIP recomputed
    from alpha must equal susieR's PIP, and every true effect must sit in a CS together with its proxies."""
    rng = np.random.default_rng(3)
    n, p = 200, 300
    X = rng.standard_normal((n, p))
    for j in (0, 10, 20):
        for k in (1, 2):
            X[:, j + k] = X[:, j] + 0.1 * rng.standard_normal(n)
    y = X[:, 0] - 1.5 * X[:, 10] + 2.0 * X[:, 20] + rng.standard_normal(n)
    res = run_susie([(X, y)], 10, rscript)[0]
    pip_alpha = 1 - np.prod(1 - res["alpha"], axis=0)
    e_pip = float(np.abs(pip_alpha - res["pip"]).max())
    groups = [{j, j + 1, j + 2} for j in (0, 10, 20)]
    hit = [any(g & set(mem) and set(mem) <= g for _, _, mem in res["cs"]) for g in groups]
    print(f"rcheck: susieR {res['info']['version']}, n_cs {len(res['cs'])}, CS {[m for _, _, m in res['cs']]}, "
          f"max |pip(alpha) - pip| {e_pip:.2e}, groups recovered {hit}")
    assert e_pip < 1e-8 and all(hit) and len(res["cs"]) == 3
    print("rcheck ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synth", "real", "selftest", "rcheck"])
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=4); ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--dataset", choices=["diabetes", "arabidopsis", "preterm"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("-K", type=int, default=12); ap.add_argument("-D", type=int, default=10)
    ap.add_argument("-m", type=int, default=None, help="default 3 (synth) / 8 (real), as the GEMSS runs")
    ap.add_argument("--no-endmodel", dest="endmodel", action="store_false")
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    if a.mode == "rcheck":
        return rcheck(a.rscript)
    a.m = a.m or (3 if a.mode == "synth" else 8)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    synth(a) if a.mode == "synth" else real(a)


if __name__ == "__main__":
    main()
