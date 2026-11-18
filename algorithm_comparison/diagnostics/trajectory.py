"""R5 of the special-issue comparison plan (Round 3): GEMSS training trajectory.

One GEMSS 'joint' fit per (seed, K), run in chunks; at each checkpoint the fit is
read out exactly as tuned_compare does (top-D per component -> K->m clustering)
and scored, together with label-free signals that could select a stopping
point without ground truth:
  elbo        large-batch ELBO estimate (256 draws, as the wrapper's final ELBO)
  raw_dissim  mean pairwise Jaccard distance among the K raw top-D supports
  n_distinct  number of distinct raw top-D supports
  perplexity  exp(entropy(alpha))
Plan sec. 14 adds --estimator (src/wrappers/weightfix_gemss.py): 'current' (default)
is GEMSS's own loop and reproduces the R5 rows bit for bit; frozen / strat / score /
ngvi / fixedpoint replace only the optimisation loop. Extra columns: estimator,
alpha_max, union_f1_topw (union-F1 of the m highest-weight components' top-D
supports, no clustering), signal_share_w (alpha-weighted top-D signal share).
The torch RNG state is saved and restored around every checkpoint evaluation,
so the optimisation path is identical to a single uninterrupted run: the
6000-step checkpoint reproduces tuned_compare --torchseed (R4, r = 2) and the
24000-step checkpoint reproduces R2.
Plan sec. 18 (coupling test) adds: --init (start options of weightfix_gemss.apply_init), the
coupling estimators f0_joint / f0_dec / f0_ind / ngvi_dec, the original repulsion probe's
independent control 'probe_ind' (K separate one-component GEMSS fits seeded 1000*seed+k, as in
stresstest repulsion_probe.py), --task classification (binarized response, Bernoulli likelihood via
LogisticBayesianFeatureSelector, as the probe), --var-slab, and an --arm label column.
"""
import argparse, csv, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hard_data_factory import generate_overlapping_dataset
from src.evaluation import calculate_structural_metrics, calculate_metrics, _jaccard
from gemss.feature_selection.inference import BayesianFeatureSelector
from ensemble_parity import cluster_supports
from tuned_compare import GEMSS_HP
from src.wrappers.weightfix_gemss import ESTIMATORS, INITS, make_runner, apply_init
from src.wrappers.logistic_gemss import LogisticBayesianFeatureSelector


class ProbeIndependent:
    """repulsion_probe.py's 'independent' scheme: K separate one-component fits, seed 1000*seed+k,
    each GEMSS's own loop with 16 draws per step. Mixture-like accessors for the readout."""

    def __init__(self, cls, K, seed, **kw):
        self.sels = []
        for k in range(K):
            torch.manual_seed(1000 * seed + k)
            self.sels.append(cls(n_components=1, **kw))
        self.runners = [make_runner(sl, "current") for sl in self.sels]
        self.batch_size = self.sels[0].batch_size

    def run(self, n):
        for r in self.runners:
            r.run(n)

    def state(self):
        mu = torch.cat([sl.mixture.mu.detach() for sl in self.sels])
        var = torch.cat([sl.mixture.get_variance().detach() for sl in self.sels])
        with torch.no_grad():
            elbo = float(np.mean([sl.elbo(sl.mixture.sample(256)[0]).item() for sl in self.sels]))
        return mu, var, torch.full((len(self.sels),), 1.0 / len(self.sels)), elbo


def mean_dissim(sets):
    sets = [set(s) for s in sets if s]
    if len(sets) < 2:
        return 0.0
    return float(np.mean([1 - _jaccard(sets[a], sets[b])
                          for a in range(len(sets)) for b in range(a + 1, len(sets))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["sss", "ss"], required=True,
                    help="sss = GEMSSjoint; ss = GEMSSjointSS (slab weight D/p)")
    ap.add_argument("--K", type=int, nargs="+", default=[3, 6, 12, 18, 24])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--D", type=int, default=10)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--checkpoints", type=int, nargs="+",
                    default=[1000, 2000, 4000, 6000, 12000, 24000, 48000])
    ap.add_argument("--chunk", type=int, default=2000,
                    help="max steps per optimize() call (bounds the per-step history memory)")
    ap.add_argument("--dump-dir", dest="dump_dir", default=None)
    ap.add_argument("--estimator", choices=ESTIMATORS + ("probe_ind",), default="current")
    ap.add_argument("--init", choices=INITS, default="default")
    ap.add_argument("--task", choices=["regression", "classification"], default="regression")
    ap.add_argument("--var-slab", dest="var_slab", type=float, default=None)
    ap.add_argument("--arm", default=None, help="label written to the 'arm' column (default: estimator)")
    ap.add_argument("--anneal-omega0", dest="anneal_omega0", type=float, default=None,
                    help="likelihood tempering: omega rises geometrically from this value to 1 (plan sec. 19)")
    ap.add_argument("--anneal-steps", dest="anneal_steps", type=int, default=4000)
    ap.add_argument("--strat-draws", dest="strat_draws", type=int, default=None)
    ap.add_argument("--ngvi-beta", dest="ngvi_beta", type=float, default=0.02)
    ap.add_argument("--ngvi-beta-w", dest="ngvi_beta_w", type=float, default=None)
    ap.add_argument("--fp-every", dest="fp_every", type=int, default=100)
    ap.add_argument("--fp-draws", dest="fp_draws", type=int, default=64)
    ap.add_argument("--fp-eta", dest="fp_eta", type=float, default=1.0)
    ap.add_argument("--step-rule", dest="step_rule", choices=["fixed", "invL"], default="fixed",
                    help="plan sec. 37: F3 step size fixed (ngvi_beta) or min(ngvi_beta, 1/L) per component")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
    hp = dict(GEMSS_HP)
    if args.prior == "ss":
        ws = args.D / args.p
        hp.update(prior="ss", weight_slab=ws, weight_spike=1 - ws)
    keys = ("prior", "var_spike", "weight_slab", "weight_spike", "lr", "batch_size")
    sel_kwargs = {k: hp[k] for k in keys}
    if args.var_slab is not None:
        sel_kwargs["var_slab"] = args.var_slab
    cls = LogisticBayesianFeatureSelector if args.task == "classification" else BayesianFeatureSelector
    arm = args.arm or args.estimator

    fields = ["prior", "rank", "seed", "K", "step", "union_f1", "sol_f1", "dissim",
              "signal_share_topD", "elbo", "raw_dissim", "n_distinct", "perplexity",
              "seconds", "supports", "estimator", "alpha_max", "union_f1_topw", "signal_share_w",
              "arm", "init", "overlap", "task", "p", "D", "anneal", "n", "noise", "step_rule", "step_mean"]
    est_opts = dict(strat_draws=args.strat_draws, ngvi_beta=args.ngvi_beta, ngvi_beta_w=args.ngvi_beta_w,
                    fp_every=args.fp_every, fp_draws=args.fp_draws, fp_eta=args.fp_eta,
                    anneal=(args.anneal_omega0, args.anneal_steps) if args.anneal_omega0 else None)
    if args.step_rule != "fixed":
        est_opts["step_rule"] = args.step_rule
    anneal_tag = f"{args.anneal_omega0}/{args.anneal_steps}" if args.anneal_omega0 else "none"
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); f.flush()
    cps = sorted(args.checkpoints)
    for seed in args.seeds:
        X, y, truth, planted = generate_overlapping_dataset(
            n_samples=args.n, n_features=args.p, n_solutions=3, sparsity=args.D,
            latent_rank=args.rank, overlap=args.overlap, noise_std=args.noise,
            nan_ratio=0.0, binarize=(args.task == "classification"), nonlinear=False, noise_dist="gauss",
            seed=seed)
        truth = set(truth)
        for K in args.K:
            if args.estimator == "probe_ind":
                sel = runner = ProbeIndependent(cls, K, seed, n_features=args.p, X=X, y=y, sss_sparsity=args.D,
                                                n_iter=args.chunk, **sel_kwargs)
            else:
                torch.manual_seed(seed)                      # = tuned_compare --torchseed
                sel = cls(n_features=args.p, n_components=K, X=X, y=y,
                          sss_sparsity=args.D, n_iter=args.chunk, **sel_kwargs)
                apply_init(sel.mixture, args.init, X, y, args.D, seed)
                runner = make_runner(sel, args.estimator, **est_opts)
            done, t0, mus = 0, time.perf_counter(), {}
            for cp in cps:
                while done < cp:
                    n = min(args.chunk, cp - done)
                    runner.run(n)                        # 'current': sel.optimize, as before
                    done += n
                state = torch.get_rng_state()
                if args.estimator == "probe_ind":
                    mu_t, var_t, alpha_t, elbo = sel.state()
                else:
                    with torch.no_grad():
                        z_eval, _ = sel.mixture.sample(max(256, sel.batch_size))
                        elbo = float(sel.elbo(z_eval).item())
                    mu_t, var_t, alpha_t = sel.mixture.mu.detach(), sel.mixture.get_variance().detach(), \
                        sel.mixture.get_alpha().detach()
                torch.set_rng_state(state)
                mu = mu_t.cpu().numpy().copy()     # copy: .numpy() of the parameter is a view that later steps overwrite
                alpha = alpha_t.cpu().numpy()
                raw = [tuple(sorted(np.argsort(np.abs(mu[k]))[::-1][:args.D].tolist())) for k in range(K)]
                sols = cluster_supports(raw, args.p, args.m, args.D)
                sm = calculate_structural_metrics(sols, planted)
                uf = calculate_metrics(sols, truth, args.p)["F1_Score"]
                a = alpha[alpha > 0]
                share = np.array([len(set(s) & truth) / args.D for s in raw])
                topw = np.argsort(-alpha)[:args.m]
                uf_w = calculate_metrics({f"w{i}": {"support": list(raw[k])} for i, k in enumerate(topw)},
                                         truth, args.p)["F1_Score"]
                w.writerow(dict(
                    prior=args.prior, rank=args.rank, seed=seed, K=K, step=cp, union_f1=uf,
                    sol_f1=sm["sol_f1"], dissim=mean_dissim(s["support"] for s in sols.values()),
                    signal_share_topD=float(np.mean([len(set(s) & truth) / args.D for s in raw])),
                    elbo=elbo, raw_dissim=mean_dissim(raw), n_distinct=len(set(raw)),
                    perplexity=float(np.exp(-(a * np.log(a)).sum())),
                    seconds=time.perf_counter() - t0,
                    supports=";".join(",".join(map(str, sorted(s["support"]))) for s in sols.values()),
                    estimator=args.estimator, alpha_max=float(alpha.max()), union_f1_topw=uf_w,
                    signal_share_w=float(alpha @ share), arm=arm, init=args.init, overlap=args.overlap,
                    task=args.task, p=args.p, D=args.D, anneal=anneal_tag, n=args.n, noise=args.noise,
                    step_rule=args.step_rule,
                    step_mean=float(np.mean(runner.step_hist)) if getattr(runner, "step_hist", None) else np.nan))
                f.flush()
                mus[f"mu_{cp}"], mus[f"alpha_{cp}"] = mu, alpha
                mus[f"var_{cp}"] = var_t.cpu().numpy()
                print(f"  {args.prior}/{arm}/{args.init} ov={args.overlap} sd={seed} K={K} step={cp} u={uf:.3f} u_w={uf_w:.3f} "
                      f"elbo={elbo:.1f} amax={alpha.max():.2f}", flush=True)
            if args.dump_dir:
                np.savez_compressed(os.path.join(args.dump_dir, f"{args.prior}_r{args.rank}_sd{seed}_K{K}.npz"), **mus)
    f.close()


if __name__ == "__main__":
    main()
