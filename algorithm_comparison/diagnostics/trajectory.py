"""R5 of the special-issue comparison plan (Round 3): GEMSS training trajectory.

One GEMSS 'joint' fit per (seed, K), run in chunks; at each checkpoint the fit is
read out exactly as tuned_compare does (top-D per component -> K->m clustering)
and scored, together with label-free signals that could select a stopping
point without ground truth:
  elbo        large-batch ELBO estimate (256 draws, as the wrapper's final ELBO)
  raw_dissim  mean pairwise Jaccard distance among the K raw top-D supports
  n_distinct  number of distinct raw top-D supports
  perplexity  exp(entropy(alpha))
The torch RNG state is saved and restored around every checkpoint evaluation,
so the optimisation path is identical to a single uninterrupted run: the
6000-step checkpoint reproduces tuned_compare --torchseed (R4, r = 2) and the
24000-step checkpoint reproduces R2.
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

    fields = ["prior", "rank", "seed", "K", "step", "union_f1", "sol_f1", "dissim",
              "signal_share_topD", "elbo", "raw_dissim", "n_distinct", "perplexity",
              "seconds", "supports"]
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); f.flush()
    cps = sorted(args.checkpoints)
    for seed in args.seeds:
        X, y, truth, planted = generate_overlapping_dataset(
            n_samples=args.n, n_features=args.p, n_solutions=3, sparsity=args.D,
            latent_rank=args.rank, overlap=args.overlap, noise_std=args.noise,
            nan_ratio=0.0, binarize=False, nonlinear=False, noise_dist="gauss", seed=seed)
        truth = set(truth)
        for K in args.K:
            torch.manual_seed(seed)                      # = tuned_compare --torchseed
            sel = BayesianFeatureSelector(n_features=args.p, n_components=K, X=X, y=y,
                                          sss_sparsity=args.D, n_iter=args.chunk, **sel_kwargs)
            done, t0, mus = 0, time.perf_counter(), {}
            for cp in cps:
                while done < cp:
                    sel.n_iter = min(args.chunk, cp - done)
                    sel.optimize(regularize=False, verbose=False)
                    done += sel.n_iter
                state = torch.get_rng_state()
                with torch.no_grad():
                    z_eval, _ = sel.mixture.sample(max(256, sel.batch_size))
                    elbo = float(sel.elbo(z_eval).item())
                torch.set_rng_state(state)
                mu = sel.mixture.mu.detach().cpu().numpy()
                alpha = sel.mixture.get_alpha().detach().cpu().numpy()
                raw = [tuple(sorted(np.argsort(np.abs(mu[k]))[::-1][:args.D].tolist())) for k in range(K)]
                sols = cluster_supports(raw, args.p, args.m, args.D)
                sm = calculate_structural_metrics(sols, planted)
                uf = calculate_metrics(sols, truth, args.p)["F1_Score"]
                a = alpha[alpha > 0]
                w.writerow(dict(
                    prior=args.prior, rank=args.rank, seed=seed, K=K, step=cp, union_f1=uf,
                    sol_f1=sm["sol_f1"], dissim=mean_dissim(s["support"] for s in sols.values()),
                    signal_share_topD=float(np.mean([len(set(s) & truth) / args.D for s in raw])),
                    elbo=elbo, raw_dissim=mean_dissim(raw), n_distinct=len(set(raw)),
                    perplexity=float(np.exp(-(a * np.log(a)).sum())),
                    seconds=time.perf_counter() - t0,
                    supports=";".join(",".join(map(str, sorted(s["support"]))) for s in sols.values())))
                f.flush()
                mus[f"mu_{cp}"], mus[f"alpha_{cp}"] = mu, alpha
                print(f"  {args.prior} sd={seed} K={K} step={cp} u={uf:.3f} elbo={elbo:.1f}", flush=True)
            if args.dump_dir:
                np.savez_compressed(os.path.join(args.dump_dir, f"{args.prior}_r{args.rank}_sd{seed}_K{K}.npz"), **mus)
    f.close()


if __name__ == "__main__":
    main()
