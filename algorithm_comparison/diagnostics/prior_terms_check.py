"""Plan sec. 50: checks for the prior options of the F3 target.
1. Closed-form (log p, gradient, Hessian diagonal) of every supported prior vs autograd of the prior's own log_prob
   (Student-t and standard spike-and-slab; the exact sss is checked in exact_prior_check.py).
2. Smoke: d_select's semi mode with each --prior through F3J and F3Ens (short runs).
"""
import os, subprocess, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemss.feature_selection.models import SpikeAndSlabPrior, StudentTPrior  # noqa: E402
from src.wrappers.weightfix_gemss import RegressionTarget  # noqa: E402

torch.manual_seed(0)
z = (torch.randn(16, 50, dtype=torch.float64) * torch.tensor([0.003, 0.05, 0.5, 3.0], dtype=torch.float64).repeat(13)[:50])
for P in [StudentTPrior(df=3.0, scale=1.0), StudentTPrior(df=1.0, scale=0.01), StudentTPrior(df=1.0, scale=0.1),
          SpikeAndSlabPrior(32.0, 0.1, 0.05, 0.95), SpikeAndSlabPrior(32.0, 0.01, 0.05, 0.95)]:
    t = RegressionTarget.__new__(RegressionTarget); t.prior = P
    lp, g, H = t._prior_terms(z)
    zz = z.clone().requires_grad_(True)
    lpa = P.log_prob(zz)
    ga, = torch.autograd.grad(lpa.sum(), zz, create_graph=True)
    Ha = torch.stack([torch.autograd.grad(ga[:, j].sum(), zz, retain_graph=True)[0][:, j] for j in range(z.shape[1])], 1)
    c = (lp - lpa).detach()
    print(f"{type(P).__name__:18s} {vars(P)}: |dlogp - const| {float((c - c.mean()).abs().max()):.1e}  "
          f"|dg| {float((g - ga).abs().max()):.1e}  |dH| {float((H - Ha).abs().max() / Ha.abs().max()):.1e} (rel)")
if "--smoke" in sys.argv:
    here = os.path.dirname(os.path.abspath(__file__))
    for pr, extra in [("ss", ["--var-spike", "0.01"]), ("student", ["--t-df", "1", "--t-scale", "0.1"]),
                      ("exact", ["--var-spike", "0.03"])]:
        for arm in ["F3J", "F3Ens"]:
            out = f"/tmp/_prior_smoke_{pr}_{arm}.csv" if len(sys.argv) < 3 else os.path.join(sys.argv[2], f"{pr}_{arm}.csv")
            cmd = [sys.executable, os.path.join(here, "d_select.py"), "semi", "--no-cv", "--xdata", "diabetes", "--S", "3",
                   "--pve", "0.6", "--reps", "1", "--D", "3", "--arm", arm, "--prior", pr, *extra, "--n-iter", "300",
                   "--out", out]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.join(here, ".."))
            print(pr, arm, "OK" if r.returncode == 0 else "FAIL", (r.stdout.strip().splitlines() or [""])[-1][:160],
                  r.stderr[-400:] if r.returncode else "")
