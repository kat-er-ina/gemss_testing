"""Plan sec. 44: tab:RQ2 under the published protocol (selection on all data) and with selection inside the CV.
Usage: analyze_rq2_honest.py <results dir>
Means over repetitions (rep 0 = the paper's folds, also shown alone). Weighted F1, as the paper's scorer.
"""
import glob, os, sys
import numpy as np, pandas as pd

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
ORDER = ["GEMSS_default", "GEMSS_outlier", "RandLasso_ensemble", "EnumLasso", "Masking_logistic", "ALFESE_mi", "BBSSL"]

d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(sys.argv[1], "*.csv")) if os.path.getsize(f) > 0],
              ignore_index=True)
pub, ins = d[d.protocol == "published"], d[d.protocol == "inside"]
fmt = lambda a, lo, hi: f"{a:.2f} [{lo:.2f}-{hi:.2f}]" if pd.notna(a) else "n/a"
for ds in ["diabetes", "preterm", "pcos"]:
    rows = []
    for m in ORDER:
        P, I = pub[(pub.dataset == ds) & (pub.method == m)], ins[(ins.dataset == ds) & (ins.method == m)]
        if P.empty and I.empty:
            continue
        r = dict(method=m, reps=f"{len(P)}/{len(I)}")
        if len(P):
            r.update(pub_nsol=P.nsol.mean(), pub_F1=fmt(P.f1_avg.mean(), P.f1_min.mean(), P.f1_max.mean()),
                     pub_dissim=P.dissim.mean())
        if len(I):
            r.update(in_nsol=I.nsol.mean(), honest_F1=fmt(I.f1_avg.mean(), I.f1_min.mean(), I.f1_max.mean()),
                     in_best11=I.f1_b11_avg.mean(), in_dissim=I.dissim.mean())
        if len(P) and len(I):
            r["drop_avg"] = I.f1_avg.mean() - P.f1_avg.mean()
        rows.append(r)
    print(f"\n================ {ds}  (pub = selection on all data; honest = selection and end-model choice inside the CV;"
          f" in_best11 = selection inside, best of 11 on held-out)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    r0 = d[(d.dataset == ds) & (d.rep == 0)]
    if len(r0):
        print("rep 0 only (the paper's folds):")
        print(r0.pivot_table(index="method", columns="protocol", values="f1_avg").reindex(
            [m for m in ORDER if m in set(r0.method)]).round(3).to_string())
