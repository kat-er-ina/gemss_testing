"""§5 real-world solution dissimilarity.

Computes the mean pairwise dissimilarity (1 - Jaccard) over the eight candidate
solutions that the GEMSS Explorer returns for each real dataset, reproducing the
diabetes (0.91) and Arabidopsis (0.97) numbers cited in Section 5.

The third §5 dataset, food-science (0.92), is NOT reproduced here: its data is
not public and is deliberately excluded from this companion.

Inputs are the Explorer exports (the no-code app's output), vendored under
algorithm_comparison/explorer_reports/:
  * diabetes_report.html, arabidopsis_report.html  -- marimo HTML exports; each
    embeds `feature_names":[...]` arrays, the first 8 of which are the candidates'
    (full) feature lists.

Usage:
  .venv/bin/python scripts/explorer_dissim.py [--reports-dir PATH]
"""
import argparse
import itertools
import os
import re
import statistics as st


def _dissim(solutions):
    sets = [set(s) for s in solutions if s]
    if len(sets) < 2:
        return None
    pw = [1 - len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
    return {"mean": st.mean(pw), "min": min(pw), "max": max(pw),
            "n_pairs": len(pw), "sizes": [len(s) for s in sets]}


def from_html(path, ncand=8):
    """First `ncand` `feature_names` arrays = the candidate solutions (full lists)."""
    s = open(path, encoding="utf-8", errors="replace").read()
    # unescape the marimo/HTML layering: & -> & ; &#92; -> \ ; &quot; -> " ; \" -> "
    s = s.replace("\\u0026", "&").replace("&#92;", "\\").replace("&quot;", '"').replace('\\"', '"')
    arrays = re.findall(r'feature_names"\s*:\s*\[(.*?)\]', s)
    sols = [sorted(set(re.findall(r'"([^"]+)"', a))) for a in arrays]
    sols = [x for x in sols if x]
    return sols[:ncand]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_reports = os.path.normpath(os.path.join(here, "..", "explorer_reports"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default=default_reports)
    args = ap.parse_args()

    datasets = [
        ("diabetes", from_html, "diabetes_report.html"),
        ("Arabidopsis", from_html, "arabidopsis_report.html"),
    ]
    print(f"reports dir: {args.reports_dir}\n")
    print(f"{'dataset':14s} {'#sol':>4s} {'dissim mean':>11s} {'range':>16s}")
    for name, loader, fname in datasets:
        path = os.path.join(args.reports_dir, fname)
        if not os.path.exists(path):
            print(f"{name:14s}  MISSING: {path}")
            continue
        sols = loader(path)
        r = _dissim(sols)
        if r is None:
            print(f"{name:14s}  <2 solutions")
            continue
        print(f"{name:14s} {len(sols):>4d} {r['mean']:>11.3f}   [{r['min']:.3f}, {r['max']:.3f}]")


if __name__ == "__main__":
    main()
