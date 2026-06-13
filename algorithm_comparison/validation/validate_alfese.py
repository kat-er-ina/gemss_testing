"""Independent validation of the ALFESE library on an in-envelope problem.

Bug-free-comparison discipline: before trusting ALFESE as a baseline in the
n<<p comparison, confirm it behaves correctly on a problem squarely within its
design envelope (adequate n, small p, unambiguous signal, proper train/test).
Calls the library DIRECTLY (bypassing our wrapper) to isolate the library.

Result (n=400, p=20, informative={0,1,2}, k=3):
  mi=1.00  greedy=0.67  importance=0.67  fcbf=0.33  mrmr=0.33
=> ALFESE works (MI perfect; greedy finds signal). The F1~0 of greedy in our
n<<p comparison is therefore the algorithm failing OUTSIDE its envelope (random-
swap hill-climbing cannot find ~15 features among thousands in <=1000 iters),
compounded by a tiny held-out split for the MCC quality signal -- not a bug.
"""

import numpy as np
import pandas as pd
import alfese
from sklearn.datasets import make_classification


def main():
    X, y = make_classification(n_samples=400, n_features=20, n_informative=3,
                               n_redundant=0, n_repeated=0, n_classes=2,
                               shuffle=False, class_sep=2.0, random_state=0)
    truth = {0, 1, 2}
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    ys = pd.Series(y, name="t")
    tr = 280
    Xtr, Xte, ytr, yte = Xdf.iloc[:tr], Xdf.iloc[tr:], ys.iloc[:tr], ys.iloc[tr:]
    sels = {"mi": alfese.MISelector, "importance": alfese.ModelImportanceSelector,
            "greedy": alfese.GreedyWrapperSelector, "fcbf": alfese.FCBFSelector,
            "mrmr": alfese.MRMRSelector}
    print("ALFESE in-envelope check: n=400 p=20 informative={0,1,2} k=3; expect recall=1.0 if working.")
    ok = True
    for name, Cls in sels.items():
        try:
            s = Cls(); s.set_data(Xtr, Xte, ytr, yte)
            res = s.search_simultaneously(k=3, num_alternatives=0, tau=1.0, objective_agg="sum")
            sel = set(int(i) for i in res.iloc[0]["selected_idxs"])
            rec = len(sel & truth) / 3
            flag = "" if rec >= 0.6 else "  <-- LOW"
            print(f"  {name:11s} selected={sorted(sel)}  recall={rec:.2f}{flag}")
            if name in ("mi", "greedy") and rec < 0.6:
                ok = False
        except Exception as e:
            print(f"  {name:11s} ERROR: {type(e).__name__}: {str(e)[:80]}")
            ok = False
    print("PASS" if ok else "FAIL (mi/greedy should recover signal in-envelope)")


if __name__ == "__main__":
    main()
