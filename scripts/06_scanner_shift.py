"""Do patient-level guarantees survive a change of scanner?

Train and calibrate on patients scanned on the dominant scanner model
(Siemens TrioTim, 3 T); evaluate on held-out TrioTim patients (in-distribution)
and on every patient scanned elsewhere (other Siemens 3 T, 1.5 T systems, GE).

Usage: python scripts/06_scanner_shift.py --features resnet50 --head elm [--repeats 5]
Output: results/scanner_shift.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from btmri import conformal as cp
from btmri.heads import RidgeHead, select_lambda_lopo
from btmri.metrics import binary_metrics
from btmri.splits import split_patients

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = np.logspace(0, 4, 5)
ALPHA = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="resnet50")
    ap.add_argument("--head", default="elm")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    t = pd.read_csv(ROOT / "data" / "dataset" / "slices.csv")
    keep = t.label.values >= 0
    t = t[keep].reset_index(drop=True)
    X = np.load(ROOT / "data" / "features" / f"{args.features}.npy")[np.flatnonzero(keep)].astype(np.float64)
    y, g = t.label.values.astype(int), t.patient.values
    width = 2048 if args.head == "elm" else 0
    trio = (t.scanner_model == "TrioTim").values
    fs = t.field_strength_T.values
    domain = np.where(trio, "TrioTim 3T (in-distribution)",
                      np.where(fs < 2, "1.5T scanners", "other 3T scanners"))
    rows = []
    for rep in range(args.repeats):
        rng = np.random.default_rng(rep)
        m = split_patients(g[trio], 0.8, rng)
        tr = np.flatnonzero(trio)[m]
        lam, _ = select_lambda_lopo(X[tr], y[tr], g[tr], LAMBDAS, n_random=width, random_state=rep)
        h = RidgeHead(lam=lam, n_random=width, random_state=rep).fit(X[tr], y[tr], g[tr])
        test = np.setdiff1d(np.arange(len(y)), tr)
        grp, D = h.lopo_decision_on(X[test])
        oof = h.lopo_decision()
        dec = h.decision_function(X[test])
        sets = {
            "jackknife+_pooled": cp.jackknife_plus(D, grp, oof, y[tr], g[tr], ALPHA, "pooled"),
            "jackknife+_pooled_mondrian": cp.jackknife_plus(D, grp, oof, y[tr], g[tr], ALPHA, "pooled", mondrian=True),
        }
        for dom in np.unique(domain[test]):
            sel = domain[test] == dom
            met = binary_metrics(dec[sel], y[test][sel])
            for name, s in sets.items():
                ev = cp.evaluate_sets(s[sel], y[test][sel], g[test][sel])
                rows.append({"repeat": rep, "domain": dom, "method": name, "n_patients": len(np.unique(g[test][sel])),
                             "auc": met["auc"], "balanced_accuracy": met["balanced_accuracy"], **ev})
        print(f"rep {rep} done", flush=True)
    out = pd.DataFrame(rows)
    out.insert(0, "config", f"{args.features}_{args.head}")
    out.to_csv(ROOT / "results" / "scanner_shift.csv", index=False)
    print(out.groupby(["domain", "method"])[["n_patients", "auc", "coverage_patient", "coverage_class1",
                                             "mean_size"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
