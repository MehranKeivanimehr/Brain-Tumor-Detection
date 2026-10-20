"""Patient-level conformal prediction on unseen patients.

Usage: python scripts/05_conformal.py --features resnet50 --head elm [--repeats 3]

Outer loop: 5-fold patient-level CV; the held-out fold plays the role of new
patients. Inside each training fold, prediction sets are calibrated by:

  naive_slice_split   random 75/25 split of training *slices* (calibration
                      slices share patients with the training slices)
  patient_split       75/25 split of training *patients*, ordinary quantile
                      over all calibration slices
  patient_subsample   same split, one random slice per calibration patient
                      (Dunn et al. 2023; >= 1 - alpha for a new patient)
  patient_pooled_cdf  same split, hierarchical conformal prediction (average of
                      per-patient score CDFs; Lee, Barber & Willett 2026; >= 1 - alpha)
  jackknife+_sub      patient-level jackknife+ from the exact LOPO models
  jackknife+_pooled   same, every slice used, patients weighted equally
  cv+10_pooled        patient-level CV+ with 10 refitted fold models
and '_mondrian' variants calibrate each class separately.

Outputs results/conformal.csv and data/outputs/conformal_sets_<cfg>.npz.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from btmri import conformal as cp
from btmri.heads import RidgeHead, select_lambda_lopo
from btmri.splits import patient_folds, split_patients

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = np.logspace(0, 4, 5)
ALPHAS = (0.05, 0.10, 0.20)


def conformal_fold(X, y, g, tr, te, width, seed, rng, timings):
    """All prediction-set methods for one outer fold. Returns {method: {alpha: sets}}.

    The ridge penalty is chosen once per outer fold by exact leave-one-patient-out
    on the training patients and shared by every method, so the methods differ only
    in how they calibrate.
    """
    Xte = X[te]
    sets = {}
    lam, _ = select_lambda_lopo(X[tr], y[tr], g[tr], LAMBDAS, n_random=width, random_state=seed)

    def make_head(Xa, ya, ga, *_):
        return RidgeHead(lam=lam, n_random=width, random_state=seed).fit(Xa, ya, ga)

    # naive: calibration slices drawn from training patients
    idx = rng.permutation(tr)
    a, b = idx[: int(0.75 * len(idx))], idx[int(0.75 * len(idx)) :]
    h = make_head(X[a], y[a], None)
    dc, dt = h.decision_function(X[b]), h.decision_function(Xte)
    sets["naive_slice_split"] = {al: cp.split_cp(dc, y[b], dt, al) for al in ALPHAS}
    sets["naive_slice_split_mondrian"] = {al: cp.split_cp(dc, y[b], dt, al, mondrian=True) for al in ALPHAS}

    # patient split
    m = split_patients(g[tr], 0.75, rng)
    a, b = tr[m], tr[~m]
    h = make_head(X[a], y[a], None)
    dc, dt = h.decision_function(X[b]), h.decision_function(Xte)
    sets["patient_split"] = {al: cp.split_cp(dc, y[b], dt, al) for al in ALPHAS}
    sets["patient_subsample"] = {al: cp.subsample_once(dc, y[b], g[b], dt, al, rng=rng) for al in ALPHAS}
    sets["patient_pooled_cdf"] = {al: cp.pooled_cdf(dc, y[b], g[b], dt, al) for al in ALPHAS}
    sets["patient_pooled_cdf_mondrian"] = {al: cp.pooled_cdf(dc, y[b], g[b], dt, al, mondrian=True) for al in ALPHAS}

    # patient-level jackknife+ from exact LOPO models: one fit on all training patients
    t0 = time.time()
    h = make_head(X[tr], y[tr], g[tr])
    grp, D = h.lopo_decision_on(Xte)
    oof = h.lopo_decision()
    timings.setdefault("jackknife+_exact_lopo_s", []).append(time.time() - t0)
    for mode in ("sub", "pooled"):
        full = "subsample" if mode == "sub" else "pooled"
        sets[f"jackknife+_{mode}"] = {al: cp.jackknife_plus(D, grp, oof, y[tr], g[tr], al, full, rng=rng)
                                      for al in ALPHAS}
    sets["jackknife+_pooled_mondrian"] = {al: cp.jackknife_plus(D, grp, oof, y[tr], g[tr], al, "pooled",
                                                                mondrian=True) for al in ALPHAS}

    # patient-level CV+ with 10 refitted models (the generic alternative)
    t0 = time.time()
    inner = StratifiedGroupKFold(10, shuffle=True, random_state=seed).split(X[tr], y[tr], g[tr])
    oof10 = np.zeros(len(tr))
    pts = np.unique(g[tr])
    D10 = np.zeros((len(te), len(pts)))
    for a_, b_ in inner:
        hk = make_head(X[tr][a_], y[tr][a_], None)
        oof10[b_] = hk.decision_function(X[tr][b_])
        dk = hk.decision_function(Xte)
        for p in np.unique(g[tr][b_]):
            D10[:, np.searchsorted(pts, p)] = dk
    timings.setdefault("cv+10_refits_s", []).append(time.time() - t0)
    sets["cv+10_pooled"] = {al: cp.jackknife_plus(D10, pts, oof10, y[tr], g[tr], al, "pooled") for al in ALPHAS}
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="resnet50")
    ap.add_argument("--head", default="elm")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    t = pd.read_csv(ROOT / "data" / "dataset" / "slices.csv")
    keep = t.label.values >= 0
    t = t[keep].reset_index(drop=True)
    X = np.load(ROOT / "data" / "features" / f"{args.features}.npy")[np.flatnonzero(keep)].astype(np.float64)
    y, g = t.label.values.astype(int), t.patient.values
    width = 2048 if args.head == "elm" else 0
    cfg = f"{args.features}_{args.head}"

    res_path = ROOT / "results" / "conformal.csv"
    rows, timings, example = [], {}, None
    if args.repeats < 1:
        return
    for rep in range(args.repeats):
        rng = np.random.default_rng(1000 + rep)
        for k, (tr, te) in enumerate(patient_folds(y, g, 5, seed=rep)):
            t0 = time.time()
            sets = conformal_fold(X, y, g, tr, te, width, rep, rng, timings)
            for method, by_alpha in sets.items():
                for al, s in by_alpha.items():
                    rows.append({"config": cfg, "repeat": rep, "fold": k, "method": method, "alpha": al,
                                 **cp.evaluate_sets(s, y[te], g[te])})
            if rep == 0 and k == 0:
                example = {"rows": te, **{f"{mth}|{al}": s for mth, d in sets.items() for al, s in d.items()}}
            print(f"{cfg} rep{rep} fold{k}: {time.time() - t0:.0f}s", flush=True)
    new = pd.DataFrame(rows)
    (ROOT / "results" / "conformal_runs").mkdir(exist_ok=True)
    new.to_csv(ROOT / "results" / "conformal_runs" / f"{cfg}.csv", index=False)
    # re-read just before writing so concurrent runs of other configs are kept
    old = pd.read_csv(res_path) if res_path.exists() else pd.DataFrame(columns=new.columns)
    pd.concat([old[old.config != cfg], new], ignore_index=True).to_csv(res_path, index=False)
    (ROOT / "data" / "outputs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "data" / "outputs" / f"conformal_sets_{cfg}.npz", **example)
    tim = {k: float(np.mean(v)) for k, v in timings.items()}
    pd.Series(tim).to_json(ROOT / "results" / f"conformal_timing_{cfg}.json")
    summ = pd.DataFrame(rows).groupby(["method", "alpha"])[["coverage_patient", "coverage_class1", "mean_size"]].mean()
    print(summ.round(3).to_string())
    print(tim)


if __name__ == "__main__":
    main()
