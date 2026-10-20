"""Benchmark: feature sets x heads, evaluated under a leaky (slice-level) and an
honest (patient-level) cross-validation protocol.

Usage: python scripts/04_benchmark.py [--features ...] [--heads ...] [--seeds 0]

Hyper-parameters are chosen inside each training fold only:
* ridge / ELM: penalty by exact leave-one-out on the training fold (leave-one-
  slice-out for the slice protocol, leave-one-patient-out for the patient
  protocol: each protocol tunes the way a practitioner following it would);
* logistic regression: C by an inner 3-fold split of the same kind;
* kNN: fixed k = 15.
Ridge/ELM probabilities are Platt-calibrated on the same out-of-sample
training-fold decisions.

Outputs: data/outputs/benchmark_oof.npz (out-of-fold decisions) and
results/benchmark.csv (metrics with patient-level bootstrap CIs).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsClassifier

from btmri.heads import RidgeHead, Standardizer, select_lambda_lopo
from btmri.metrics import binary_metrics, cluster_bootstrap, platt
from btmri.splits import patient_folds, slice_folds

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = np.logspace(0, 4, 5)
ELM_WIDTH = 2048


def load():
    t = pd.read_csv(ROOT / "data" / "dataset" / "slices.csv")
    keep = t.label.values >= 0
    return t[keep].reset_index(drop=True), np.flatnonzero(keep)


def fit_predict(head, Xtr, ytr, gtr, Xte, protocol, seed):
    """Return (test decisions, test probabilities or None, info)."""
    info = {}
    if head in ("ridge", "elm"):
        width = ELM_WIDTH if head == "elm" else 0
        tune_groups = gtr if protocol == "patient" else np.arange(len(ytr))
        lam, curve = select_lambda_lopo(Xtr, ytr, tune_groups, LAMBDAS, n_random=width, random_state=seed)
        h = RidgeHead(lam=lam, n_random=width, random_state=seed).fit(Xtr, ytr, tune_groups)
        cal = platt(h.lopo_decision(), ytr)
        dec = h.decision_function(Xte)
        info.update(lam=lam, tune_auc=float(curve.max()))
        return dec, cal(dec), info
    std = Standardizer().fit(Xtr)
    Ztr, Zte = std.transform(Xtr), std.transform(Xte)
    if head == "knn":
        m = KNeighborsClassifier(n_neighbors=15, n_jobs=4).fit(Ztr, ytr)
        p = m.predict_proba(Zte)[:, 1]
        return p - 0.5, p, info
    if head == "logistic":
        inner = (GroupKFold(3).split(Ztr, ytr, gtr) if protocol == "patient"
                 else KFold(3, shuffle=True, random_state=seed).split(Ztr, ytr))
        inner = list(inner)
        best, best_c = -np.inf, None
        for C in (0.001, 0.01, 0.1):
            accs = []
            for a, b in inner:
                lr = LogisticRegression(C=C, max_iter=3000).fit(Ztr[a], ytr[a])
                accs.append(binary_metrics(lr.decision_function(Ztr[b]), ytr[b])["auc"])
            if np.mean(accs) > best:
                best, best_c = np.mean(accs), C
        lr = LogisticRegression(C=best_c, max_iter=3000).fit(Ztr, ytr)
        info.update(C=best_c)
        return lr.decision_function(Zte), lr.predict_proba(Zte)[:, 1], info
    raise ValueError(head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="classical,scattering,resnet50,densenet121")
    ap.add_argument("--heads", default="ridge,elm,logistic,knn")
    ap.add_argument("--protocols", default="slice,patient")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--n_boot", type=int, default=500)
    args = ap.parse_args()

    table, rows = load()
    y, g = table.label.values.astype(int), table.patient.values
    out_dir = ROOT / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_path = out_dir / "benchmark_oof.npz"
    oof = dict(np.load(oof_path)) if oof_path.exists() else {}
    res_path = ROOT / "results" / "benchmark.csv"
    results = pd.read_csv(res_path).to_dict("records") if res_path.exists() else []
    done = {(r["features"], r["head"], r["protocol"], r["seed"]) for r in results}

    for feat in args.features.split(","):
        X = np.load(ROOT / "data" / "features" / f"{feat}.npy")[rows].astype(np.float64)
        for seed in map(int, args.seeds.split(",")):
            folds = {"patient": patient_folds(y, g, 5, seed), "slice": slice_folds(y, 5, seed)}
            for protocol in args.protocols.split(","):
                for head in args.heads.split(","):
                    if (feat, head, protocol, seed) in done:
                        continue
                    t0 = time.time()
                    dec, prob, infos = np.zeros(len(y)), np.zeros(len(y)), []
                    for tr, te in folds[protocol]:
                        d, p, info = fit_predict(head, X[tr], y[tr], g[tr], X[te], protocol, seed)
                        dec[te], prob[te] = d, p
                        infos.append(info)
                    m = binary_metrics(dec, y, prob)
                    lo, hi = cluster_bootstrap(lambda d, t: binary_metrics(d, t)["auc"], g, args.n_boot, d=dec, t=y)
                    blo, bhi = cluster_bootstrap(lambda d, t: binary_metrics(d, t)["balanced_accuracy"], g,
                                                 args.n_boot, d=dec, t=y)
                    rec = {"features": feat, "head": head, "protocol": protocol, "seed": seed,
                           **m, "auc_lo": lo, "auc_hi": hi, "bacc_lo": blo, "bacc_hi": bhi,
                           "n_slices": len(y), "n_patients": len(np.unique(g)),
                           "fold_info": json.dumps(infos), "seconds": round(time.time() - t0, 1)}
                    results.append(rec)
                    oof[f"{feat}|{head}|{protocol}|{seed}|dec"] = dec.astype(np.float32)
                    oof[f"{feat}|{head}|{protocol}|{seed}|prob"] = prob.astype(np.float32)
                    pd.DataFrame(results).to_csv(res_path, index=False)
                    np.savez_compressed(oof_path, **oof)
                    print(f"{feat:12s} {head:9s} {protocol:8s} seed{seed}: AUC {m['auc']:.3f} [{lo:.3f},{hi:.3f}] "
                          f"bAcc {m['balanced_accuracy']:.3f} ECE {m['ece']:.3f} ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
