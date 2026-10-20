"""Metrics with patient-level uncertainty.

Slices of a patient are correlated, so confidence intervals resample patients
(cluster bootstrap), never slices.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def platt(dec_fit, y_fit):
    """1D logistic calibration map fitted on out-of-patient decision values."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(dec_fit)[:, None], y_fit)
    return lambda d: lr.predict_proba(np.asarray(d)[:, None])[:, 1]


def ece(prob, y, n_bins=15):
    """Expected calibration error with equal-mass bins."""
    prob, y = np.asarray(prob), np.asarray(y)
    order = np.argsort(prob)
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) * abs(prob[b].mean() - y[b].mean()) for b in bins if len(b)) / len(y))


def binary_metrics(dec, y, prob=None, threshold=0.0):
    dec, y = np.asarray(dec), np.asarray(y)
    pred = (dec >= threshold).astype(int)
    tp, tn = np.sum((pred == 1) & (y == 1)), np.sum((pred == 0) & (y == 0))
    sens, spec = tp / max((y == 1).sum(), 1), tn / max((y == 0).sum(), 1)
    out = {
        "auc": float(roc_auc_score(y, dec)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float((sens + spec) / 2),
        "sensitivity": float(sens),
        "specificity": float(spec),
    }
    if prob is not None:
        prob = np.asarray(prob)
        out["brier"] = float(np.mean((prob - y) ** 2))
        out["ece"] = ece(prob, y)
    return out


def cluster_bootstrap(fn, groups, n_boot=1000, seed=0, **arrays):
    """Percentile 95% CI of ``fn(**arrays)`` resampling patients with replacement."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[k] for k in pick])
        try:
            vals.append(fn(**{k: np.asarray(v)[idx] for k, v in arrays.items()}))
        except ValueError:  # e.g. a resample with one class only
            continue
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
