r"""Conformal prediction sets for binary slice classification with patient structure.

Nonconformity score for a decision value f (larger = class 1):

    s(x, y) = -(2y - 1) * f(x)

Slices from the same patient are not exchangeable with slices from a new
patient, so the exchangeable unit here is the *patient* (Dunn, Wasserman &
Ramdas 2023; Lee, Barber & Willett 2023). Methods:

* ``split_cp``: classical split conformal on calibration slices
  (Vovk et al. 2005; Lei et al. 2018). Valid for new patients only if the
  calibration slices come from patients unseen in training.
* ``subsample_once``: one uniformly drawn calibration slice per calibration
  patient, then split conformal ("subsampling once", Dunn et al. 2023).
  Exactly valid (>= 1 - alpha) for a random slice of a new patient when
  patients are exchangeable.
* ``pooled_cdf``: average of the per-patient empirical CDFs of the scores with
  the (1 + 1/m) correction at the patient level. This is the hierarchical
  conformal prediction of Lee, Barber & Willett (2026, eq. 6 / Thm 1): coverage
  >= 1 - alpha (and <= 1 - alpha + 2/(m+1) without ties) for a random slice of a
  new patient, when patients are exchangeable and slices are exchangeable
  within a patient. The guarantee averages over patients; it does not promise
  1 - alpha for every individual patient.
* ``jackknife_plus``: patient-level jackknife+ / CV+ (Barber et al. 2021;
  Romano, Sesia & Candes 2020) using leave-one-patient-out models. In
  ``pooled`` mode every slice of patient k gets weight 1/N_k: this is the
  classification-score form of the hierarchical jackknife+ of Lee, Barber &
  Willett (2026, eq. 20, Thm 5: >= 1 - 2 alpha for a hierarchically symmetric
  algorithm). ``subsample`` mode keeps one random slice per patient (plain
  jackknife+ over patients, also >= 1 - 2 alpha). With the closed-form heads in
  :mod:`btmri.heads` the LOPO models are exact and cost one fit.

``mondrian=True`` calibrates each class separately (class-conditional
coverage; for the tumor class this is a sensitivity guarantee).
Prediction sets are boolean arrays of shape (n, 2): [contains 0, contains 1].
"""

from __future__ import annotations

import math

import numpy as np


def scores(decision: np.ndarray, y) -> np.ndarray:
    return -(2 * np.asarray(y) - 1) * np.asarray(decision)


def _test_scores(decision: np.ndarray) -> np.ndarray:
    d = np.asarray(decision)
    return np.stack([d, -d], axis=-1)  # s(x, 0) = f, s(x, 1) = -f


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    n = len(cal_scores)
    k = math.ceil((1 - alpha) * (n + 1))
    if n == 0 or k > n:
        return np.inf
    return float(np.sort(cal_scores)[k - 1])


def split_cp(dec_cal, y_cal, dec_test, alpha, mondrian=False):
    s_cal = scores(dec_cal, y_cal)
    S = _test_scores(dec_test)
    y_cal = np.asarray(y_cal)
    if mondrian:
        q = np.array([conformal_quantile(s_cal[y_cal == c], alpha) for c in (0, 1)])
    else:
        q = np.full(2, conformal_quantile(s_cal, alpha))
    return S <= q[None, :]


def subsample_once(dec_cal, y_cal, g_cal, dec_test, alpha, mondrian=False, rng=None):
    rng = np.random.default_rng(rng)
    dec_cal, y_cal, g_cal = map(np.asarray, (dec_cal, y_cal, g_cal))
    S = _test_scores(dec_test)
    q = np.empty(2)
    for c in (0, 1):
        pick = []
        for g in np.unique(g_cal):
            idx = np.flatnonzero((g_cal == g) & ((y_cal == c) if mondrian else True))
            if len(idx):
                pick.append(rng.choice(idx))
        pick = np.asarray(pick, dtype=int)
        q[c] = conformal_quantile(scores(dec_cal[pick], y_cal[pick]), alpha)
        if not mondrian:
            q[:] = q[c]
            break
    return S <= q[None, :]


def _pooled_threshold(s, g, alpha):
    """inf{t : (1/m) sum_g Fhat_g(t) >= (1 - alpha)(1 + 1/m)}."""
    groups, inv = np.unique(g, return_inverse=True)
    m = len(groups)
    if m == 0:
        return np.inf
    target = (1 - alpha) * (1 + 1 / m)
    if target > 1:
        return np.inf
    w = 1.0 / np.bincount(inv)[inv] / m
    order = np.argsort(s, kind="stable")
    cdf = np.cumsum(w[order])
    k = np.searchsorted(cdf, target - 1e-12, side="left")
    return float(s[order][min(k, len(s) - 1)])


def pooled_cdf(dec_cal, y_cal, g_cal, dec_test, alpha, mondrian=False):
    dec_cal, y_cal, g_cal = map(np.asarray, (dec_cal, y_cal, g_cal))
    s = scores(dec_cal, y_cal)
    S = _test_scores(dec_test)
    if mondrian:
        q = np.array([_pooled_threshold(s[y_cal == c], g_cal[y_cal == c], alpha) for c in (0, 1)])
    else:
        q = np.full(2, _pooled_threshold(s, g_cal, alpha))
    return S <= q[None, :]


def jackknife_plus(
    dec_test_per_model: np.ndarray,
    model_groups: np.ndarray,
    dec_cal_oof: np.ndarray,
    y_cal: np.ndarray,
    g_cal: np.ndarray,
    alpha: float,
    mode: str = "subsample",
    mondrian: bool = False,
    rng=None,
) -> np.ndarray:
    """Patient-level jackknife+ / CV+ prediction sets.

    dec_test_per_model : (n_test, m) decision of the model that excluded
        training patient ``model_groups[i]`` (LOPO: one model per patient;
        CV+: column i holds the decision of the fold model that excluded i).
    dec_cal_oof : out-of-patient decisions for every training slice.
    y_cal, g_cal : labels and patient ids of the training slices.
    mode : 'subsample' (one random slice per patient; >= 1 - 2 alpha) or
        'pooled' (all slices, each patient weighted equally).
    """
    rng = np.random.default_rng(rng)
    dec_cal_oof, y_cal, g_cal = map(np.asarray, (dec_cal_oof, y_cal, g_cal))
    s_cal = scores(dec_cal_oof, y_cal)
    n_test, m = dec_test_per_model.shape
    sets = np.zeros((n_test, 2), dtype=bool)
    for c in (0, 1):
        s_test = -(2 * c - 1) * dec_test_per_model  # (n_test, m): s(x, c; mu_{-i})
        counts = np.zeros(n_test)
        m_eff = 0
        for i, g in enumerate(model_groups):
            sel = (g_cal == g) & ((y_cal == c) if mondrian else True)
            si = s_cal[sel]
            if si.size == 0:
                continue
            m_eff += 1
            if mode == "subsample":
                counts += s_test[:, i] > rng.choice(si)
            else:
                si = np.sort(si)
                counts += np.searchsorted(si, s_test[:, i], side="left") / si.size
        sets[:, c] = counts < (1 - alpha) * (m_eff + 1)
    return sets


def evaluate_sets(sets: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    y, groups = np.asarray(y), np.asarray(groups)
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    per_patient = np.array([covered[groups == g].mean() for g in np.unique(groups)])
    out = {
        "coverage_slice": float(covered.mean()),
        "coverage_patient": float(per_patient.mean()),
        "coverage_patient_p10": float(np.quantile(per_patient, 0.10)),
        "coverage_class0": float(covered[y == 0].mean()),
        "coverage_class1": float(covered[y == 1].mean()),
        "mean_size": float(size.mean()),
        "frac_singleton": float((size == 1).mean()),
        "frac_both": float((size == 2).mean()),
        "frac_empty": float((size == 0).mean()),
    }
    single = size == 1
    if single.any():
        out["singleton_accuracy"] = float(covered[single].mean())
    return out
