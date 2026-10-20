"""Cross-validation splits. The patient-level splitter is the default everywhere;
the slice-level splitter exists only to measure how much it leaks."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def patient_folds(y, groups, n_splits=5, seed=0):
    """Stratified K-fold over patients: no patient appears in two folds."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros(len(y)), y, groups))
    for tr, te in folds:
        assert not set(np.unique(np.asarray(groups)[tr])) & set(np.unique(np.asarray(groups)[te]))
    return folds


def slice_folds(y, n_splits=5, seed=0):
    """Stratified K-fold over slices, ignoring patients (the leaky protocol)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def split_patients(groups, frac, rng):
    """Random subset of patients holding ~``frac`` of them; returns a row mask."""
    pts = np.unique(groups)
    pick = rng.choice(pts, size=int(round(frac * len(pts))), replace=False)
    return np.isin(groups, pick)
