"""Chapter zero: audit of the original MATLAB pipeline from its saved features.

``legacy/data/Trainsetnew0.5.mat`` (250 x 69) and ``Testsetnew0.5.mat``
(21 x 69) hold the features computed by ``legacy/matlab/main.m`` on the
Kaggle "Brain MRI Images for Brain Tumor Detection" slices (columns 1-10:
DWT/GLCM statistics of a Gabor+GMM mask; columns 11-69: 59-bin LBP of the
median-filtered slice). The LBP block is a deterministic function of the
image, so identical LBP vectors identify identical images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURE_NAMES_10 = ["contrast", "energy", "homogeneity", "mean", "std", "entropy", "rms", "variance",
                    "smoothness", "idm"]


def load(legacy_dir: Path):
    tr = sio.loadmat(legacy_dir / "Trainsetnew0.5.mat")
    te = sio.loadmat(legacy_dir / "Testsetnew0.5.mat")
    X, Xt = tr["data"].astype(float), te["Testdata"].astype(float)
    y = np.array([0] * 95 + [1] * 155)  # main.m: 95 'no' then 155 'yes'
    yt = np.array([0] * 8 + [1] * 13)
    names = [f"No{i + 1}" for i in range(95)] + [f"Y{i + 1}" for i in range(155)]
    tnames = [f"No{i + 1}" for i in range(8)] + [f"Y{i + 1}" for i in range(13)]
    return X, y, names, Xt, yt, tnames


def duplicate_groups(lbp: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Union-find over rows with identical LBP vectors -> group id per row."""
    n = len(lbp)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = np.abs(lbp[i + 1 :] - lbp[i]).max(1) if i + 1 < n else np.array([])
        for j in np.flatnonzero(d < tol) + i + 1:
            parent[find(j)] = find(i)
    return np.array([find(i) for i in range(n)])


def elm_matlab(Xtr, ytr, Xte, n_hidden=160, rng=None):
    """Faithful port of legacy/matlab/elm.m (classifier, 'sine' activation):
    W ~ U(-1, 1), b ~ U(0, 1), beta = pinv(H') T', no input scaling."""
    T = np.where(np.eye(2)[ytr] == 1, 1.0, -1.0)
    W = rng.random((n_hidden, Xtr.shape[1])) * 2 - 1
    b = rng.random((n_hidden, 1))
    H = np.sin(W @ Xtr.T + b)
    beta = np.linalg.pinv(H.T) @ T
    return (H.T @ beta).argmax(1), (np.sin(W @ Xte.T + b).T @ beta).argmax(1)


def run(legacy_dir: Path, n_seeds: int = 2000, seed: int = 0) -> dict:
    X, y, names, Xt, yt, tnames = load(legacy_dir)
    lbp, lbpt = X[:, 10:], Xt[:, 10:]
    out: dict = {}

    # 1. duplicates
    test_to_train = {}
    for j in range(len(Xt)):
        d = np.abs(lbp - lbpt[j]).max(1)
        k = int(np.argmin(d))
        if d[k] < 1e-9:
            test_to_train[tnames[j]] = names[k]
    within_test = [(tnames[i], tnames[j]) for i in range(len(Xt)) for j in range(i + 1, len(Xt))
                   if np.abs(lbpt[i] - lbpt[j]).max() < 1e-9]
    groups = duplicate_groups(lbp)
    within_train_pairs = sum(int(c * (c - 1) / 2) for c in np.bincount(groups) if c > 1)
    unseen = [t for t in tnames if t not in test_to_train]
    out["duplicates"] = {
        "test_images": len(Xt),
        "test_images_copied_from_train": len(test_to_train),
        "test_to_train": test_to_train,
        "identical_pairs_within_test": within_test,
        "distinct_tumor_test_images": int(13 - sum(1 for a, b in within_test if a.startswith("Y"))),
        "identical_pairs_within_train": within_train_pairs,
        "unique_train_images": int(len(np.unique(groups))),
        "unseen_test_images": unseen,
    }

    # 2. degenerate features
    out["features"] = {
        "smoothness_range": [float(X[:, 8].min()), float(X[:, 8].max())],
        "idm_range": [float(X[:, 9].min()), float(X[:, 9].max())],
        "all_foreground_masks_train": int(np.sum(np.isclose(X[:, 3], 2.0) & np.isclose(X[:, 7], 0.0))),
        "all_foreground_masks_test": int(np.sum(np.isclose(Xt[:, 3], 2.0) & np.isclose(Xt[:, 7], 0.0))),
    }

    # 3. the original protocol: one random ELM vs best-of-10 selected on test labels
    rng = np.random.default_rng(seed)
    unseen_idx = np.array([tnames.index(t) for t in unseen])
    acc, acc_unseen, acc_train = [], [], []
    for _ in range(n_seeds):
        ptr, pte = elm_matlab(X, y, Xt, rng=rng)
        acc.append((pte == yt).mean())
        acc_unseen.append((pte[unseen_idx] == yt[unseen_idx]).mean())
        acc_train.append((ptr == y).mean())
    acc = np.asarray(acc)
    best10 = acc.reshape(-1, 10).max(1)
    out["original_protocol"] = {
        "majority_baseline_test": float((yt == 1).mean()),
        "elm_single_run_test_mean": float(acc.mean()),
        "elm_single_run_test_sd": float(acc.std()),
        "elm_single_run_test_min_max": [float(acc.min()), float(acc.max())],
        "elm_single_run_train_mean": float(np.mean(acc_train)),
        "elm_single_run_unseen3_mean": float(np.mean(acc_unseen)),
        "best_of_10_on_test_mean": float(best10.mean()),
        "best_of_10_on_test_max": float(best10.max()),
        "single_run_test_accuracies": acc.tolist(),
        "best_of_10_test_accuracies": best10.tolist(),
    }

    # 4. honest re-evaluation: 5-fold CV grouped by duplicate image, 20 repeats
    def grouped_cv(make, cols=slice(None), reps=20):
        accs = []
        for r in range(reps):
            cv = StratifiedGroupKFold(5, shuffle=True, random_state=r)
            for tr, te in cv.split(X, y, groups):
                m = make(r).fit(X[tr][:, cols], y[tr])
                accs.append((m.predict(X[te][:, cols]) == y[te]).mean())
        return float(np.mean(accs)), float(np.std(accs))

    class _ELM:
        def __init__(self, r, n_hidden=160, drop_idm=False):
            self.rng, self.n_hidden, self.drop = np.random.default_rng(r), n_hidden, drop_idm

        def fit(self, A, b):
            self.A, self.b = A, b
            return self

        def predict(self, B):
            keep = [c for c in range(self.A.shape[1]) if not (self.drop and c == 9)]
            return elm_matlab(self.A[:, keep], self.b, B[:, keep], self.n_hidden, self.rng)[1]

    logreg = lambda r: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))  # noqa: E731
    svm = lambda r: make_pipeline(StandardScaler(), SVC())  # noqa: E731
    out["grouped_cv"] = {
        "elm_as_configured": grouped_cv(lambda r: _ELM(r)),
        "elm_without_idm_20_hidden": grouped_cv(lambda r: _ELM(r, 20, drop_idm=True)),
        "logistic_all_features": grouped_cv(logreg),
        "rbf_svm_all_features": grouped_cv(svm),
        "logistic_lbp_only": grouped_cv(logreg, slice(10, 69)),
        "logistic_segmentation_features_only": grouped_cv(logreg, slice(0, 10)),
        "majority_baseline": float((y == 1).mean()),
        "n_unique_images": int(len(np.unique(groups))),
    }
    return out
