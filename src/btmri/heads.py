r"""Closed-form classification heads with exact leave-one-patient-out (LOPO) identities.

A head is generalized ridge regression on a fixed feature map:

    phi(x) = [1, x]                         (linear ridge)
    phi(x) = [1, g(W x + b)],  W, b random  (extreme learning machine /
                                              random-feature model)

with targets T in {-1, +1}^{n x k} and an unpenalised intercept:

    A = Phi^T Phi + diag(0, lam, ..., lam),   B = A^{-1} Phi^T T.

Removing a whole patient (group G of rows) has an exact closed form
(Cook & Weisberg 1982, subset deletion; Woodbury identity):

    H_GG  = Phi_G A^{-1} Phi_G^T
    E_G   = T_G - Phi_G B
    B_{-G} = B - A^{-1} Phi_G^T (I - H_GG)^{-1} E_G
    T_G - Phi_G B_{-G} = (I - H_GG)^{-1} E_G

so out-of-patient predictions for every training patient, and the model
trained without each patient, cost one factorisation plus one small
(n_G x n_G) solve per patient instead of one refit per patient.

Input standardisation is estimated once on the training set and then held
fixed; the identities are exact for that fixed transform (the effect of
re-estimating it per patient is quantified in the tests and is negligible).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class Standardizer:
    def fit(self, X):
        self.mean_ = X.mean(0)
        self.scale_ = X.std(0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_


class RidgeHead:
    """Ridge (``n_random=0``) or extreme-learning-machine head (``n_random>0``).

    Parameters
    ----------
    lam : ridge penalty on all non-intercept weights.
    n_random : width of the random hidden layer (0 = linear ridge on inputs).
    activation : 'tanh' or 'relu' for the random layer.
    random_state : seed for the random layer.
    """

    def __init__(self, lam: float = 1.0, n_random: int = 0, activation: str = "tanh", random_state: int = 0):
        self.lam = float(lam)
        self.n_random = int(n_random)
        self.activation = activation
        self.random_state = random_state

    # ---- feature map -------------------------------------------------
    def _init_map(self, d: int):
        if self.n_random > 0:
            rng = np.random.default_rng(self.random_state)
            # pre-activations ~ N(0, 1) for standardised inputs
            self.W_ = rng.standard_normal((d, self.n_random)) / np.sqrt(d)
            self.b_ = rng.uniform(-1, 1, self.n_random)

    def _phi(self, Z):
        if self.n_random > 0:
            H = Z @ self.W_ + self.b_
            H = np.tanh(H) if self.activation == "tanh" else np.maximum(H, 0.0)
        else:
            H = Z
        return np.hstack([np.ones((H.shape[0], 1)), H])

    def features(self, X):
        return self._phi(self.std_.transform(np.asarray(X, dtype=np.float64)))

    # ---- fitting ---------------------------------------------------------
    @staticmethod
    def _targets(y):
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) == 2:
            T = np.where(y == classes[1], 1.0, -1.0)[:, None]
        else:
            T = np.where(y[:, None] == classes[None, :], 1.0, -1.0)
        return classes, T

    def fit(self, X, y, groups=None, standardizer: Standardizer | None = None):
        """Fit the head. ``standardizer`` (optional) fixes the input transform."""
        X = np.asarray(X, dtype=np.float64)
        self.std_ = standardizer if standardizer is not None else Standardizer().fit(X)
        self._init_map(X.shape[1])
        Phi = self.features(X)
        self.classes_, T = self._targets(y)
        self._Phi, self._T = Phi, T
        self._G, self._PtT = Phi.T @ Phi, Phi.T @ T  # reused across penalties
        self._groups = None if groups is None else np.asarray(groups)
        self._solve()
        return self

    def _solve(self):
        reg = np.full(self._G.shape[0], self.lam)
        reg[0] = 0.0
        self.cho_ = cho_factor(self._G + np.diag(reg), lower=False, check_finite=False)
        self.B_ = cho_solve(self.cho_, self._PtT, check_finite=False)
        self._lopo = None

    def set_lambda(self, lam: float):
        """Change the penalty without recomputing the feature map or Gram matrix."""
        self.lam = float(lam)
        self._solve()
        return self

    def decision_function(self, X):
        D = self.features(X) @ self.B_
        return D[:, 0] if D.shape[1] == 1 else D

    def predict(self, X):
        D = self.decision_function(X)
        if D.ndim == 1:
            return np.where(D >= 0, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(D, 1)]

    # ---- exact leave-one-group-out ---------------------------------------
    def _compute_lopo(self, keep_models: bool):
        if self._groups is None:
            raise ValueError("fit(..., groups=...) is required for LOPO quantities")
        Phi, T, B = self._Phi, self._T, self.B_
        E = T - Phi @ B
        out = np.empty_like(T)
        uniq, inv = np.unique(self._groups, return_inverse=True)
        # one BLAS-3 product gives (A^{-1} phi_i)^T for every row; each patient then
        # needs only n_G x n_G work instead of triangular solves against A
        C = cho_solve(self.cho_, np.eye(Phi.shape[1]), check_finite=False)
        PC = Phi @ C
        if len(uniq) == len(self._groups) and not keep_models:
            # all groups are single rows: classical vectorised leave-one-out (PRESS)
            h = np.einsum("np,np->n", Phi, PC)
            out = T - E / (1.0 - h)[:, None]
            self._lopo = (uniq, out, None)
            return
        rows = np.argsort(inv, kind="stable")
        bounds = np.r_[0, np.cumsum(np.bincount(inv))]
        dB = np.empty((len(uniq), Phi.shape[1], T.shape[1])) if keep_models else None
        for j in range(len(uniq)):
            idx = rows[bounds[j] : bounds[j + 1]]
            PCg = PC[idx]  # rows of Phi_G A^{-1}
            H = PCg @ Phi[idx].T  # H_GG
            X = np.linalg.solve(np.eye(len(idx)) - H, E[idx])  # LOPO residuals
            out[idx] = T[idx] - X
            if keep_models:
                dB[j] = -PCg.T @ X
        self._lopo = (uniq, out, dB)

    def lopo_decision(self):
        """Out-of-patient decision values for every training row (exact)."""
        if self._lopo is None:
            self._compute_lopo(keep_models=False)
        D = self._lopo[1]
        return D[:, 0] if D.shape[1] == 1 else D

    def lopo_decision_on(self, X):
        """Decision values on new rows under each leave-one-patient-out model.

        Returns ``(groups, D)`` with ``D`` of shape (n_rows, n_groups) for binary
        problems (or (n_rows, n_groups, k)).
        """
        if self._lopo is None or self._lopo[2] is None:
            self._compute_lopo(keep_models=True)
        uniq, _, dB = self._lopo
        Phi = self.features(X)
        base = Phi @ self.B_  # (n, k)
        delta = np.tensordot(Phi, dB, axes=([1], [1]))  # (n, groups, k), BLAS-backed
        D = base[:, None, :] + delta
        return uniq, (D[..., 0] if D.shape[2] == 1 else D)


def brute_force_lopo(head_factory, X, y, groups, refit_standardizer=False):
    """Reference implementation: refit the head once per left-out group.

    With ``refit_standardizer=False`` the input transform is the one estimated
    on all training rows (the setting in which the closed form is exact); with
    ``True`` it is re-estimated without the left-out group.
    """
    X, y, groups = np.asarray(X, dtype=np.float64), np.asarray(y), np.asarray(groups)
    fixed = Standardizer().fit(X)
    out = np.empty(len(y))
    for g in np.unique(groups):
        tr = groups != g
        h = head_factory().fit(X[tr], y[tr], standardizer=None if refit_standardizer else fixed)
        out[~tr] = h.decision_function(X[~tr])
    return out


def select_lambda_lopo(X, y, groups, lambdas, n_random=0, random_state=0, score="auc"):
    """Pick the ridge penalty by exact leave-one-patient-out AUC (or accuracy)."""
    from sklearn.metrics import roc_auc_score

    res = []
    h = RidgeHead(lam=lambdas[0], n_random=n_random, random_state=random_state).fit(X, y, groups)
    for lam in lambdas:
        d = h.set_lambda(lam).lopo_decision()
        if score == "auc":
            s = roc_auc_score(y, d)
        else:
            s = np.mean((d >= 0) == (np.asarray(y) == h.classes_[1]))
        res.append(s)
    res = np.asarray(res)
    best = int(np.flatnonzero(res >= res.max() - 1e-12)[-1])  # ties -> stronger penalty
    return float(lambdas[best]), res
