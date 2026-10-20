import numpy as np

from btmri import conformal as cp
from btmri.heads import RidgeHead


def _hierarchical(n_groups, per, rng, d=10, patient_sd=1.2):
    groups = np.repeat(np.arange(n_groups), per)
    eff = rng.normal(0, patient_sd, (n_groups, d))[groups]
    y = rng.integers(0, 2, len(groups))
    X = np.where(y[:, None] == 1, 0.6, -0.6) + eff + rng.normal(0, 1.0, (len(groups), d))
    return X, y, groups


def test_quantile_and_pooled_agree_with_one_slice_per_group():
    rng = np.random.default_rng(0)
    s = rng.normal(size=50)
    g = np.arange(50)
    assert cp.conformal_quantile(s, 0.1) == cp._pooled_threshold(s, g, 0.1)


def test_split_cp_marginal_coverage_iid():
    rng = np.random.default_rng(1)
    covs = []
    for _ in range(200):
        f_cal, f_te = rng.normal(size=200), rng.normal(size=200)
        y_cal, y_te = (f_cal + rng.normal(size=200) > 0).astype(int), (f_te + rng.normal(size=200) > 0).astype(int)
        sets = cp.split_cp(f_cal, y_cal, f_te, 0.1)
        covs.append(sets[np.arange(200), y_te].mean())
    assert abs(np.mean(covs) - 0.9) < 0.015


def test_patient_level_methods_cover_new_patients_slice_split_does_not():
    rng = np.random.default_rng(2)
    res = {"slice": [], "sub": [], "jk": []}
    for rep in range(12):
        X, y, g = _hierarchical(60, 12, rng, patient_sd=2.0)
        Xn, yn, gn = _hierarchical(30, 12, rng, patient_sd=2.0)
        # naive: random slice split, calibration slices share patients with training
        idx = rng.permutation(len(y))
        tr, ca = idx[: int(0.7 * len(y))], idx[int(0.7 * len(y)):]
        h = RidgeHead(lam=0.3, n_random=256, random_state=rep).fit(X[tr], y[tr])
        sets = cp.split_cp(h.decision_function(X[ca]), y[ca], h.decision_function(Xn), 0.1)
        res["slice"].append(cp.evaluate_sets(sets, yn, gn)["coverage_patient"])
        # patient split + one slice per calibration patient
        pts = rng.permutation(60)
        trp = np.isin(g, pts[:42])
        h2 = RidgeHead(lam=0.3, n_random=256, random_state=rep).fit(X[trp], y[trp])
        sets = cp.subsample_once(h2.decision_function(X[~trp]), y[~trp], g[~trp], h2.decision_function(Xn), 0.1,
                                 rng=rep)
        res["sub"].append(cp.evaluate_sets(sets, yn, gn)["coverage_patient"])
        # patient-level jackknife+ with exact LOPO models
        h3 = RidgeHead(lam=0.3, n_random=256, random_state=rep).fit(X, y, g)
        grp, D = h3.lopo_decision_on(Xn)
        sets = cp.jackknife_plus(D, grp, h3.lopo_decision(), y, g, 0.1, mode="pooled")
        res["jk"].append(cp.evaluate_sets(sets, yn, gn)["coverage_patient"])
    m = {k: np.mean(v) for k, v in res.items()}
    assert m["slice"] < 0.85  # leaked calibration under-covers new patients (~0.79)
    assert m["sub"] > 0.88 and m["jk"] > 0.88  # target 0.90


def test_mondrian_controls_each_class():
    rng = np.random.default_rng(3)
    f_cal = np.r_[rng.normal(1, 1, 900), rng.normal(-1, 1, 100)]
    y_cal = np.r_[np.ones(900, int), np.zeros(100, int)]
    f_te = np.r_[rng.normal(1, 1, 9000), rng.normal(-1, 1, 1000)]
    y_te = np.r_[np.ones(9000, int), np.zeros(1000, int)]
    ev = cp.evaluate_sets(cp.split_cp(f_cal, y_cal, f_te, 0.1, mondrian=True), y_te, np.arange(len(y_te)) // 10)
    assert ev["coverage_class0"] > 0.85 and ev["coverage_class1"] > 0.85
