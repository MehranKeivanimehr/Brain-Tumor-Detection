import numpy as np
import pytest

from btmri.heads import RidgeHead, Standardizer, brute_force_lopo, select_lambda_lopo


def _grouped_data(n_groups=25, per=8, d=12, seed=0, n_classes=2):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    offset = rng.normal(0, 1.0, (n_groups, d))[groups]  # patient effect
    y = rng.integers(0, n_classes, len(groups))
    mu = rng.normal(0, 1.5, (n_classes, d))
    X = mu[y] + offset + rng.normal(0, 1.0, (len(groups), d))
    return X, y, groups


@pytest.mark.parametrize("n_random", [0, 64])
def test_lopo_matches_brute_force(n_random):
    X, y, g = _grouped_data()
    head = RidgeHead(lam=3.0, n_random=n_random, random_state=7).fit(X, y, g)
    fast = head.lopo_decision()
    slow = brute_force_lopo(lambda: RidgeHead(lam=3.0, n_random=n_random, random_state=7), X, y, g)
    np.testing.assert_allclose(fast, slow, rtol=1e-8, atol=1e-8)


def test_lopo_models_on_new_points_match_refits():
    X, y, g = _grouped_data(seed=1)
    Xt, _, _ = _grouped_data(n_groups=3, seed=2)
    head = RidgeHead(lam=1.0, n_random=32, random_state=3).fit(X, y, g)
    groups, D = head.lopo_decision_on(Xt)
    fixed = Standardizer().fit(X)
    for j, grp in enumerate(groups):
        tr = g != grp
        ref = RidgeHead(lam=1.0, n_random=32, random_state=3).fit(X[tr], y[tr], standardizer=fixed)
        np.testing.assert_allclose(D[:, j], ref.decision_function(Xt), rtol=1e-8, atol=1e-8)


def test_lopo_multiclass():
    X, y, g = _grouped_data(n_classes=3, seed=4)
    head = RidgeHead(lam=2.0).fit(X, y, g)
    fast = head.lopo_decision()
    fixed = Standardizer().fit(X)
    for grp in np.unique(g):
        tr = g != grp
        ref = RidgeHead(lam=2.0).fit(X[tr], y[tr], standardizer=fixed)
        np.testing.assert_allclose(fast[g == grp], ref.decision_function(X[g == grp]), atol=1e-8)


def test_fixed_standardizer_is_a_small_approximation():
    X, y, g = _grouped_data(n_groups=60, seed=5)
    exact = brute_force_lopo(lambda: RidgeHead(lam=3.0), X, y, g, refit_standardizer=True)
    fast = RidgeHead(lam=3.0).fit(X, y, g).lopo_decision()
    # predictions agree in sign almost everywhere and are numerically close
    assert np.mean(np.sign(exact) == np.sign(fast)) > 0.98
    assert np.corrcoef(exact, fast)[0, 1] > 0.995


def test_select_lambda_prefers_regularisation_when_noisy():
    X, y, g = _grouped_data(d=40, per=4, seed=6)
    lam, scores = select_lambda_lopo(X, y, g, lambdas=np.logspace(-3, 3, 7))
    assert lam > 1e-3 and len(scores) == 7


def test_singleton_groups_use_vectorised_loo():
    X, y, _ = _grouped_data(n_groups=10, per=6, seed=8)
    g = np.arange(len(y))  # every row its own group: classical leave-one-out
    fast = RidgeHead(lam=0.5, n_random=16, random_state=1).fit(X, y, g).lopo_decision()
    slow = brute_force_lopo(lambda: RidgeHead(lam=0.5, n_random=16, random_state=1), X, y, g)
    np.testing.assert_allclose(fast, slow, atol=1e-8)


def test_set_lambda_equals_fresh_fit():
    X, y, g = _grouped_data(seed=9)
    h = RidgeHead(lam=0.1, n_random=32, random_state=2).fit(X, y, g)
    a = h.set_lambda(5.0).lopo_decision()
    b = RidgeHead(lam=5.0, n_random=32, random_state=2).fit(X, y, g).lopo_decision()
    np.testing.assert_allclose(a, b, atol=1e-10)
