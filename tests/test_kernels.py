import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pymgarch.correlation import correlation_targets, dcc_path
from pymgarch.distributions import mvnorm_llt


def _random_eps(seed, T=120, N=3):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((T, N))
    mix = np.eye(N) + 0.3
    return z @ np.linalg.cholesky(mix / mix[0, 0]).T


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    a=st.floats(0.005, 0.15),
    persistence=st.floats(0.5, 0.99),
)
def test_dcc_recursion_produces_valid_correlations(seed, a, persistence):
    b = persistence - a
    if b <= 0:
        return
    eps = _random_eps(seed)
    Sbar, Nbar = correlation_targets(eps)
    path = dcc_path(eps, a, b, 0.0, Sbar, Nbar)
    assert path.ok
    T, N = eps.shape
    assert path.R.shape == (T, N, N)
    for t in (0, T // 2, T - 1):
        R = path.R[t]
        assert np.allclose(R, R.T, atol=1e-12)
        assert np.allclose(np.diag(R), 1.0, atol=1e-12)
        assert np.linalg.eigvalsh(R)[0] > 0


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_logdet_quad_match_numpy(seed):
    eps = _random_eps(seed, T=60, N=4)
    Sbar, Nbar = correlation_targets(eps)
    path = dcc_path(eps, 0.05, 0.9, 0.0, Sbar, Nbar)
    assert path.ok
    for t in (0, 30, 59):
        R = path.R[t]
        sign, logdet = np.linalg.slogdet(R)
        assert sign > 0
        quad = eps[t] @ np.linalg.solve(R, eps[t])
        assert path.logdet[t] == pytest.approx(logdet, rel=1e-10)
        assert path.quad[t] == pytest.approx(quad, rel=1e-10)


def test_asymmetric_term_reduces_to_symmetric_when_g_zero():
    eps = _random_eps(3)
    Sbar, Nbar = correlation_targets(eps)
    sym = dcc_path(eps, 0.05, 0.9, 0.0, Sbar, Nbar)
    asym = dcc_path(eps, 0.05, 0.9, 0.0, Sbar, None)
    assert np.allclose(sym.R, asym.R)


def test_gaussian_llt_is_finite_and_negative_on_average():
    eps = _random_eps(5)
    Sbar, Nbar = correlation_targets(eps)
    path = dcc_path(eps, 0.05, 0.9, 0.0, Sbar, Nbar)
    llt = mvnorm_llt(path.logdet, path.quad, eps.shape[1])
    assert np.all(np.isfinite(llt))
    assert llt.mean() < 0
