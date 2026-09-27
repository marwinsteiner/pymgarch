import numpy as np
import pytest

from pymgarch import _kernels
from pymgarch.correlation import correlation_targets

numba = pytest.importorskip("numba")


def test_kernels_are_jitted():
    assert _kernels.HAVE_NUMBA
    assert hasattr(_kernels.dcc_recursion, "py_func")


def test_bekk_jitted_matches_python():
    rng = np.random.default_rng(2)
    u = rng.standard_normal((150, 3))
    Sigma = np.cov(u.T)
    avec = np.array([0.3, 0.25, 0.35])
    bvec = np.array([0.9, 0.92, 0.88])
    C = Sigma * (1.0 - np.outer(avec, avec) - np.outer(bvec, bvec))
    args = (u, avec, bvec, C, Sigma, 1)
    fj, lj, Hj, _hlj = _kernels.bekk_recursion(*args)
    fp, lp, Hp, _hlp = _kernels.bekk_recursion.py_func(*args)
    assert fj == fp == 0
    assert np.allclose(lj, lp, atol=1e-12)
    assert np.allclose(Hj, Hp, atol=1e-12)
    # want_path=0 gives identical llt with a stub path
    f0, l0, H0, _ = _kernels.bekk_recursion(u, avec, bvec, C, Sigma, 0)
    assert f0 == 0 and np.allclose(l0, lj, atol=1e-12)
    assert H0.shape == (1, 3, 3)


def test_jitted_matches_python():
    rng = np.random.default_rng(0)
    eps = rng.standard_normal((200, 3))
    Sbar, _ = correlation_targets(eps)
    omega = 0.05 * Sbar
    args = (eps, 0.04, 0.91, 0.0, omega, Sbar.copy())
    flag_j, ld_j, q_j, R_j, ql_j = _kernels.dcc_recursion(*args)
    flag_p, ld_p, q_p, R_p, ql_p = _kernels.dcc_recursion.py_func(*args)
    assert flag_j == flag_p == 0
    assert np.allclose(ld_j, ld_p, atol=1e-12)
    assert np.allclose(q_j, q_p, atol=1e-12)
    assert np.allclose(R_j, R_p, atol=1e-12)
    assert np.allclose(ql_j, ql_p, atol=1e-12)
