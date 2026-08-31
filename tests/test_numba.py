import numpy as np
import pytest

from pymgarch import _kernels
from pymgarch.correlation import correlation_targets

numba = pytest.importorskip("numba")


def test_kernels_are_jitted():
    assert _kernels.HAVE_NUMBA
    assert hasattr(_kernels.dcc_recursion, "py_func")


def test_jitted_matches_python():
    rng = np.random.default_rng(0)
    eps = rng.standard_normal((200, 3))
    Sbar, Nbar = correlation_targets(eps)
    omega = 0.05 * Sbar
    args = (eps, 0.04, 0.91, 0.0, omega, Sbar.copy())
    flag_j, ld_j, q_j, R_j, ql_j = _kernels.dcc_recursion(*args)
    flag_p, ld_p, q_p, R_p, ql_p = _kernels.dcc_recursion.py_func(*args)
    assert flag_j == flag_p == 0
    assert np.allclose(ld_j, ld_p, atol=1e-12)
    assert np.allclose(q_j, q_p, atol=1e-12)
    assert np.allclose(R_j, R_p, atol=1e-12)
    assert np.allclose(ql_j, ql_p, atol=1e-12)
