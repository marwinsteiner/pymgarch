"""Hot loops for the (A)DCC correlation recursion.

Every function here is written in a numba-compatible subset of numpy (explicit
loops, no exceptions, no scipy) so the same code runs jitted when numba is
installed and as plain Python otherwise. Failures are signalled through an
integer flag rather than exceptions because numba cannot raise from nopython
mode reliably across versions.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised indirectly via test_numba
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(func):
            return func

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return wrap


@njit(cache=True)
def _chol_lower(A, L):
    """Lower Cholesky of A into L. Returns 0 on success, 1 if not PD."""
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 1e-14:
                    return 1
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return 0


@njit(cache=True)
def _logdet_quad(L, x):
    """Given lower Cholesky L of R and vector x: (log|R|, x' R^{-1} x)."""
    n = L.shape[0]
    y = np.empty(n)
    logdet = 0.0
    quad = 0.0
    for i in range(n):
        s = x[i]
        for k in range(i):
            s -= L[i, k] * y[k]
        y[i] = s / L[i, i]
        quad += y[i] * y[i]
        logdet += np.log(L[i, i])
    return 2.0 * logdet, quad


@njit(cache=True)
def dcc_recursion(eps, a, b, g, omega, qinit):
    """Run the scalar (A)DCC recursion over standardized residuals.

    Q_t = omega + a e_{t-1} e_{t-1}' + g n_{t-1} n_{t-1}' + b Q_{t-1},
    n_t = min(e_t, 0); Q_1 = qinit; R_t = cov2cor(Q_t).

    Returns (flag, logdet, quad, R, q_last) where flag != 0 signals a
    numerical failure (non-PD Q_t or R_t) and the arrays are then invalid.
    """
    T, N = eps.shape
    R = np.empty((T, N, N))
    logdet = np.empty(T)
    quad = np.empty(T)
    Q = qinit.copy()
    Rt = np.empty((N, N))
    L = np.zeros((N, N))
    for t in range(T):
        if t > 0:
            e = eps[t - 1]
            for i in range(N):
                for j in range(N):
                    x = omega[i, j] + a * e[i] * e[j] + b * Q[i, j]
                    if g > 0.0:
                        ni = e[i] if e[i] < 0.0 else 0.0
                        nj = e[j] if e[j] < 0.0 else 0.0
                        x += g * ni * nj
                    Q[i, j] = x
        for i in range(N):
            if Q[i, i] <= 0.0:
                return 1, logdet, quad, R, Q
        for i in range(N):
            di = np.sqrt(Q[i, i])
            for j in range(N):
                Rt[i, j] = Q[i, j] / (di * np.sqrt(Q[j, j]))
        ok = _chol_lower(Rt, L)
        if ok != 0:
            return 1, logdet, quad, R, Q
        ld, qd = _logdet_quad(L, eps[t])
        logdet[t] = ld
        quad[t] = qd
        R[t, :, :] = Rt
    return 0, logdet, quad, R, Q


@njit(cache=True)
def garch_forward(vol_params, p, o, q, z, u_lags, s2_lags):
    """Simulate a GARCH/GJR variance recursion forward through shocks z.

    vol_params: (omega, alpha_1..p, gamma_1..o, beta_1..q) in arch's order.
    z: (T, M) standardized innovations for M parallel paths.
    u_lags: (max(p,o), M) most recent residuals, row 0 = most recent.
    s2_lags: (q, M) most recent conditional variances, row 0 = most recent.

    Returns (u, sigma2) each (T, M). Lag buffers are updated in place.
    """
    T, M = z.shape
    omega = vol_params[0]
    u = np.empty((T, M))
    sigma2 = np.empty((T, M))
    for t in range(T):
        for m in range(M):
            s2 = omega
            for i in range(p):
                s2 += vol_params[1 + i] * u_lags[i, m] * u_lags[i, m]
            for j in range(o):
                if u_lags[j, m] < 0.0:
                    s2 += vol_params[1 + p + j] * u_lags[j, m] * u_lags[j, m]
            for k in range(q):
                s2 += vol_params[1 + p + o + k] * s2_lags[k, m]
            sigma2[t, m] = s2
            u[t, m] = np.sqrt(s2) * z[t, m]
        # shift lag buffers (row 0 most recent)
        nlag = u_lags.shape[0]
        for i in range(nlag - 1, 0, -1):
            for m in range(M):
                u_lags[i, m] = u_lags[i - 1, m]
        if nlag > 0:
            for m in range(M):
                u_lags[0, m] = u[t, m]
        for k in range(q - 1, 0, -1):
            for m in range(M):
                s2_lags[k, m] = s2_lags[k - 1, m]
        if q > 0:
            for m in range(M):
                s2_lags[0, m] = sigma2[t, m]
    return u, sigma2
