"""Correlation targeting and (A)DCC path evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._kernels import dcc_recursion


def cov2cor(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def correlation_targets(eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uncentered targets from standardized residuals, matching rmgarch.

    Sbar = cov2cor(E'E / T); Nbar = E_-' E_- / T with E_- = min(E, 0).
    """
    T = eps.shape[0]
    Sbar = cov2cor(eps.T @ eps / T)
    neg = np.minimum(eps, 0.0)
    Nbar = neg.T @ neg / T
    return Sbar, Nbar


def adcc_delta(Sbar: np.ndarray, Nbar: np.ndarray) -> float:
    """delta = lambda_max(Sbar^{-1/2} Nbar Sbar^{-1/2}).

    With g >= 0, the targeted intercept (1-a-b)Sbar - g*Nbar is PSD iff
    a + b + delta * g <= 1, which becomes a linear constraint in (a, b, g).
    """
    vals, vecs = np.linalg.eigh(Sbar)
    inv_sqrt = (vecs / np.sqrt(vals)) @ vecs.T
    M = inv_sqrt @ Nbar @ inv_sqrt
    return float(np.linalg.eigvalsh(M)[-1])


@dataclass
class CorrPath:
    ok: bool
    logdet: np.ndarray
    quad: np.ndarray
    R: np.ndarray
    q_last: np.ndarray


def dcc_path(
    eps: np.ndarray,
    a: float,
    b: float,
    g: float,
    Sbar: np.ndarray,
    Nbar: np.ndarray | None,
    qinit: np.ndarray | None = None,
) -> CorrPath:
    """Evaluate the (A)DCC recursion with correlation targeting."""
    omega = (1.0 - a - b) * Sbar
    if g > 0.0:
        if Nbar is None:
            raise ValueError("asymmetric recursion requires Nbar")
        omega = omega - g * Nbar
    if qinit is None:
        qinit = Sbar
    flag, logdet, quad, R, q_last = dcc_recursion(
        np.ascontiguousarray(eps, dtype=np.float64),
        float(a),
        float(b),
        float(g),
        np.ascontiguousarray(omega, dtype=np.float64),
        np.ascontiguousarray(qinit, dtype=np.float64),
    )
    return CorrPath(flag == 0, logdet, quad, R, q_last)


def one_step_q(
    q_last: np.ndarray,
    eps_last: np.ndarray,
    a: float,
    b: float,
    g: float,
    Sbar: np.ndarray,
    Nbar: np.ndarray | None,
) -> np.ndarray:
    """Q_{T+1} given the terminal state; deterministic given data."""
    omega = (1.0 - a - b) * Sbar
    if g > 0.0:
        omega = omega - g * Nbar
    Q = omega + a * np.outer(eps_last, eps_last) + b * q_last
    if g > 0.0:
        n = np.minimum(eps_last, 0.0)
        Q = Q + g * np.outer(n, n)
    return Q
