"""Minimal fastICA for GO-GARCH factor extraction.

Self-contained (no scikit-learn): PCA whitening, symmetric fastICA with the
logcosh (tanh) nonlinearity, and a deterministic identification convention so
fits are reproducible: factor signs are flipped so each mixing-matrix column's
largest-magnitude element is positive, and factors are ordered by descending
mixing-column norm.

ICA is identified only up to sign and permutation, and only when the sources
are non-Gaussian -- which GARCH factors are unconditionally (volatility
mixing produces excess kurtosis even with Gaussian innovations).
"""

from __future__ import annotations

import numpy as np


class ICAResult:
    """Mixing/unmixing pair for x_t = A f_t with f_t = U x_t."""

    def __init__(self, A: np.ndarray, U: np.ndarray, n_iter: int, converged: bool):
        self.A = A
        self.U = U
        self.n_iter = n_iter
        self.converged = converged


def _sym_decorrelate(W: np.ndarray) -> np.ndarray:
    """W <- (W W')^{-1/2} W, making the rows exactly orthonormal."""
    vals, vecs = np.linalg.eigh(W @ W.T)
    return (vecs / np.sqrt(vals)) @ vecs.T @ W


def fastica(
    x: np.ndarray,
    seed: int = 0,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> ICAResult:
    """Estimate the ICA decomposition of demeaned data x (T, N).

    Returns mixing A and unmixing U with U = A^{-1}, ordered and signed by
    the identification convention above. Raises if the sample covariance is
    numerically singular.
    """
    x = np.asarray(x, dtype=float)
    T, N = x.shape
    cov = x.T @ x / T
    vals, vecs = np.linalg.eigh(cov)
    if vals[0] <= 1e-12 * vals[-1]:
        raise ValueError(
            "sample covariance is numerically singular; remove collinear "
            "or constant columns before GO-GARCH"
        )
    # PCA whitening: z = x K' with cov(z) = I
    K = (vecs / np.sqrt(vals)).T  # (N, N), rows scaled eigenvectors
    z = x @ K.T

    rng = np.random.default_rng(seed)
    W = _sym_decorrelate(rng.standard_normal((N, N)))
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        wz = z @ W.T  # (T, N) current source estimates
        g = np.tanh(wz)
        g_prime = 1.0 - g**2
        W_new = (g.T @ z) / T - np.diag(g_prime.mean(axis=0)) @ W
        W_new = _sym_decorrelate(W_new)
        # convergence: rows aligned with previous rows up to sign
        delta = np.max(np.abs(np.abs(np.einsum("ij,ij->i", W_new, W)) - 1.0))
        W = W_new
        if delta < tol:
            converged = True
            break

    U = W @ K  # total unmixing: f = U x
    A = np.linalg.inv(U)

    # identification: sign so each A column's largest |element| is positive,
    # order by descending column norm (variance contribution)
    for j in range(N):
        k = int(np.argmax(np.abs(A[:, j])))
        if A[k, j] < 0:
            A[:, j] = -A[:, j]
            U[j, :] = -U[j, :]
    order = np.argsort(-np.linalg.norm(A, axis=0))
    A = A[:, order]
    U = U[order, :]
    return ICAResult(A=A, U=U, n_iter=it, converged=converged)
