import numpy as np
import pytest
from conftest import SBAR

from pymgarch import CopulaGARCH
from pymgarch.copula import kendall_correlation, pit_empirical


class TestGaussianCopulaEqualsDCC:
    """A Gaussian copula with parametric normal margins IS the DCC model:
    identical joint likelihood and identical stage-2 optimum."""

    def test_exact_equivalence(self, dcc_returns, dcc_fit):
        cop = CopulaGARCH(copula="gaussian", dynamics="dcc").fit(dcc_returns)
        assert cop.params["alpha"] == pytest.approx(
            dcc_fit.params["alpha"], abs=1e-4
        )
        assert cop.params["beta"] == pytest.approx(dcc_fit.params["beta"], abs=1e-3)
        assert cop.loglikelihood == pytest.approx(dcc_fit.loglikelihood, abs=1e-3)


class TestStudentTCopula:
    @pytest.fixture(scope="class")
    def tfit(self, t_returns):
        return CopulaGARCH(copula="t", dynamics="dcc").fit(t_returns)

    def test_recovers_tail_dependence(self, tfit):
        assert 3.0 < tfit.params["nu"] < 30.0
        assert 0.0 <= tfit.params["alpha"] < 0.2
        assert 0.5 < tfit.params["beta"] < 1.0

    def test_t_beats_gaussian_copula_on_t_data(self, tfit, t_returns):
        gfit = CopulaGARCH(copula="gaussian", dynamics="dcc").fit(t_returns)
        assert tfit.loglikelihood > gfit.loglikelihood

    def test_correlation_paths_valid(self, tfit):
        R = tfit.copula_correlations
        assert np.allclose(np.einsum("tii->ti", R), 1.0, atol=1e-10)
        for t in (0, tfit.nobs - 1):
            assert np.linalg.eigvalsh(R[t])[0] > 0


class TestStaticCopulas:
    def test_static_gaussian_r_is_shock_correlation(self, dcc_returns):
        res = CopulaGARCH(copula="gaussian", dynamics="static").fit(dcc_returns)
        assert res.psi.size == 0
        expected = np.corrcoef(res.eta.T)
        assert np.abs(res.R[0] - expected).max() < 0.01
        assert np.allclose(res.R[0], res.R[-1])

    def test_dynamic_beats_static_on_dcc_data(self, dcc_returns, dcc_fit):
        static = CopulaGARCH(copula="gaussian", dynamics="static").fit(dcc_returns)
        assert dcc_fit.loglikelihood > static.loglikelihood

    def test_static_t_kendall_r_near_truth(self, t_returns):
        res = CopulaGARCH(copula="t", dynamics="static").fit(t_returns)
        assert res.psi_names == ["nu"]
        assert 3.0 < res.params["nu"] < 30.0
        assert np.abs(res.R[0] - SBAR).max() < 0.12


class TestMargins:
    def test_empirical_margins_run(self, dcc_returns):
        res = CopulaGARCH(
            copula="gaussian", dynamics="dcc", margins="empirical"
        ).fit(dcc_returns)
        assert np.all((res.u > 0) & (res.u < 1))
        assert np.isfinite(res.loglikelihood)

    def test_empirical_margins_reject_se(self, dcc_returns):
        with pytest.raises(ValueError, match="empirical"):
            CopulaGARCH(margins="empirical").fit(dcc_returns, compute_se=True)

    def test_empirical_pit_is_uniform_ranks(self):
        x = np.random.default_rng(0).standard_normal((100, 2))
        u = pit_empirical(x)
        assert sorted(u[:, 0]) == pytest.approx(
            (np.arange(1, 101) / 101.0).tolist()
        )


class TestInference:
    def test_sandwich_se_for_gaussian_dcc(self, dcc_returns):
        res = CopulaGARCH(copula="gaussian", dynamics="dcc").fit(
            dcc_returns, compute_se=True
        )
        se = res.std_errors
        assert se is not None
        assert 1e-4 < se["alpha"] < 0.2
        assert 1e-4 < se["beta"] < 0.5


class TestSimulateAndFilter:
    def test_simulate_shapes_and_seed(self, t_returns):
        res = CopulaGARCH(copula="t", dynamics="dcc").fit(t_returns)
        sim1 = res.simulate(horizon=3, n_paths=200, seed=5)
        sim2 = res.simulate(horizon=3, n_paths=200, seed=5)
        assert sim1["returns"].shape == (3, 200, 3)
        assert sim1["covariances"].shape == (3, 3, 3)
        assert np.allclose(sim1["returns"], sim2["returns"])
        # simulated covariance should be in the ballpark of the fitted
        # terminal conditional covariance scale
        H_term = np.diag(res.mset.sigma[-1] ** 2)
        ratio = np.diag(sim1["covariances"][0]) / np.diag(H_term)
        assert np.all((ratio > 0.5) & (ratio < 2.0))

    def test_filter_on_training_data_reproduces_fit(self, t_returns):
        res = CopulaGARCH(copula="t", dynamics="dcc").fit(t_returns)
        flt = res.filter(t_returns)
        assert flt.filtered
        assert flt.loglikelihood == pytest.approx(res.loglikelihood, rel=1e-6)
        assert np.allclose(flt.R, res.R, atol=1e-10)


def test_kendall_correlation_is_pd():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((300, 4)) @ np.linalg.cholesky(
        0.6 * np.ones((4, 4)) + 0.4 * np.eye(4)
    ).T
    R = kendall_correlation(z)
    assert np.allclose(np.diag(R), 1.0)
    assert np.linalg.eigvalsh(R)[0] > 0


def test_summary_renders(dcc_returns):
    res = CopulaGARCH(copula="gaussian", dynamics="dcc").fit(dcc_returns)
    text = res.summary()
    assert "Copula-GARCH" in text and "alpha" in text and "AIC" in text
