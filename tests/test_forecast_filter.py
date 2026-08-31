import numpy as np
import pytest

from pymgarch.correlation import cov2cor, one_step_q


class TestFilter:
    def test_filter_on_training_data_reproduces_fit(self, dcc_fit, dcc_returns):
        flt = dcc_fit.filter(dcc_returns)
        assert np.allclose(flt.sigma, dcc_fit.sigma, rtol=1e-8)
        assert np.allclose(flt.R, dcc_fit.R, atol=1e-10)
        assert flt.loglikelihood == pytest.approx(dcc_fit.loglikelihood, rel=1e-8)
        assert flt.filtered

    def test_filter_rejects_wrong_width(self, dcc_fit, dcc_returns):
        with pytest.raises(ValueError, match="columns"):
            dcc_fit.filter(dcc_returns.iloc[:, :2])

    def test_filtered_result_cannot_forecast(self, dcc_fit, dcc_returns):
        flt = dcc_fit.filter(dcc_returns)
        with pytest.raises(NotImplementedError):
            flt.forecast(horizon=2)


class TestAnalyticForecast:
    def test_one_step_correlation_matches_recursion(self, dcc_fit):
        fc = dcc_fit.forecast(horizon=1)
        a, b = dcc_fit.psi
        q1 = one_step_q(
            dcc_fit.q_last, dcc_fit.eps[-1], a, b, 0.0, dcc_fit.Sbar, dcc_fit.Nbar
        )
        assert np.allclose(fc.correlations[0], cov2cor(q1), atol=1e-12)

    def test_long_horizon_decays_to_target(self, dcc_fit):
        fc = dcc_fit.forecast(horizon=50)
        d1 = np.abs(fc.correlations[0] - dcc_fit.Sbar).max()
        d50 = np.abs(fc.correlations[-1] - dcc_fit.Sbar).max()
        assert d50 < d1
        assert d50 < 0.01 or d50 < 0.5 * d1

    def test_covariance_assembly(self, dcc_fit):
        fc = dcc_fit.forecast(horizon=3)
        for h in range(3):
            d = np.sqrt(fc.variances[h])
            assert np.allclose(
                fc.covariances[h], fc.correlations[h] * np.outer(d, d), atol=1e-12
            )

    def test_adcc_analytic_refuses(self, adcc_returns):
        from pymgarch import ADCC

        res = ADCC().fit(adcc_returns, compute_se=False)
        with pytest.raises(NotImplementedError, match="simulation"):
            res.forecast(horizon=5, method="analytic")


class TestSimulationForecast:
    def test_first_step_is_deterministic(self, dcc_fit):
        an = dcc_fit.forecast(horizon=1, method="analytic")
        sim = dcc_fit.forecast(horizon=1, method="simulation", n_paths=50, seed=1)
        # h=1 variance and correlation are known at T, so simulation must
        # match the analytic forecast exactly regardless of path count
        assert np.allclose(sim.variances[0], an.variances[0], rtol=1e-6)
        assert np.allclose(sim.correlations[0], an.correlations[0], atol=1e-10)

    def test_multi_step_close_to_analytic_for_dcc(self, dcc_fit):
        an = dcc_fit.forecast(horizon=5)
        sim = dcc_fit.forecast(horizon=5, method="simulation", n_paths=800, seed=2)
        assert np.allclose(sim.variances, an.variances, rtol=0.15)
        assert np.abs(sim.correlations - an.correlations).max() < 0.05

    def test_adcc_simulation_forecast_runs(self, adcc_returns):
        from pymgarch import ADCC

        res = ADCC().fit(adcc_returns, compute_se=False)
        fc = res.forecast(horizon=3, method="simulation", n_paths=100, seed=3)
        assert fc.correlations.shape == (3, 3, 3)
        assert np.all(np.isfinite(fc.covariances))
