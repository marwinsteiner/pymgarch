import numpy as np
import pandas as pd
import pytest
from conftest import GARCH11, SBAR, TRUE_DCC

from pymgarch import CCC, DCC, GOGARCH, dcc_test, simulate_dcc


class TestDCCTest:
    def test_rejects_on_dcc_data(self, dcc_returns):
        res = dcc_test(dcc_returns)
        assert res.pvalue < 0.05
        assert res.df == 2

    def test_accepts_on_constant_correlation_data(self):
        # a = b = 0 collapses the recursion to constant correlation
        sim = simulate_dcc([GARCH11] * 3, a=0.0, b=0.0, Sbar=SBAR, nobs=1500, seed=4)
        df = pd.DataFrame(sim["returns"], columns=["A", "B", "C"])
        res = dcc_test(df)
        assert res.pvalue > 0.05

    def test_accepts_precomputed_eps(self, dcc_fit):
        res = dcc_test(eps=dcc_fit.eps)
        assert np.isfinite(res.statistic)
        assert res.pvalue < 0.05

    def test_lags_change_df(self, dcc_returns):
        res = dcc_test(dcc_returns, n_lags=3)
        assert res.df == 4

    def test_rejects_single_asset(self):
        with pytest.raises(ValueError, match="two assets"):
            dcc_test(eps=np.random.default_rng(0).standard_normal((100, 1)))


class TestNewsImpact:
    def test_dcc_surface_shape_and_center(self, dcc_fit):
        ni = dcc_fit.news_impact(pair=(0, 1))
        n = ni["x"].shape[0]
        assert ni["z"].shape == (n, n)
        # at (0,0): Q = (1-a-b)Sbar + b*Sbar, correlation shrinks toward Sbar
        a, b = dcc_fit.psi
        Q = (1 - a - b) * dcc_fit.Sbar + b * dcc_fit.Sbar
        expected = Q[0, 1] / np.sqrt(Q[0, 0] * Q[1, 1])
        mid = n // 2
        assert ni["z"][mid, mid] == pytest.approx(expected, abs=1e-10)

    def test_dcc_surface_symmetric_without_asymmetry(self, dcc_fit):
        ni = dcc_fit.news_impact(pair=(0, 1))
        assert np.allclose(ni["z"], ni["z"][::-1, ::-1], atol=1e-12)

    def test_adcc_surface_asymmetric(self, adcc_returns):
        from pymgarch import ADCC

        res = ADCC().fit(adcc_returns, compute_se=False)
        ni = res.news_impact(pair=(0, 1))
        lo, hi = 0, -1
        # joint negative shocks raise correlation more than joint positive
        assert ni["z"][lo, lo] > ni["z"][hi, hi]

    def test_covariance_kind_scales(self, dcc_fit):
        c = dcc_fit.news_impact(pair=(0, 1), kind="correlation")
        v = dcc_fit.news_impact(pair=(0, 1), kind="covariance")
        assert np.all(np.sign(v["z"]) == np.sign(c["z"]))
        assert not np.allclose(v["z"], c["z"])

    def test_ccc_refuses(self, dcc_returns):
        res = CCC().fit(dcc_returns)
        with pytest.raises(ValueError, match="constant"):
            res.news_impact()

    def test_gogarch_factor_surface(self, dcc_returns):
        g = GOGARCH().fit(dcc_returns, seed=0)
        ni = g.news_impact(pair=(0, 1), factors=(0, 1))
        assert np.all(np.isfinite(ni["z"]))
        nc = g.news_impact(pair=(0, 1), factors=(0, 1), kind="correlation")
        assert np.all(np.abs(nc["z"]) <= 1.0 + 1e-12)


class TestResultSimulate:
    def test_shapes_and_seed(self, dcc_fit):
        s1 = dcc_fit.simulate(horizon=4, n_paths=300, seed=7)
        s2 = dcc_fit.simulate(horizon=4, n_paths=300, seed=7)
        assert s1["returns"].shape == (4, 300, 3)
        assert np.allclose(s1["returns"], s2["returns"])

    def test_h1_moments_match_forecast(self, dcc_fit):
        sim = dcc_fit.simulate(horizon=1, n_paths=8000, seed=1)
        fc = dcc_fit.forecast(horizon=1)
        # h=1 variance is deterministic; sample var of draws approximates it
        sample_var = sim["returns"][0].var(axis=0)
        assert np.allclose(sample_var, fc.variances[0], rtol=0.1)
        sample_corr = np.corrcoef(sim["returns"][0].T)
        assert np.abs(sample_corr - fc.correlations[0]).max() < 0.05

    def test_ccc_simulate_constant_correlation(self, dcc_returns):
        res = CCC().fit(dcc_returns)
        sim = res.simulate(horizon=2, n_paths=4000, seed=2)
        corr = np.corrcoef(sim["returns"][0].T)
        assert np.abs(corr - res.Sbar).max() < 0.06

    def test_gogarch_simulate(self, dcc_returns):
        g = GOGARCH().fit(dcc_returns, seed=0)
        sim = g.simulate(horizon=3, n_paths=200, seed=3)
        assert sim["returns"].shape == (3, 200, 3)
        assert np.all(sim["factor_variances"] > 0)


class TestFilteredForecast:
    def test_filter_train_forecast_equals_fit_forecast(self, dcc_fit, dcc_returns):
        flt = dcc_fit.filter(dcc_returns)
        fc_fit = dcc_fit.forecast(horizon=5)
        fc_flt = flt.forecast(horizon=5)
        assert np.allclose(fc_flt.variances, fc_fit.variances, rtol=1e-6)
        assert np.allclose(fc_flt.correlations, fc_fit.correlations, atol=1e-10)

    def test_filtered_simulation_forecast_runs(self, dcc_fit, dcc_returns):
        flt = dcc_fit.filter(dcc_returns.iloc[:1200])
        fc = flt.forecast(horizon=3, method="simulation", n_paths=100, seed=5)
        assert np.all(np.isfinite(fc.covariances))

    def test_filtered_forecast_uses_new_state(self, dcc_fit, dcc_returns):
        flt = dcc_fit.filter(dcc_returns.iloc[:800])
        fc_flt = flt.forecast(horizon=1)
        fc_fit = dcc_fit.forecast(horizon=1)
        # different terminal states -> different one-step forecasts
        assert not np.allclose(fc_flt.variances[0], fc_fit.variances[0])


class TestCCCStudentT:
    def test_recovers_nu_on_t_data(self, t_returns):
        res = CCC(dist="t").fit(t_returns)
        assert res.psi_names == ["nu"]
        assert 4.0 < res.params["nu"] < 25.0

    def test_t_beats_gaussian_on_t_data(self, t_returns):
        rt = CCC(dist="t").fit(t_returns)
        rn = CCC().fit(t_returns)
        assert rt.loglikelihood > rn.loglikelihood

    def test_dcc_t_beats_ccc_t_on_dcc_data(self, t_returns):
        ct = CCC(dist="t").fit(t_returns)
        dt = DCC(dist="t").fit(t_returns, compute_se=False)
        assert dt.loglikelihood > ct.loglikelihood

    def test_filter_roundtrip(self, t_returns):
        res = CCC(dist="t").fit(t_returns)
        flt = res.filter(t_returns)
        assert flt.loglikelihood == pytest.approx(res.loglikelihood, rel=1e-8)


def test_true_dcc_data_needs_dcc(dcc_returns):
    # sanity tie-in: the test motivates the model class on its own DGP
    res = dcc_test(dcc_returns)
    assert res.pvalue < 0.05, TRUE_DCC
