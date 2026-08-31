import numpy as np
import pytest
from conftest import TRUE_DCC

from pymgarch import ADCC, CCC, DCC
from pymgarch.estimation import ParamLayout, stage2_llt


class TestDCC:
    def test_recovers_true_parameters(self, dcc_fit):
        params = dcc_fit.params
        assert params["alpha"] == pytest.approx(TRUE_DCC["a"], abs=0.03)
        assert params["beta"] == pytest.approx(TRUE_DCC["b"], abs=0.06)

    def test_fitted_ll_beats_true_parameters(self, dcc_fit):
        layout = ParamLayout(asymmetric=False, studentt=False)
        llt_true = stage2_llt(
            np.array([TRUE_DCC["a"], TRUE_DCC["b"]]),
            dcc_fit.eps,
            dcc_fit.Sbar,
            dcc_fit.Nbar,
            layout,
        )
        llt_fit = stage2_llt(
            dcc_fit.psi, dcc_fit.eps, dcc_fit.Sbar, dcc_fit.Nbar, layout
        )
        assert float(np.sum(llt_fit)) >= float(np.sum(llt_true)) - 1e-6

    def test_correlation_path_shape_and_validity(self, dcc_fit):
        R = dcc_fit.conditional_correlations
        assert R.shape == (dcc_fit.nobs, 3, 3)
        assert np.allclose(np.diagonal(R, axis1=1, axis2=2), 1.0, atol=1e-10)
        H = dcc_fit.conditional_covariances
        assert np.all(np.diagonal(H, axis1=1, axis2=2) > 0)

    def test_standard_errors_present(self, dcc_fit):
        se = dcc_fit.std_errors
        assert se is not None
        assert se["alpha"] > 0 and se["beta"] > 0
        assert se["alpha"] < 0.2 and se["beta"] < 0.3
        assert dcc_fit.se_method == "two-stage-robust"

    def test_summary_renders(self, dcc_fit):
        text = dcc_fit.summary()
        assert "DCC" in text and "alpha" in text and "AIC" in text

    def test_compute_se_false_skips_inference(self, dcc_returns):
        res = DCC().fit(dcc_returns, compute_se=False)
        assert res.std_errors is None


class TestCCC:
    def test_dcc_dominates_ccc_on_dcc_data(self, dcc_returns, dcc_fit):
        ccc = CCC().fit(dcc_returns)
        assert dcc_fit.loglikelihood > ccc.loglikelihood
        assert ccc.R.shape == dcc_fit.R.shape
        assert np.allclose(ccc.R[0], ccc.R[-1])

    def test_ccc_rejects_t(self):
        with pytest.raises(NotImplementedError):
            CCC(dist="t")


class TestADCC:
    def test_recovers_asymmetry(self, adcc_returns):
        res = ADCC().fit(adcc_returns, compute_se=False)
        params = res.params
        assert params["gamma"] > 0.01
        assert params["alpha"] == pytest.approx(0.03, abs=0.04)
        assert params["beta"] == pytest.approx(0.90, abs=0.10)

    def test_adcc_nests_dcc(self, dcc_fit):
        # ADCC likelihood at (a, b, g=0) equals the DCC likelihood
        layout = ParamLayout(asymmetric=True, studentt=False)
        psi = np.array([dcc_fit.psi[0], dcc_fit.psi[1], 0.0])
        llt = stage2_llt(psi, dcc_fit.eps, dcc_fit.Sbar, dcc_fit.Nbar, layout)
        layout0 = ParamLayout(asymmetric=False, studentt=False)
        llt0 = stage2_llt(
            dcc_fit.psi, dcc_fit.eps, dcc_fit.Sbar, dcc_fit.Nbar, layout0
        )
        assert np.allclose(llt, llt0)


class TestStudentT:
    def test_recovers_nu(self, t_returns):
        res = DCC(dist="t").fit(t_returns, compute_se=False)
        assert 4.0 < res.params["nu"] < 20.0
        assert res.params["alpha"] == pytest.approx(TRUE_DCC["a"], abs=0.04)
        assert res.params["beta"] == pytest.approx(TRUE_DCC["b"], abs=0.10)

    def test_t_beats_gaussian_on_t_data(self, t_returns):
        rt = DCC(dist="t").fit(t_returns, compute_se=False)
        rn = DCC(dist="norm").fit(t_returns, compute_se=False)
        assert rt.loglikelihood > rn.loglikelihood


def test_ndarray_input_works(dcc_returns):
    res = DCC().fit(dcc_returns.to_numpy(), compute_se=False)
    assert res.names == ["y0", "y1", "y2"]
    assert res.nobs == len(dcc_returns)
