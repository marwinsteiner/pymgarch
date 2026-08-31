import numpy as np
import pytest

from pymgarch import MarginalSet, UnivariateSpec


@pytest.fixture(scope="module")
def one_fit(dcc_returns):
    spec = UnivariateSpec()
    return spec.fit(dcc_returns["A"])


def test_recompute_matches_arch_at_fitted_params(one_fit):
    mset = MarginalSet.from_results([one_fit], ["A"])
    params = mset.full_params(0)
    resids, sigma2, llt = mset._recompute(0, params)
    assert np.allclose(resids, np.asarray(one_fit.resid), atol=1e-12)
    assert np.allclose(
        np.sqrt(sigma2), np.asarray(one_fit.conditional_volatility), rtol=1e-8
    )
    assert float(np.sum(llt)) == pytest.approx(float(one_fit.loglikelihood), rel=1e-8)


def test_from_returns_builds_aligned_set(dcc_returns):
    mset = MarginalSet.from_returns(dcc_returns, UnivariateSpec())
    assert mset.nassets == 3
    assert mset.nobs == len(dcc_returns)
    assert mset.std_resid.shape == (mset.nobs, 3)
    # standardized residuals should be roughly unit variance
    assert np.all(np.abs(mset.std_resid.std(axis=0) - 1.0) < 0.1)


def test_byo_results_roundtrip(dcc_returns):
    spec = UnivariateSpec()
    results = [spec.fit(dcc_returns[c]) for c in dcc_returns.columns]
    mset = MarginalSet.from_results(results, list(dcc_returns.columns))
    assert mset.names == ["A", "B", "C"]
    assert mset.total_params == sum(len(r.params) for r in results)


def test_byo_rejects_non_arch_objects():
    with pytest.raises(TypeError, match="ARCHModelResult"):
        MarginalSet.from_results([np.array([1.0])])


def test_garch_family_extraction(one_fit):
    mset = MarginalSet.from_results([one_fit], ["A"])
    fam = mset.garch_family(0)
    assert fam is not None
    vol_p, p, o, q = fam
    assert (p, o, q) == (1, 0, 1)
    assert vol_p.shape == (3,)  # omega, alpha, beta


def test_filter_new_data_reproduces_training_path(one_fit, dcc_returns):
    mset = MarginalSet.from_results([one_fit], ["A"])
    resids, sigma = mset.filter_new_data(0, dcc_returns["A"].to_numpy())
    assert np.allclose(resids, np.asarray(one_fit.resid), atol=1e-12)
    assert np.allclose(sigma, np.asarray(one_fit.conditional_volatility), rtol=1e-8)
