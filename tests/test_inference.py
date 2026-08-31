import numpy as np


def test_vcov_is_symmetric_and_positive(dcc_fit):
    V = dcc_fit.vcov
    assert V is not None
    assert np.allclose(V, V.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(V) > 0)


def test_se_scale_is_plausible(dcc_fit):
    # On T=1500 simulated data the DCC parameter SEs should be small but
    # nonzero; grossly wrong sandwich assembly shows up here.
    se = dcc_fit.std_errors
    assert 1e-4 < se["alpha"] < 0.1
    assert 1e-4 < se["beta"] < 0.2


def test_true_params_within_wide_confidence_band(dcc_fit):
    from conftest import TRUE_DCC

    se = dcc_fit.std_errors
    p = dcc_fit.params
    assert abs(p["alpha"] - TRUE_DCC["a"]) < 5 * se["alpha"] + 0.01
    assert abs(p["beta"] - TRUE_DCC["b"]) < 5 * se["beta"] + 0.02
