"""Fit a DCC model, inspect it, and forecast the covariance matrix."""

from pathlib import Path

import numpy as np
import pandas as pd

from pymgarch import DCC

returns = pd.read_csv(Path(__file__).parent / "data" / "dji30ret_5.csv")

# Student-t DCC(1,1) on default constant-mean GARCH(1,1) marginals
res = DCC(dist="t").fit(returns)
print(res.summary())

# conditional correlation between AA and AXP on the last day of the sample
print(f"\nlast-day corr(AA, AXP): {res.conditional_correlations[-1][0, 1]:.4f}")

# 10-step-ahead covariance forecast (analytic)
fc = res.forecast(horizon=10)
print(f"h=1  corr(AA, AXP): {fc.correlations[0][0, 1]:.4f}")
print(f"h=10 corr(AA, AXP): {fc.correlations[9][0, 1]:.4f}")
print(f"h=1  cov matrix diagonal: {np.round(np.diag(fc.covariances[0]), 3)}")
