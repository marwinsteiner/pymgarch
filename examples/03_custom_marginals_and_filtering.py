"""Custom marginal specs, bring-your-own arch fits, and filtering new data."""

from pathlib import Path

import pandas as pd
from arch import arch_model

from pymgarch import DCC, UnivariateSpec

returns = pd.read_csv(Path(__file__).parent / "data" / "dji30ret_5.csv")
train, test = returns.iloc[:800], returns.iloc[800:]

# GJR-GARCH(1,1) with Student-t innovations for every marginal
spec = UnivariateSpec(vol="GARCH", p=1, o=1, q=1, dist="t")
res = DCC().fit(train, marginals=spec, compute_se=False)
print(f"GJR-t marginals: joint loglik = {res.loglikelihood:.2f}")

# equivalent bring-your-own route: any fitted arch results work
fitted = [
    arch_model(train[c], p=1, o=1, q=1, dist="t", rescale=False).fit(disp="off")
    for c in train.columns
]
res_byo = DCC().fit(train, marginals=fitted, compute_se=False)
print(f"bring-your-own:  joint loglik = {res_byo.loglikelihood:.2f}")

# apply the fitted parameters to the full sample without re-estimating
flt = res.filter(returns)
print(f"filtered {flt.nobs} obs; "
      f"corr(AA, AXP) on last test day: {flt.conditional_correlations[-1][0, 1]:.4f}")
