# Quickstart

Fit a Gaussian DCC(1,1) with default constant-mean GARCH(1,1) marginals:

```python
import pymgarch as mg

res = mg.DCC().fit(returns)          # (T, N) DataFrame, percent scale
print(res.summary())

res.params                            # {"alpha": ..., "beta": ...}
res.std_errors                        # two-stage robust standard errors
res.conditional_correlations          # (T, N, N)
res.conditional_covariances           # (T, N, N)
```

## Distributions and asymmetry

```python
mg.DCC(dist="t").fit(returns)         # multivariate Student-t stage 2
mg.ADCC().fit(returns)                # asymmetric (Cappiello-Engle-Sheppard)
mg.CCC().fit(returns)                 # constant-correlation baseline
```

## Custom marginals

Apply one spec to every column, or bring fitted arch results:

```python
spec = mg.UnivariateSpec(vol="GARCH", p=1, o=1, q=1, dist="t")   # GJR-t
res = mg.DCC().fit(returns, marginals=spec)

from arch import arch_model
fitted = [
    arch_model(returns[c], rescale=False).fit(disp="off")
    for c in returns.columns
]
res = mg.DCC().fit(returns, marginals=fitted)
```

## Forecasting and filtering

```python
fc = res.forecast(horizon=10)                                   # analytic
fc = res.forecast(horizon=10, method="simulation", n_paths=2000)
fc.variances        # (h, N)
fc.correlations     # (h, N, N)
fc.covariances      # (h, N, N)

flt = res.filter(new_returns)   # apply fitted params to new data
```

ADCC has no standard analytic multi-step correlation forecast; use
`method="simulation"` there.

## Scaling note

Like `arch`, the optimizers behave best when returns are on a percent scale
(roughly unit variance). pymgarch fits marginals with `rescale=False` so
parameters stay comparable with R fits on the same series; multiply decimal
returns by 100 before fitting.
