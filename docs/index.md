# pymgarch

Multivariate GARCH for Python: DCC, ADCC, and CCC correlation dynamics on top
of `arch` univariate marginals, validated against R's rmgarch.

Python has had no maintained general-purpose multivariate GARCH framework:
the existing packages stop at Gaussian DCC(1,1), while R users have had a
full model stack in rmgarch (now tsmarch) for a decade. pymgarch closes that
gap incrementally. Stage 1 (univariate volatility) is delegated to `arch`;
stage 2 (correlation dynamics) is what this library implements, with
Engle-Sheppard two-stage standard errors and replication tests against
rmgarch's fitted parameters and likelihoods.

```{toctree}
:maxdepth: 2

quickstart
examples
models
inference
api
```

## Install

```
pip install pymgarch          # or: pip install pymgarch[numba]
```
