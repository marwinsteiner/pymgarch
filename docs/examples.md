# Examples

Runnable versions of everything below live in the repository's `examples/`
directory, with the data they use in `examples/data/` (daily percent log
returns of five Dow constituents, from the rmgarch `dji30ret` dataset). The
outputs shown are the scripts' actual output on that data.

## Fit, inspect, forecast

```python
import pandas as pd
from pymgarch import DCC

returns = pd.read_csv("examples/data/dji30ret_5.csv")

res = DCC(dist="t").fit(returns)
print(res.summary())
```

```text
DCC (Student-t)
===============
Assets: 5   Obs: 1000
Joint log-likelihood: -9827.6170   AIC: 19701.23   BIC: 19814.11

param           coef     std err         z
alpha       0.004342    0.007205     0.603
beta        0.854632    0.152756     5.595
nu          7.109071    1.271749     5.590
Covariance: two-stage-robust
```

```python
res.conditional_correlations[-1][0, 1]   # last-day corr(AA, AXP): 0.4649

fc = res.forecast(horizon=10)            # analytic
fc.correlations[0][0, 1]                 # h=1:  0.4585
fc.correlations[9][0, 1]                 # h=10: 0.4651 (decaying to target)
```

## Model comparison

```python
from pymgarch import ADCC, CCC, DCC

ccc = CCC().fit(returns)
dcc = DCC().fit(returns, compute_se=False)
adcc = ADCC().fit(returns, compute_se=False)
```

```text
model         loglik         AIC         BIC
CCC        -10111.09    20262.17    20360.33
DCC        -10110.11    20264.22    20372.19
ADCC       -10109.54    20265.07    20377.95

ADCC asymmetry gamma = 0.0109 (joint negative shocks raise correlations)
```

On this short sample the dynamic-correlation improvement is modest and the
information criteria favor the constant-correlation baseline -- a useful
reminder that DCC is not automatically the right model. ADCC has no analytic
multi-step correlation forecast, so forecast it by simulation:

```python
fc = adcc.forecast(horizon=5, method="simulation", n_paths=2000, seed=0)
fc.correlations[4][0, 1]                 # simulated h=5 corr(AA, AXP): 0.4601
```

## Custom marginals and filtering

```python
from arch import arch_model
from pymgarch import DCC, UnivariateSpec

train, test = returns.iloc[:800], returns.iloc[800:]

# GJR-GARCH(1,1)-t marginals via a spec ...
spec = UnivariateSpec(vol="GARCH", p=1, o=1, q=1, dist="t")
res = DCC().fit(train, marginals=spec, compute_se=False)

# ... or bring your own fitted arch results (identical model, same fit)
fitted = [
    arch_model(train[c], p=1, o=1, q=1, dist="t", rescale=False).fit(disp="off")
    for c in train.columns
]
res_byo = DCC().fit(train, marginals=fitted, compute_se=False)
```

```text
GJR-t marginals: joint loglik = -7944.40
bring-your-own:  joint loglik = -7944.40
```

Apply the fitted parameters to new data without re-estimating:

```python
flt = res.filter(returns)                # train + test, params fixed
flt.conditional_correlations[-1][0, 1]   # last test day: 0.4793
```
