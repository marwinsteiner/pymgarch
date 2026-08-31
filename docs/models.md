# Models

All stage-2 models operate on standardized residuals $\varepsilon_t = D_t^{-1}
(r_t - \mu_t)$, where $D_t = \mathrm{diag}(\sigma_{1t}, \dots, \sigma_{Nt})$
comes from the fitted arch marginals. The conditional covariance is
$H_t = D_t R_t D_t$.

## CCC (Bollerslev 1990)

Constant correlation $R = \mathrm{cov2cor}(E'E/T)$. No stage-2 parameters;
closed form given the marginals. Gaussian likelihood only in v0.1.

## DCC(1,1) (Engle 2002)

With correlation targeting $\bar S = \mathrm{cov2cor}(E'E/T)$:

$$
Q_t = (1 - a - b)\bar S + a\,\varepsilon_{t-1}\varepsilon_{t-1}' + b\,Q_{t-1},
\qquad
R_t = \mathrm{diag}(Q_t)^{-1/2} Q_t \,\mathrm{diag}(Q_t)^{-1/2},
$$

with $a, b \ge 0$, $a + b < 1$, $Q_1 = \bar S$. Estimated by two-stage QML
(SLSQP, multiple starts). Distributions: Gaussian and covariance-standardized
multivariate Student-t (degrees of freedom estimated jointly with $a, b$).

## ADCC (Cappiello, Engle and Sheppard 2006)

Scalar asymmetric DCC with $n_t = \min(\varepsilon_t, 0)$ elementwise and
$\bar N = T^{-1}\sum_t n_t n_t'$:

$$
Q_t = \left[(1-a-b)\bar S - g\bar N\right]
 + a\,\varepsilon_{t-1}\varepsilon_{t-1}'
 + g\,n_{t-1}n_{t-1}'
 + b\,Q_{t-1}.
$$

The intercept is positive semi-definite iff $a + b + \delta g \le 1$ with
$\delta = \lambda_{\max}(\bar S^{-1/2} \bar N \bar S^{-1/2})$, which is a
linear constraint in $(a, b, g)$ and is imposed exactly during estimation.

## Forecasting

Marginal variance forecasts are delegated to arch. One-step correlations are
deterministic. For $h \ge 2$, DCC uses the Engle-Sheppard approximation

$$
E[R_{T+h}] \approx \left(1 - (a+b)^{h-1}\right)\bar S + (a+b)^{h-1} R_{T+1},
$$

and every model supports Monte Carlo forecasts (`method="simulation"`) that
propagate the exact recursions; ADCC requires them. The analytic covariance
assembly $\hat H = \hat D \hat R \hat D$ ignores a Jensen gap that the
simulation method does not.

## Reported likelihood

`loglikelihood` is the joint likelihood of the returns under the stage-2
distribution (the same quantity rmgarch's `likelihood()` reports): the
multivariate density of $\varepsilon_t$ at $R_t$ plus the Jacobian
$-\sum_i \log \sigma_{it}$, summed over $t$.
