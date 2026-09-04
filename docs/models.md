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

## GO-GARCH (van der Weide 2002)

Orthogonal-factor model $r_t = \mu + A f_t$ with statistically independent
factors, each following univariate GARCH with unit unconditional variance.
The rotation $A$ is recovered by PCA whitening plus fastICA (own
implementation, logcosh nonlinearity, deterministic sign/permutation
convention, seedable). The conditional covariance is $H_t = A D_t A'$ with
$D_t$ the diagonal of factor variances.

Because the factors are independent, the joint likelihood decomposes into
univariate factor likelihoods plus the constant Jacobian $-T\log|\det A|$,
so estimation after ICA is exactly $N$ arch fits. ICA is identified only up
to sign and permutation and requires non-Gaussian factors -- GARCH factors
are unconditionally leptokurtic, which is what makes this work.

```python
res = mg.GOGARCH().fit(returns, seed=0)
res.conditional_covariances
res.forecast(horizon=10)          # analytic factor forecasts -> A D A'
```

## Copula-GARCH (Patton 2006)

Arch marginals, a probability integral transform to uniforms, and a Gaussian
or Student-t copula with constant or DCC-driven correlation:

$$
\log f(r_t) = \sum_i \log f_i(r_{it}) + \log c(u_t; R_t).
$$

Margins are transformed either parametrically (through each marginal's
fitted arch distribution; Normal and Student-t supported) or empirically
(ranks, rescaled by $T/(T+1)$). The static t copula estimates $R$ by the
Kendall-tau transform $\sin(\pi\tau/2)$ and $\nu$ by MLE; dynamic variants
run the scalar DCC recursion on the copula shocks.

The Student-t copula is parameterized through the covariance-standardized t
family -- the same copula as the textbook one (copulas are invariant to
monotone marginal rescaling), but the recursion inputs stay unit-variance
and the existing DCC kernels apply unchanged. A useful corollary: a Gaussian
copula with parametric normal margins reproduces plain DCC exactly, which
the test suite asserts.

```python
res = mg.CopulaGARCH(copula="t", dynamics="dcc").fit(returns)
res.copula_correlations               # (T, N, N)
sim = res.simulate(horizon=10, n_paths=2000, seed=0)
sim["returns"]                        # (h, n_paths, N) full predictive draws
```

Copula forecasts are simulation-based (the copula gives full predictive
distributions, not just second moments); rmgarch makes the same choice.

## Composite likelihood for large N

`DCC(...).fit(returns, method="composite")` replaces the full N-dimensional
stage-2 likelihood with the mean of bivariate pair likelihoods (Engle,
Shephard and Sheppard 2008): each pair runs its own 2x2 recursion against
the corresponding submatrices of the correlation targets, so an objective
evaluation costs $O(TP)$ with $P$ pairs instead of an $N^3$ Cholesky per
observation. `pairs="contiguous"` (default, $P = N-1$) follows the paper's
recommendation for large cross-sections; `pairs="all"` uses every pair.

Point estimates are consistent; standard errors use the composite scores in
the two-stage sandwich (Godambe information); the reported joint likelihood
is still evaluated on the full model so results stay comparable across
methods. There is no packaged reference implementation in R or Python for
this estimator (it exists in Sheppard's MATLAB MFE toolbox), so validation
is against the full-likelihood estimator on moderate N plus simulation
recovery.

## Scalar and diagonal BEKK (Engle and Kroner 1995)

The one model here that does not decompose into marginals plus correlation:
the covariance is modeled directly on demeaned returns,

$$
H_t = C + (a a') \circ (u_{t-1} u_{t-1}') + (b b') \circ H_{t-1},
$$

the Hadamard form of diagonal BEKK ($A = \mathrm{diag}(a)$); scalar BEKK
constrains $a$ to a common value. Variance targeting sets
$C = \Sigma \circ (1 - aa' - bb')$ with $\Sigma$ the sample covariance,
leaving only $(a, b)$ to estimate by Gaussian QML, with per-asset
$a_i^2 + b_i^2 < 1$ constraints (which imply pairwise stationarity by
Cauchy-Schwarz) and an explicit PSD check on $C$. Forecasts are closed form:
$E[H_{T+h}] = C + (aa' + bb') \circ E[H_{T+h-1}]$.

```python
res = mg.BEKK("diagonal").fit(returns)
res.conditional_covariances
res.forecast(horizon=10)["covariances"]
```

Validation note: no maintained R reference exists for BEKK (mgarchBEKK is
dead), so the test suite relies on simulation-recovery and closed-form
forecast checks rather than cross-language fixtures.

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
