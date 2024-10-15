# Models
**PostForecasts.jl** provides four models for postprocessing point predictions for probabilistic forecasts: 
- Normal error distribution
- Conformal prediction
- Isotonic distributional regression
- Quantile regression

Every model belongs to the `PostModel` supertype. Models that work exclusively with a single point forecast as a regressor are of type `UniPostModel`, while models that support multiple regressors are of type `MultiPostModel`.

## Normal error distribution
The naive model for probabilistic forecasting, which assumes normally distributed errors of point forecasts. 

The predictive distribution conditional on point forecast $\hat{y}$ is a Gaussian $\mathcal{N}(\hat{y} + \mu, \sigma)$, where $\mu$ and $\sigma$ are mean and sample standard deviation of errors in the training window.

The $\tau$-th quantile conditional on $\hat{y}$ of such parameterized distribution can be obtained via an analytic expression:

$\hat{q}_{\tau|\hat{y}} = \hat{y} + \mu + \sigma \sqrt{2} \cdot \text{erf}^{-1} (2\tau - 1).$

```@docs
Normal
getmean
getstd
```

## Conformal prediction and historical simulation
Conformal prediction is a machine learning framework for computing prediction intervals based on the outputs of an arbirary point forecasting model. The implemented version of Conformal Prediction is analogous to the inductive approach used by [Kath and Ziel (2021)](https://doi.org/10.1016/j.ijforecast.2020.09.006). 

In the training step, the non-conformity scores $\lambda_i$ are calculated on the training set $(\hat{y}_i, y_i)_{i\in\text{training window}}$ as $\lambda_i := |\hat{y}_i - y_i|$.

In the prediction step, the $\tau$-th quantile conditional on $\hat{y_t}$ is obtained by shifting the prediction by an appropriate empirical quantile of non-conformity scores:

$\hat{q}_{\tau|\hat{y}} = \hat{y} - \mathbf{1}_{\{\tau < 0.5\}} Q_{1 - 2\tau}(\lambda) + \mathbf{1}_{\{\tau > 0.5\}} Q_{2\tau - 1}(\lambda),$

where $Q_{\alpha}(\lambda)$ is the $\alpha$-th empirical quantile of non-conformity scores from the training window. Although the intervals in the form of $[\hat{y} - Q_{\alpha}(\lambda), \hat{y} +Q_{\alpha}(\lambda)]$ are valid $\alpha$ prediction intervals without any requirements on the underlying distribution, translating them into quantiles requires the assumption of symmetrically distributed errors.

However, it is also possible to use conformal prediction to obtain non-symmetric distributions, by using non-absolute errors $\lambda_i := \hat{y}_i - y_i$. Then, in the predcition step $\tau$-th quantile conditional on $\hat{y_t}$ is computed as:

$\hat{q}_{\tau|\hat{y}} = \hat{y} + Q_{\tau}(\lambda),$

the method known as historical simulation [(Nowotarski and Weron, 2018)](https://doi.org/10.1016/j.rser.2017.05.234)

```@docs
CP
getscores
```

## Isotonic distributional regression
Isotonic Distributional Regression [(IDR; Henzi et al., 2021)](https://doi.org/10.1111/rssb.12450) has been recently introduced as a nonparametric method for estimating distributions that are isotonic in the regressed variable, which means that the quantiles of such distributions are non-decreasing w.r.t the regressor. In the training step, $n$ observations $(\hat{y}_i, y_i)_{i \in \text{training window}}$ are first sorted to be ascending in $\hat{y}_i$. Then, $n$ conditional distributions $\hat{F_i}(z) = \hat{F}(z|x_i)$ are obtained by solving the following min-max problem via abridged pool-adjacent-violators algorithm:
 
$\hat{F}_i(z) = \min_{k=1,...,i} \max_{j=k,..,n} \frac{1}{j-k+1}\sum_{l=k}^{j} \mathbb{1}\{y_{l} < z\},$

where $z \in (y_i)_{i \in \text{training window}}$
To obtain conditional distribution for any $\hat{y}\in\mathbb{R}$, the obtained distribution functions are interpolated

$\hat{F}(z|\hat{y}) = \frac{\hat{y}-\hat{y}_i}{\hat{y}_{i+1} - \hat{y}_i}\hat{F}_i(z) + \frac{\hat{y}_{i+1} - \hat{y}}{\hat{y}_{i+1} - \hat{y}_i} \hat{F}_{i+1}(z),$

If $\hat{y} < \hat{y}_1$ or $\hat{y} > \hat{y}_n$, we set $\hat{F}(z|\hat{y})$ to $\hat{F}_1(z)$ or $\hat{F}_n(z)$, respectively. Finally, since ProbcastSeries stores predictive distributions in the form of quantiles, we determine quantiles at specified levels as

$\hat{q}_{\tau|\hat{y}} = \min\{z : \hat{F}(z|\hat{y}) \geq \tau\}.$

The multivariate version of IDR is not supported, but ForecastSeries containing multiple forecasts can be used as input for computing ProbcastSeries. In such a case, multiple univariate IDR models are estimated and the resulting distributions functions $\hat{F}(z)$ are averaged. Since $z$ is limited to true values of the timeseries in training window, the distributions resulting from estimated IDRs are defined at the exact same points, which allows to efficiently and precisely compute the average across probability. 

The implemented IDR estimation uses abridged pool-adjacent-violators algorithm introduced by [Henzi et al. (2022)](https://doi.org/10.1007/s11009-022-09937-2)

```@docs
IDR
getcdf
getx
gety
```

## Quantile regression
Quantile Regression Averaging [(QRA; Nowotarski and Weron, 2014)](https://doi.org/10.1007/s00180-014-0523-0) is a well-established method in for obtaining probabilistic forecasts of electricity prices and load. It learns conditional quantiles as linear combination of $m$ point forecasts:

$\hat{q}_{\tau|\hat{y}^{(1)}, ..., \hat{y}^{(m)}} = \beta^{(\tau)}_0 + \beta^{(\tau)}_1\hat{y}^{(1)} + ... + \beta^{(\tau)}_m\hat{y}^{(m)}.$

The coefficients $\beta^{(\tau)}_{0...m}$ are selected to minimize the pinball loss on the training window and estimated by solving a linear programming problem. For this task, Probcasts.jl employs [JuMP.jl](https://jump.dev/JuMP.jl/stable/) and HiGHS.jl packages. Different LP solvers compatible with JuMP can be used, but the constructor defaults to an open source [HiGHS](https://highs.dev).

Apart from the standard QRA introduced by [Nowotarski and Weron (2014)](https://doi.org/10.1007/s00180-014-0523-0), **PostForecasts.jl** allows to readily compute Quantile Regression Machine [(QRM; Marcjasz et al., 2020)](https://doi.org/10.1016/j.ijforecast.2019.07.002) and Quantile Regression with probability (F) or Quantile averaging [(QRF or QRQ; Uniejewski et al., 2019)][https://doi.org/10.1016/j.eneco.2018.02.007]. See [*Different flavors of quantile regression*](https://lipiecki.github.io/PostForecasts.jl/dev/examples/#Different-flavors-of-quantile-regression) for details.

```@docs
QR
getweights
getquantprob
```

## Training and prediction
```@docs
train
predict
predict!
```
