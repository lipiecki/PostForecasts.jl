# Models

Point2Prob.jl provides four model for postprocessing point predictions for probabilistic forecasts: 
- Normal Errors
- Conformal Prediction
- Isotonic Distributional Regression
- Quantile Regression Averaging.

## Normal Errors
The naive model for probabilistic forecasting, which assumes normally dsitributed errors of input point forecasts. 

The predictive distribution conditional on point forecast $\hat{y}$ is a Gaussian $\mathcal{N}(\mu + \hat{y}, \sigma)$, where $\mu$ and $\sigma$ are mean and sample standard deviation of errors in the calibration window.

The $\tau$-th quantile conditional on $\hat{y}$ of such parameterized distribution can be obtained via an analytic expression:

$\hat{q}_{\tau|\hat{y}} = \hat{y} + \mu + \sigma \sqrt{2} \cdot \text{erf}^{-1} (2\tau - 1).$

```@docs
Normal
getmean
getstd
```

## Conformal Prediction
Conformal prediction is a machine learning framework for computing prediction intervals based on the outputs of an arbirary point forecasting model. The implemented version of Conformal Prediction is analogous to the inductive approach used by [Kath and Ziel (2021)](https://doi.org/10.1016/j.ijforecast.2020.09.006). 

In the training step, the non-conformity scores $\lambda_i$ are calculated on the calibration set $(\hat{y}_i, y_i)_{i\in\text{calibration window}}$ as $\lambda_i := |\hat{y}_i - y_i|$.

In the prediction step, the $\tau$-th quantile conditional on $\hat{y_t}$ is obtained by shifting the prediction by an appropriate empirical quantile of non-conformity scores:

$\hat{q}_{\tau|\hat{y}} = \hat{y} - \mathbf{1}_{\{\tau \leq 0.5\}} Q_{2\tau}(\lambda) + \mathbf{1}_{\{\tau > 0.5\}} Q_{2(1 - \tau)}(\lambda),$

where $Q_{\alpha}(\lambda)$ is the $\alpha$-th empirical quantile of non-conformity scores from the calibration window. Although the intervals in the form of $[\hat{y} - Q_{\alpha}(\lambda), \hat{y} +Q_{\alpha}(\lambda)]$ are valid $(1-\alpha)$ prediction intervals without any requirements on the underlying distribution, translating them into quantiles requires the assumption of symmetrically distributed errors.

```@docs
CP
getscores
```

## Isotonic Distributional Regression
[Isotonic distributional regression](https://doi.org/10.1111/rssb.12450) has been recently introduced as a nonparametric method for estimating distributions that are isotonic in the regressed variable, which means that the quantiles of such distributions are non-decreasing w.r.t the regressor. In the calibration step, $n$ observations $(\hat{y}_i, y_i)_{i \in \text{calibration window}}$ are first sorted to be ascending in $\hat{y}_i$. Then, $n$ conditional distributions $\hat{F_i}(z) = \hat{F}(z|x_i)$ are obtained by solving the following min-max problem via abridged pool-adjacent-violators algorithm:
 
$\hat{F}_i(z) = \min_{k=1,...,i} \max_{j=k,..,n} \frac{1}{j-k+1}\sum_{l=k}^{j} \mathbb{1}\{y_{l} < z\},$

where $z \in (y_i)_{i \in \text{calibration window}}$
To obtain conditional distribution for any $\hat{y}\in\mathbb{R}$, the obtained distribution functions are interpolated

$\hat{F}(z|\hat{y}) = \frac{\hat{y}-\hat{y}_i}{\hat{y}_{i+1} - \hat{y}_i}\hat{F}_i(z) + \frac{\hat{y}_{i+1} - \hat{y}}{\hat{y}_{i+1} - \hat{y}_i} \hat{F}_{i+1}(z),$

If $\hat{y} < \hat{y}_1$ or $\hat{y} > \hat{y}_n$, we set $\hat{F}(z|\hat{y})$ to $\hat{F}_1(z)$ or $\hat{F}_n(z)$, respectively. Finally, since ProbcastSeries stores predictive distributions in the form of quantiles, we determine quantiles at specified levels as

$\hat{q}_{\tau|\hat{y}} = \min\{z : \hat{F}(z|\hat{y}) \geq \tau\}.$

The multivariate version of IDR is not supported, but ForecastSeries containing multiple forecasts can be used as input for computing ProbcastSeries. In such a case, multiple univariate IDR models are estimated and the resulting distributions functions $\hat{F}(z)$ are averaged. Since $z$ is limited to true values of the timeseries in calibration window, the distributions resulting from estimated IDRs are defined at the exact same points, which allows to efficiently and precisely compute the average across probability. 

```@docs
IDR
getcdf
getx
gety
```

## Quantile Regression Averaging
[Quantile Regression Averaging](https://doi.org/10.1007/s00180-014-0523-0) is a well-established method in for obtaining probabilistic forecasts of electricity prices and load. It learns conditional quantiles as linear combination of $m$ point forecasts:

$\hat{q}_{\tau|\hat{y}^{(1)}, ..., \hat{y}^{(m)}} = \beta^{(\tau)}_0 + \beta^{(\tau)}_1\hat{y}^{(1)} + ... + \beta^{(\tau)}_m\hat{y}^{(m)}.$

The coefficients $\beta^{(\tau)}_{0...m}$ are selected to minimize the pinball loss on the calibration window and estimated by solving a linear programming problem. For this task, Probcasts.jl employs [JuMP.jl](https://jump.dev/JuMP.jl/stable/) and HiGHS.jl packages. Different LP solvers compatible with JuMP can be used, but the constructor defaults to an open source [HiGHS](https://highs.dev).

```@docs
QR
getweights
```

## Training and prediction

```@docs
train
predict
predict!
```