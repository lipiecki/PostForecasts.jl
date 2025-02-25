# Postprocessing

## From point to probabilistic forecasts
Building probabilistic forecasts from point predictions is the core functionality of **PostForecasts.jl**. The function `point2quant` turns `PointForecasts` into `QuantForecasts`, allowing to easily postprocess point predictions using a selected method, length of the training window and retraining frequency. See [Models](models.md#Models) for details on the available postprocessing methods.

```@docs
point2quant
```

## Conformalizing probabilistic forecasts
Apart from postprocessing point forecasts, the package offers postprocessing of probabilistic forecasts in a form of conformalization [(Romano et al., 2019)](https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf). Conformalizing quantiles is performed by adjusting the prediction of each quantile according to the formula $\hat{q}^{(c)}_{\tau} = \hat{q}_{\tau} + Q_{1 - \tau}(\lambda)$, where $Q_{1 - \tau}(\lambda)$ is the $(1-\tau)$-th sample quantile of non-conformity scores $\lambda_i := y_i - \hat{q}_{i,\tau}$ from the training window. See an example on [Conformalizing weather forecasts](examples.md#Conformalizing-weather-forecasts).

```@docs
conformalize
conformalize!
```
