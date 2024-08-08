# Conformalized Quantile Forecasts

Apart from postprocessing point forecasts, the package offers postprocessing of probabilistic forecasts in a form of [conformalization](https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf) scheme. Conformalizing quantiles is performed by adjusting the prediction of each quantile according to the formula $\hat{q}^{(conf)}_{\tau} = \hat{q}_{\tau} + Q_{1 - \tau}(\lambda)$, where $Q_{1 - \tau}(\lambda)$ is the $(1-\tau)$-th empirical quantile of non-conformity scores $\lambda_i := y_i - \hat{q}_{i,\tau}$ from the training window.

**To-do: show an example**

```@docs
conformalize
conformalize!
```