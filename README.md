# PostForecasts.jl
[![View Docs](https://img.shields.io/badge/View-Docs-blue)](https://lipiecki.github.io/PostForecasts.jl/)
[![codecov](https://codecov.io/github/lipiecki/PostForecasts.jl/graph/badge.svg?token=JJDOKDJ30H)](https://codecov.io/github/lipiecki/PostForecasts.jl)

![PostForecasts.jl](https://github.com/lipiecki/PostForecasts.jl/blob/main/docs/src/images/banner.png?raw=true)

## Julia package for postprocessing forecasts
**PostForecasts.jl** provides structures and functions that allow to easily postprocess point forecasts into predictive distributions. Postprocessing methods use only the past performance of a given point forecasting model (or ensemble of models) to build probabilistic forecasts conditional on point predictions.

**PostForecasts.jl:**

- **Relies on deterministic models for reliable and repeatable results**

- **Does not require hyperparameter tuning**

- **Leverages forecaster diversity via averaging**

We believe that following these three principles allowed us to develop a robust tool for computing probabilistic forecasts that combines ease of use, high accuracy, fast results and good interpretability. This makes **PostForecasts.jl** an attractive choice for both academic and industrial applications.

## Quick start

### Dedicated types
**PostForecasts.jl** introduces `PointForecasts` and `QuantForecasts` types for storing time series data along with point and probabilistic forecasts respectively.

### Postprocessing models
The package provides interface to four selected models for probabilistic forecasting:
- `Normal`:  Normal error distribution
- `CP`: Conformal Prediction (and Historical Simulation)
- `IDR`: Isotonic Distributional Regression
- `QR`: Quantile Regression

The models belong to the `PostModel` type, they have corresponding `train` methods for calibrating the model to the provided data and `predict` methods yielding quantiles of predictive distribution.

### Easy postprocessing
The core functionality of **PostForecasts.jl** is building probabilistic forecasts from point predictions. For easy postprocessing, use `point2quant` function to turn `PointForecasts` into `QuantForecasts` with one of the implemented models.

In addition to methods for postprocessing point forecasts, the package provides `conformalize` function, allowing to correct quantile forecasts using historical errors.

### Installation
**PostForecasts.jl** is a registered Julia package, to install it just run the following lines in the Julia REPL:

```julia
julia> using Pkg
julia> Pkg.add("PostForecasts")
```

Alternatively, you can use the Pkg REPL mode:
```julia
pkg> add PostForecasts
```
