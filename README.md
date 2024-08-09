# PostForecasts.jl

[![Static Badge](https://img.shields.io/badge/view-docs-blue)](https://lipiecki.github.io/PostForecasts.jl/)
[![codecov](https://codecov.io/github/lipiecki/PostForecasts.jl/graph/badge.svg?token=JJDOKDJ30H)](https://codecov.io/github/lipiecki/PostForecasts.jl)

**Welcome to PostForecasts.jl, a Julia package for postprocessing point predictions into probabilistic forecasts.**

The package provides structures and functions that allow to easily compute predictive distributions conditional on point forecasts. Probabilistic forecasts are trained on the history of point forecasts only, i.e. pairs $(\hat{y}_t, y_t)$, where $\hat{y}_t$ is either a single point prediction or a vector of point predictions, and $y_t$ is the observed value of the timeseries at moment $t$.

**PostForecasts.jl** implements four postprocessing models that differ in terms of computational complexity and assumptions about the underlying distributions. The main advantages of our package:

- **Relies on deterministic models for reliable and repeatable results**

- **Does not require hyperparameter tuning**

- **Leverages forecaster diversity via averaging**

We believe that following these three principles allowed us to develop a robust tool for computing probabilistic forecasts that combines ease of use, high accuracy, fast results and good interpretability of the results. This makes Probcasts.jl an attractive choice for both academic and industrial applications.

## Dedicated Structures
**PostForecasts.jl** provides `PointForecasts` and `QuantForecasts` structures for storing time series data along with point and probabilistic forecasts respectively.

## From point to probabilistic forecasts
The core functionality of the package is `point2prob` function that builds probabilistic forecasts from point predictions. It creates a `QuantSeries` from `PointSeries`, using a selected model, training window length and quantile levels.

## Models
The package provides interface to four selected models for probabilistic forecasting:
- `CP`: Conformal Prediction
- `IDR`: Isotonic Distributional Regression
- `QR`: Quantile Regression
- `Normal`: Normal distribution of forecast errors

Each model belongs to the `ProbModel` type and has a corresponding `train` method for calibrating the model to the provided data; and `predict`/`predict!` methods yielding quantiles of predictive distribution based on point forecast(s).

## Conformalizing Quantile Forecasts
In addition to methods for postprocessing point forecasts, **PostForecasts.jl** provides `conformalize` function, allowing to correct the quantile forecasts using historical errors.

## Installation

To install PostForecasts.jl package, enter Julia REPL and call `Pkg.add`:

```julia
julia> using Pkg
julia> Pkg.add(url = "https://github.com/lipiecki/PostForecasts.jl")
```