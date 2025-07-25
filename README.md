# PostForecasts.jl
[![docs stable](https://img.shields.io/badge/docs-stable-blue)](https://lipiecki.github.io/PostForecasts.jl/)
[![docs dev](https://img.shields.io/badge/docs-dev-blue)](https://lipiecki.github.io/PostForecasts.jl/dev)
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

All the models belong to the `PostModel` supertype. They have corresponding `train(m, X, Y)` methods for calibrating a model `m` to inputs `X` and targets `Y`, and `predict(m, input, quantiles)` methods that yield the specified `quantiles` of the predictive distribution from `m` conditional on `input`.

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

### Citing
Check out the original software publication on **PostForecasts.jl** in [SoftwareX](https://doi.org/10.1016/j.softx.2025.102200). If you use the package in your research, then please cite it as:
```bibtex
@article{lipiecki:weron:2025,
    title = {PostForecasts.jl: A Julia package for probabilistic forecasting by postprocessing point predictions},
    journal = {SoftwareX},
    volume = {31},
    pages = {102200},
    year = {2025},
    issn = {2352-7110},
    doi = {https://doi.org/10.1016/j.softx.2025.102200},
    author = {Arkadiusz Lipiecki and Rafał Weron}
}
```

### Research papers
Below is the list of research papers using **PostForecasts.jl**:
- A. Lipiecki, B. Uniejewski & R. Weron *Postprocessing of point predictions for probabilistic forecasting of day-ahead electricity prices: The benefits of using isotonic distributional regression* [Energy Economics, 139 (2024) 107934](https://doi.org/10.1016/j.eneco.2024.107934)
- A. Lipiecki & B. Uniejewski *Isotonic Quantile Regression Averaging for uncertainty quantification of electricity price forecasts* [arXiv.2507.15079](https://doi.org/10.48550/arXiv.2507.15079) ([GitHub](https://github.com/lipiecki/isotonicQRA))
