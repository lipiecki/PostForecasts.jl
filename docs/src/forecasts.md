# Forecasts structures
To make working with forecasts easy and user-friendy, **PostForecasts.jl** introduces `Forecasts` type, a supertype that spans `PointForecasts` and `QuantForecast` structures.

## PointForecasts
`PointForecasts` is a structure designed for storing the series of point `pred`ictions (single predictions or prediction pools), along with the `obs`ervations and `id`entifiers (timestamps). The package provides functions for building `PointForecasts` objects from delimited files, calculating error measures (MAE and RMSE) and averaging point pred.

## QuantForecasts
`QuantForecasts` is a structure designed for storing the series of probabilistic `pred`ictions, represented as quantiles of predictive distribution corresponding to `prob`ability levels, along with the `obs`ervations and `id`entifiers (timestamps). The package provides functions for computing probabilstic forecasts from `PointForecasts` objects, calculating pinball loss and averaging distributions across quantiles or probabilities.

## Position-based indexing and slicing
`PointForecasts` and `QuantForecasts` support both position-based indexing and slicing. Accessing a series with a scalar index results in a named tuple, while slicing creates a new `Forecasts` object built from pred, observations and identifiers stored at respective indices.

```julia
pf = loaddata(:epex1);
firstday = pf[1]
#(pred = [27.640966097698737, 24.423563275081627, 23.54144377224293, 25.061033846927558], obs = 10.07, id = 20190101)
firstweek = pf[1:7]
#PointForecasts{Float64, Int64} with a pool of 4 forecasts at 7 timesteps, between 20190101 and 20190107
```

## Label-based indexing
Since `PointForecasts` and `QuantForecasts` structures have `id` field storing an integer identifier for every timestep, it is posibble to access its elements by providng its `id`entifier value, much in the spirit of Pandas `.loc`. Use `()` for label-based indexing. Analogously to indexing and slicing, providing a single label results in a named tuple, while a vector creates a new `Forecasts` object. Additionally, you can provide two labels, `(arg1, arg2)`, to return `Forecasts` starting at the timestep with identifier `arg1` and ending at the timestep with identifier `arg2`.

```julia
pf = loaddata(:epex1);
firstday = pf(20190101)
#(pred = [27.640966097698737, 24.423563275081627, 23.54144377224293, 25.061033846927558], obs = 10.07, id = 20190101)
firstweek = pf([20190101, 20190102, 20190103, 20190104, 20190105, 20190106, 20190107])
#PointForecasts{Float64, Int64} with a pool of 4 forecasts at 7 timesteps, between 20190101 and 20190107
firstweek2 = pf(20190101, 20190107) # same as `firstweek`
#PointForecasts{Float64, Int64} with a pool of 4 forecasts at 7 timesteps, between 20190101 and 20190107
```

```@docs
PointForecasts
QuantForecasts
findindex
decouple
npred
setpred
getpred
getobs
getid
getprob
viewpred
viewobs
viewid
viewprob
```
