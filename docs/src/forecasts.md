# Forecasts structures

`Forecasts` type is a supertype that spans `PointForecasts` and `QuantForecast` types.

## PointForecasts
`PointForecasts` is a structure intended for storing the series of point `pred`ictions (both single pred and pools of forecasts), along with the `obs`ervations and `id`entifiers (timestamps). The package provides functions for building `PointForecasts` objects from delimited files, calculating error measures (MAE and RMSE) and averaging point pred.

## QuantForecasts
`QuantForecasts` struct is inteded for storing the series of probabilistic `pred`ictions, represented as quantiles of predictive distribution corresponding to `prob`ability levels, along with the `obs`ervations and `id`entifiers (timestamps). The package provides functions for computing probabilstic forecasts from `PointForecasts` objects, calculating pinball loss and averaging distributions across quantiles or probabilities.

## Indexing and slicing
`PointForecasts` and `QuantForecasts` support both scalar indexing and slicing. Accessing a series with a scalar indexes results in a named tuple, while slicing creates a new series object built from pred,observations and identifiers stored at respective indices.

```julia
pred = loaddata(:epex_hour1)
firstday = pred[1]
firstweek = pred[1:7]
```

## Identifier-based indexing
Since `PointForecasts` and `QuantForecasts` structures have `id` field storing an integer identifier for every timestep, it is posibble to access its elements by providng its identifier value, in the spirit of Pandas' `.loc`. Use `()` for identifier-based indexing. Analogously to indexing and slicing, a single identifier results in a named tuple, while a vector creates a new series object.

```julia
pred = loaddata(:epex_hour1)
firstday = pred(20190101)
firstweek = pred([20190101, 20190102, 20190103, 20190104, 20190105, 20190106, 20190107])
```

```@docs
PointForecasts
QuantForecasts
length
getindex
firstindex
lastindex
eachindex
findindex
decouple
npred
setpred!
getpred
getobs
getid
getprob
viewpred
viewobs
viewid
viewprob
```