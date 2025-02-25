# Loading and saving forecasts
With **PostForecasts.jl** You can easily create `PointForecasts` from delimited files, load and save both `PointForecasts` and `QuantForecast` using HDF5 format and play with pre-installed datasets.

To make managing files generated with **PostForecasts.jl** easier, HDF5 files containing `PointForecasts` and `QuantForecasts` are saved with `.pointf` and `.quantf` extensions respectively.

```@docs
loaddata
loaddlm
saveforecasts
loadforecasts
loadpointf
loadquantf
```
