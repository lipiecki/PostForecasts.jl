# Datasets
**PostForecasts.jl** ships with two datasets of point forecasts for testing the functionality of the package, the detailed description along with the source is provided below.

## EPEX
The EPEX dataset consists of hourly forecasts of wholesale electricity prices in Germany, as well as the corresponding day-ahead point forecasts computed by [Lipiecki et al. (2024)](https://doi.org/10.1016/j.eneco.2024.107934) using a LASSO-Estimated AutoRegressive (LEAR) model [(Lago et al., 2021)](https://doi.org/10.1016/j.apenergy.2021.116983). The regressors used in the LEAR model include: 
- historical prices of electricity (various lags), 
- day-ahead predictions of the system-wide load,  
- day-ahead predictions of wind and solar generation.

The parameters are estimated separately for each of the 4 training window lengths, i.e., 56, 84, 1092 and 1456 most recent days, and employ cross-validation for selecting a regularization penalty. The dataset is partitioned into 24 time series corresponding to different hours of the day and spans a 5-year period (2019-2023). The following snippet loads the `PointForecasts` of EPEX prices for the 20-th trading hour of the day-ahead market (19:00):
```julia
pf::PointForecasts = loaddata(:epex20)
```

## PANGU
The PANGU dataset contains forecasts from the Pangu-Weather model, computed by [Bülte et al. (2024)](https://arxiv.org/abs/2403.13458). The model is trained on 39 years of ERA5 reanalysis data from 1979–2017. The dataset consists of 5 weather variables (listed below) for Wrocław, Poland between 2018 and 2022.
- **U10**: u-component of 10-m wind speed
- **V10**: v-component of 10-m wind speed
- **T2M**: temperature at 2m
- **T850**: temperature at 850 hPa
- **Z500**: geopotential height at 500 hPa.

One model run is initialized each day at midnight and used to forecast the variables for up to 186 hours ahead, with 6-hour resolution. The dataset is partitioned into 32 files, which correspond to different forecasting horizons (lead times). The verifying observations are sourced from the ERA5 reanalysis model. The following snippet loads the `PointForecasts` of temperature at 850 hPa with a lead time of 24 hours:
```julia
pf::PointForecasts = loaddata(:pangu24t850)
```

## Acknowledgements
We thank: 
- [Sebastian Lerch](https://sites.google.com/site/sebastianlerch/) from KIT for preparing the PANGU dataset of weather forecasts for Wrocław
- [Bartosz Uniejewski](https://scholar.google.pl/citations?user=t3QHuHEAAAAJ&hl) for generating LEAR forecasts of electricity prices for the EPEX dataset
