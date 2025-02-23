# Datasets
**PostForecasts.jl** ships with several datasets for evaluating probabilistic forecasts, the detailed description along with the source is provided below.

## EPEX

### Data
The EPEX dataset consists of point forecasts for 24 time series of electricity prices on the German day-ahead energy market. Each time series corresponds to different hour of the day. The forecasts span a 5-year period starting from 2019. Electricy prices and data used for generating point forecats are sourced from [ENTSO-E](https://transparency.entsoe.eu) transparency platform.

### LEAR forecasts
Point forecasts were generated using a LASSO-Estimated AutoRegressive (LEAR) model used in [(Lipiecki et al., 2024)](https://doi.org/10.1016/j.eneco.2024.107934). The regressors include past prices, day-ahead predictions of the system-wide load, day-ahead RES generation, four macroeconomic variables (carbon emission prices, natural gas prices, crude oil prices and coal prices). The parameters are estimated separately for each of the 4 training window lengths, i.e., 56, 84, 1092 and 1456 most recent days and employ LASSO with cross-validation.

### Loading forecasts
To load the forecasts from the EPEX dataset, you can use the `loadDataset` function. For example, to get the `ForecastSeries` corresponding to the first hour of the day, run:

```julia
fs = loaddata(:epex1)
```

### File example
`epex_hour0.csv`
```csv
date,price,lear56,lear84,lear1092,lear1456
20181227,47.41,46.02598302679128,45.54356818616377,47.03463193816627,46.56173490550826
20181228,50.04,50.618405313962285,51.29104818479025,50.82694184883733,51.254485969448226
20181229,56.69,49.53671919852577,47.554433720562926,44.578191549767155,45.50325636133202
...
```

## PANGU
Forecasts from the Pangu-Weather model, computed by [Bülte et al. (2024)](https://arxiv.org/abs/2403.13458). The model is trained on 39 years of ERA5 reanalysis data from 1979–2017. The dataset consists of 5 weather variables (listed below) for Wrocław, Poland between 2018 and 2022.
- **U10**: u-component of 10-m wind speed
- **V10**: v-component of 10-m wind speed
- **T2M**: temperature at 2m
- **T850**: temperature at 850 hPa
- **Z500**: geopotential height at 500 hPa.

One model run is initialized each day at midnight and used to forecast the variables for up to 186 hours ahead, with 6-hour resolution.

The dataset is partitioned into 32 files, which correspond to different forecasting horizons (lead times).

The verifying observations are sourced from the ERA5 reanalysis model.

### Loading forecasts
To load the forecasts from the PANGU dataset, you can use the `loadDataset` function. For example, to get the `ForecastSeries` of T2M predictions with forecast horizon of 24 hours, run:

```julia
fs = loadDataset(:pangu24t2m)
```

### File example
`pangu_lead0.csv`
```csv
timestamp,u10_pred,v10_pred,t2m_pred,t850_pred,z500_pred,u10,v10,t2m,t850,z500
2018010100,2.8765717,4.403717,281.96204,277.08377,54057.855,2.8765192,4.403531,281.9626,277.0831,54057.844
2018010200,-0.12825012,3.162384,275.21405,271.42078,52858.676,-0.12847205,3.1627147,275.21313,271.42123,52858.73
2018010300,2.0507965,1.4558716,275.82217,268.6925,52679.027,2.0507119,1.4563175,275.8231,268.69275,52679.008
...
```
## Acknowledgements
We thank our collaborator [Bartosz Uniejewski](https://scholar.google.pl/citations?user=t3QHuHEAAAAJ&hl) for LEAR foreacsts of electricity prices, and [Sebastian Lerch](https://sites.google.com/site/sebastianlerch/) from KIT for sharing the dataset of weather forecasts for Wrocław.
