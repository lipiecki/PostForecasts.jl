using PostForecasts
include("plotting.jl")
theme(:dark)

pfbuy = loaddata(Symbol(:epex, 4))      # point forecasts for 3am
pfsell = loaddata(Symbol(:epex, 20))    # point forecasts for 7pm

qfbuy = point2quant(pfbuy, method=:idr, window=182, quantiles=9, start=20230408, stop=20230421)
qfsell = point2quant(pfsell, method=:idr, window=182, quantiles=9, start=20230408, stop=20230421)

plt = plot(legend=:bottom, xlabel="Days", ylabel="Price (EUR/MWh)", xticks=1:14, framestyle=:box) 
plot_intervals!(plt, qfsell, color=1)
plot_intervals!(plt, qfbuy, color=3)
plot_quantile!(plt, qfsell, 5, color=1)
plot_quantile!(plt, qfbuy, 5, color=3)
plot_obs!(plt, qfsell, color=1, label="Sell price")
plot_obs!(plt, qfbuy, color=3, label="Buy price")
