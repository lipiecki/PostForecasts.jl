using PostForecasts, Plots
theme(:dark, palette = theme_palette(:dark))

"""
    plot_obs!(plt, fs::Forecasts; kwargs...)
Plot the observations from a `Forecasts` object `fs` on `plt`.
"""
function plot_obs!(plt, fs::Forecasts; kwargs...)
    kwargs=Dict{Symbol, Any}(kwargs)
    kwargs[:st] = haskey(kwargs, :st) ? kwargs[:st] : :scatter
    kwargs[:msw] = haskey(kwargs, :msw) ? kwargs[:msw] : 0
    kwargs[:label] = haskey(kwargs, :label) ? kwargs[:label] : nothing
    plot!(plt, viewobs(fs); kwargs...)
end

"""
    plot_quantile!(plt, qf::QuantForecasts, quantile::Integer; kwargs...)
Plot the prediction at a given `quantile` level from a `QuantForecasts` object `qf` on `plt`.
"""
function plot_quantile!(plt, qf::QuantForecasts, quantile::Integer; kwargs...)
    kwargs=Dict{Symbol, Any}(kwargs)
    kwargs[:lw] = haskey(kwargs, :lw) ? kwargs[:lw] : 2
    kwargs[:label] = haskey(kwargs, :label) ? kwargs[:label] : nothing
    plot!(plt, viewpred(qf, eachindex(qf), quantile); kwargs...)
end

"""
    plot_intervals!(plt, qf::QuantForecasts; kwargs...)
Plot the prediction intervals from a `QuantForecasts` object `qf` on `plt`.
"""
function plot_intervals!(plt, qf::QuantForecasts; kwargs...)
    kwargs=Dict{Symbol, Any}(kwargs)
    kwargs[:lw] = haskey(kwargs, :lw) ? kwargs[:lw] : 0.0
    kwargs[:fa] = haskey(kwargs, :fa) ? kwargs[:fa] : 0.15
    kwargs[:label] = haskey(kwargs, :label) ? kwargs[:label] : nothing
    if npred(qf) % 2 == 0
        for i in 1:Int(npred(qf)/2)
            plot!(plt, viewpred(qf, eachindex(qf), i),
                    fillrange=viewpred(qf, eachindex(qf), npred(qf)-i+1); kwargs...)
        end
    else
        central_quantile = Int((npred(qf)-1)/2) + 1
        for i in 1:(central_quantile-1)
            plot!(plt, viewpred(qf, eachindex(qf), central_quantile-i), 
                    fillrange=viewpred(qf, eachindex(qf), central_quantile+i); lw=0, kwargs...)
        end
    end
end


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
