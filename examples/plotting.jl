using Plots

"""
    plot_obs!(plt, f::Forecasts; kwargs...)
Plot the observations from `f::Forecasts` on `plt`.
"""
function plot_obs!(plt, f::Forecasts; kwargs...)
    kwargs=Dict{Symbol, Any}(kwargs)
    kwargs[:st] = haskey(kwargs, :st) ? kwargs[:st] : :scatter
    kwargs[:msw] = haskey(kwargs, :msw) ? kwargs[:msw] : 0
    kwargs[:label] = haskey(kwargs, :label) ? kwargs[:label] : nothing
    plot!(plt, viewobs(f); kwargs...)
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
