"""
    conformalize(qf::QuantForecasts{F, I}; window::Integer[, start, stop)
Perform conformalization of quantile forecasts provided in `qf`.
Conformalized quantiles will be calculated for observations between the `start` and `stop` `id`entifiers in `qf`. The model is retrained every step on the last `window` observations.

Return `QuantForecasts` with conformalized quantiles.
"""
function conformalize(qf::QuantForecasts{F, I}; window::Integer, start::Union{Nothing, Integer}=nothing, stop::Union{Nothing, Integer}=nothing) where {F, I} 
    model = CP(window, abs=false)
    first = isnothing(start) ? firstindex(qf)+window : findindex(qf, start)
    last = isnothing(stop) ? lastindex(qf) : findindex(qf, stop)
    first > window || throw(ArgumentError("there is less than $(window) timesteps before $(start)"))
    pred = getpred(qf, first:last)
    for t in first:last
        for i in 1:npred(qf)
            train(model, viewpred(qf, t-window:t-1, i), viewobs(qf, t-window:t-1))
            pred[t-first+1, i] = _predict(model, getpred(qf, t, i), getprob(qf, i))
        end
    end
    sort!(pred, dims=2)
    return QuantForecasts(
        pred,
        getobs(qf, first:last),
        getid(qf, first:last),
        getprob(qf)
    )
end

"""
    conformalize!(qf::QuantForecasts{F, I}; window::Integer[, start, stop)
In-place version of conformalize that mutates `qf` instead of creating a new `QuantForecasts`.
"""
function conformalize!(qf::QuantForecasts{F, I}; window::Integer, start::Union{Nothing, Integer}=nothing, stop::Union{Nothing, Integer}=nothing) where {F, I}
    model = CP(window, abs=false)
    first = isnothing(start) ? firstindex(qf)+window : findindex(qf, start)
    last = isnothing(stop) ? lastindex(qf) : findindex(qf, stop)
    first > window || throw(ArgumentError("there is less than $(window) timesteps before $(start)"))
    for t in last:-1:first
        for i in 1:npred(qf)
            train(model, viewpred(qf, t-window:t-1, i), viewobs(qf, t-window:t-1))
            setpred(qf, t, i, _predict(model, getpred(qf, t, i), getprob(qf, i)))
        end
        sort!(viewpred(qf, t))
    end
end