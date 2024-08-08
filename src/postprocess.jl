"""
    point2prob(pf::PointForecasts, window::Integer, modelname::Symbol, prob[; first = window + 1, last = length(pf), recalibration = 1])
Compute probabilistic forecast based on point forecasts `pf`. Probabilistic forecast will be calculated for observations between the index `first` and `last` of `pf`. The model is calibrated on the last `window` observations, and recalibrated every `recalibration` steps.

Available options for `model`:
- `:qr` for Quantile Regression Averaging
- `:cp` for Conformal Prediction
- `:hs` for Conformal Prediction with non-symmetric errors (a.k.a. Historical Simulation)
- `:idr` for Isotonic Distributional Regression
- `:normal` for Normal distribution of errors
- `:zeronormal` for Normal distribution of errors with fixed mean equal to 0.

`QR` supports multiple regressors; `IDR` partially supports multiple regressors - one isotonic regression is fitted to each forecast and the final predictive distribution is an average of individual distributions; `CP` and `Normal` do not support multiple regressors.

Return QuantForecasts containing quantile forecasts at specified probabilities `prob` (vector of probabilities `::AbstractVector{<:AbstractFloat}`, single probability value `::AbstractFloat` or the number of equidistant probability values `::Integer`).
"""
function point2prob(pf::PointForecasts{F, I}, window::Integer, modelname::Symbol, prob::AbstractVector{F}; first::Integer = window + firstindex(pf), last::Integer = lastindex(pf), recalibration::Integer = 1) where {F, I}
    if !issorted(prob)
        sort!(prob)
        @warn "sorting `prob` vector"
    end
    quantiles = zeros(F, last-first+1, length(prob))
    model = getmodel(Val(modelname), window, npred(pf), prob)
    if nreg(model) == 1 && npred(pf) > 1
        pf = average(pf)
    end
    for t in first:last
        if t == first || (recalibration > 0 && (t - first) % recalibration == 0)
            _train(model, viewpred(pf, t-window:t-1), viewobs(pf, t-window:t-1))
        end
        _predict!(model, @view(quantiles[t-first+1, :]), viewpred(pf, t), prob)
    end
    return QuantForecasts(
        quantiles,
        getobs(pf, first:last),
        getid(pf, first:last),
        Vector(prob))
end

point2prob(pf::PointForecasts{F, I}, window::Integer, modelname::Symbol, prob::F; kwargs...) where {F, I} = 
    point2prob(pf, window, modelname, [prob]; kwargs...)

point2prob(pf::PointForecasts{F, I}, window::Integer, modelname::Symbol, prob::Integer; kwargs...) where {F, I} = 
    point2prob(pf, window, modelname, equidistant(prob, F); kwargs...)

"""
    conformalize(qf::QuantForecasts{F, I}, window::Integer[; first::Integer, last::Integer)
Perform [conformalization](https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf) of quantile forecasts provided in `ps`.
Conformalized quantiles will be calculated for observations between the index `first` and `last` of `qf`. The model is calibrated on the last `window` observations.

Return `QuantForecasts` with conformalized quantiles.
"""
function conformalize(qf::QuantForecasts{F, I}, window::Integer; first::Integer=window+firstindex(qf), last::Integer=lastindex(qf)) where {F, I} 
    quantiles = getpred(qf, first:last)
    model = CP(window, abs=false)
    for t in first:last
        for i in 1:npred(qf)
            train(model, viewpred(qf, t-window:t-1, i), viewobs(qf, t-window:t-1))
            quantiles[t - first + 1, i] = _predict(model, quantiles[t - first + 1, i], 1.0 - getprob(qf, i))
        end
    end
    sort!(quantiles, dims = 2)
    return QuantForecasts(
        quantiles,
        getobs(qf, first:last),
        getid(qf, first:last),
        getprob(qf))
end

"""
    conformalize!(qf::QuantForecasts{F, I}, window::Integer[; first::Integer, last::Integer)
In-place version of conformalize, that mutates `qf` instead of creating a new `QuantForecasts`.
"""
function conformalize!(qf::QuantForecasts{F, I}, window::Integer; first::Integer=window+firstindex(qf), last::Integer=lastindex(qf)) where {F, I}
    model = CP(window, abs=false) 
    for t in last:-1:first
        for i in 1:npred(qf)
            train(model, viewpred(qf, t-window:t-1, i), viewobs(qf, t-window:t-1))
            setpred(qf, t, i, _predict(model, getpred(qf, t, i), 1.0 - getprob(qf, i)))
        end
        sort!(viewpred(qf, t))
    end
end
