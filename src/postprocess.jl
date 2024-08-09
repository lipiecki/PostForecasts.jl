"""
    point2prob(pf, modelname, window, prob[; first, last, retrain])
Compute probabilistic forecast based on point forecasts `pf::PointForecasts` using model `m::ProbModel`.

Return `QuantForecasts` containing quantile pred at specified `prob`abilities:
- `prob::AbstractVector{<:AbstractFloat}`: vector of probabilities
- `prob::AbstractFloat`: a single probability value
- `prob::Integer`: number of equidistant probability values (e.g. 99 for percentiles).

## Available options for `modelname`:
- `:qr` for Quantile Regression Averaging
- `:cp` for Conformal Prediction with absolute errors
- `:hs` for Conformal Prediction with non-absolute errors (a.k.a. Historical Simulation)
- `:idr` for Isotonic Distributional Regression
- `:normal` for Normal distribution of errors
- `:zeronormal` for Normal distribution of errors with fixed mean equal to 0.

## Keyword arguments:
- `first::Integer = firstindex(pf) + window`: specify the first index of `pf` for which the probabilistic forecast will be caluclated
- `last::Integer = lastindex(pf)`: specify the last index of `pf` for which the probabilistic forecast will be caluclated
- `retrain::Integer = 1`: specify how often to retrain the model. If `retrain == 0`, the model will be trained only once, otherwise it will be retrained every `retrain` steps.

## Note
`QR` supports multiple regressors; `IDR` partially supports multiple regressors - one isotonic regression is fitted to each forecast and the final predictive distribution is an average of individual distributions; `CP` and `Normal` do not support multiple regressors.
"""
function point2prob(pf::PointForecasts{F, I}, modelname::Symbol, window::Integer, prob::AbstractVector{<:AbstractFloat}; first::Integer=window+firstindex(pf), last::Integer=lastindex(pf), retrain::Integer=1) where {F, I}
    window > 0 || throw(ArgumentError("`window` must be greater than 0"))
    retrain >= 0 || throw(ArgumentError("`retrain` must be non-negative"))
    (first >= firstindex(pf)+window && last <= lastindex(pf)) || throw(ArgumentError("`first` cannot be smaller than `firstindex(pf)+window` and `last` cannot be greater than `lastindex(pf)`"))
    prob = Vector{F}(prob)
    quantiles = zeros(F, last-first+1, length(prob))
    model = getmodel(Val(modelname), window, npred(pf), prob)
    pf = (nreg(model) == 1 && npred(pf) > 1) ? average(pf) : pf
    for t in first:last
        if t == first || (retrain > 0 && (t - first) % retrain == 0)
            _train(model, viewpred(pf, t-window:t-1), viewobs(pf, t-window:t-1))
        end
        _predict!(model, @view(quantiles[t-first+1, :]), viewpred(pf, t), prob)
    end
    return QuantForecasts(Val(:nocopy),
        quantiles,
        getobs(pf, first:last),
        getid(pf, first:last),
        prob)
end

function point2prob(pf::PointForecasts{F, I}, modelname::Symbol, window::Integer, prob::AbstractFloat; kwargs...) where {F, I}
   point2prob(pf, modelname, window, [prob]; kwargs...)
end
    
function point2prob(pf::PointForecasts{F, I}, modelname::Symbol, window::Integer, prob::Integer; kwargs...) where {F, I}
    point2prob(pf, modelname, window, equidistant(prob, F); kwargs...)
end

"""
    conformalize(qf::QuantForecasts{F, I}, window::Integer[; first::Integer, last::Integer)
Perform [conformalization](https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf) of quantile forecasts provided in `ps`.
Conformalized quantiles will be calculated for observations between the index `first` and `last` of `qf`. The model is retrained every step on the last `window` observations.

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
    return QuantForecasts(Val(:nocopy),
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
