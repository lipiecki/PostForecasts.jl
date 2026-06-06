"""
    point2quant(pf; method, window, quantiles[, start, stop, retrain])
Compute probabilistic forecast based on `pf::PointForecasts` using `PostModel` specified by `method::Symbol`.

Return `QuantForecasts` containing forecasts of specified `quantiles`:
- `quantiles::AbstractVector{<:AbstractFloat}`: vector of probabilities
- `quantiles::AbstractFloat`: a single probability value
- `quantiles::Integer`: number of equidistant probability values (e.g. 99 for percentiles).

## Available options for `method`:
- `:cp` for conformal prediction
- `:hs` for historical simulation
- `:idr` for isotonic distributional regression
- `:qr` for quantile regression
- `:iqr` for isotonic quantile regression
- `:lassoqr` for lasso quantile regression
- `:normal` for normal distribution of errors
- `:zeronormal` for normal distribution of errors with fixed mean equal to 0

## Other keyword arguments:
- `window::Integer`: the number of past observations used for training the model
- `start::Integer = pf.id[begin + window]`: specify the `id`entifier in `pf` at which quantile forecasts will start (if not provided, the first available will be used)
- `stop::Integer = pf.id[end]`: specify the `id`entifier in `pf` at which quantile forecasts will stop (if not provided, the last available will be used)
- `retrain::Integer = 1`: specify how often to retrain the model. If `retrain == 0`, the model will be trained only once, otherwise it will be retrained every `retrain` steps

## Note
- the function can also be called with `method`, `window` and `quantiles` as positional arguments
- `:qr` supports multiple regressors
- `:idr` partially supports multiple regressors: one isotonic regression is fitted to each forecast and the final predictive distribution is an average of individual distributions
- `:cp`, `:normal` and `:zeronormal` do not support multiple regressors: if `pf` contains multiple point forecasts, their average will be used for postprocessing
"""
point2quant(pf::PointForecasts{F, I}; method, window, quantiles, kwargs...) where {F, I} = point2quant(pf, method, window, quantiles; kwargs...)

function point2quant(pf::PointForecasts{F, I}, method::Symbol, window::Integer, quantiles::AbstractVector{<:AbstractFloat}; start::Union{Nothing, Integer}=nothing, stop::Union{Nothing, Integer}=nothing, retrain::Integer=1) where {F, I}
    (window > 0 && window < length(pf)) || throw(ArgumentError("`window` must be greater than 0 and smaller than the length of `pf`"))
    retrain >= 0 || throw(ArgumentError("`retrain` must be non-negative"))
    first = isnothing(start) ? firstindex(pf)+window : findindex(pf, start)
    last = isnothing(stop) ? lastindex(pf) : findindex(pf, stop)
    first > window || throw(ArgumentError("there is less than $(window) timesteps before $(start)"))
    prob = Vector{F}(quantiles)
    pred = zeros(F, last-first+1, length(prob))
    model = getmodel(Val(method), window, npred(pf), prob)
    pf = (nreg(model) == 1 && npred(pf) > 1) ? average(pf) : pf
    for t in first:last
        if t == first || (retrain > 0 && (t - first) % retrain == 0)
            _train(model, viewpred(pf, t-window:t-1), viewobs(pf, t-window:t-1))
        end
        input = viewpred(pf, t)
        _predict!(model, @view(pred[t-first+1, :]), input, prob)
        advance!(model, input)
    end
    return QuantForecasts(
        pred,
        getobs(pf, first:last),
        getid(pf, first:last),
        prob
    )
end

function point2quant(pf::PointForecasts{F, I}, method::Symbol, window::Integer, quantiles::AbstractFloat; kwargs...) where {F, I}
   point2quant(pf, method, window, [quantiles]; kwargs...)
end
    
function point2quant(pf::PointForecasts{F, I}, method::Symbol, window::Integer, quantiles::Integer; kwargs...) where {F, I}
    point2quant(pf, method, window, equidistant(quantiles, F); kwargs...)
end
