"""
    getmodel([::Type{<:AbstractFloat}=Float64,] ::Val, params...)
Helper function that dispatches the model based on the model name passed as `Val`.

## Available methods
- `getmodel([type,] Val(:qra), n, m, prob)` for Quantile Regression Averaging
- `getmodel([type,] Val(:cp), n)` for Conformal Prediction
- `getmodel([type,] Val(:hs), n)` for Conformal Prediction Prediction with non-symmetric errors (a.k.a. Historical Simulation)
- `getmodel([type,] Val(:idr), n, m)` for Isotonic Distributional Regression
- `getmodel([type,] Val(:normal))` for Normal distribution of errors
- `getmodel([type,] Val(:zeronormal))` for Normal distribution of errors with fixed mean equal to 0

where `n` is the length of the training window, `m` is the number of regressors and `prob` is the probability (scalar value or vector).

Return an appropriate `PostModel`.
"""
getmodel(modelname::Val, params::Vararg) = getmodel(Float64, modelname, params...)
getmodel(::Type{<:AbstractFloat}, ::Val, ::Vararg) = throw(ArgumentError("provided model name not recognized"))

"""
    nreg(m::PostModel)
Return the number of regressors of model `m`.
"""
nreg(::UniPostModel) = 1

"""
    matchwindow(m::PostModel, window::Integer)
Return `true` if `window` matches the specification of model `m`, otherwise return `false`.
"""
matchwindow(::PostModel, ::Integer) = true

"""
    checkmatch(fs:; checkpred::Bool=false)
Check if Forecasts in `fs` correspond to the same tiemseries, i.e.:
- all their `id`entifiers match
- all their `obs`ervations match.

Additionally, provided that `checkpred=true`, check if:
- their number of forecasts match
- their `prob`abilities match.

Return nothing or throw an `ArgumentError` if any of the requirements above is not met.
"""
function checkmatch(fs::AbstractVector{<:PointForecasts}; checkpred::Bool=false)
    first, state = iterate(fs, firstindex(fs))
    nextstatepair = iterate(fs, state)
    while !isnothing(nextstatepair)
        next, state = nextstatepair
        length(first) == length(next) || throw(ArgumentError("`PointForecasts` have different lengths"))
        (!checkpred || npred(first) == npred(next)) || throw(ArgumentError("`PointForecasts` have different sizes of forecast pools"))
        all(first.obs .≈ next.obs) || throw(ArgumentError("`PointForecasts` observations (elements of `obs` field) do not match"))
        all(first.id .≈ next.id) || throw(ArgumentError("`PointForecasts` identifiers (elements of `id` field) do not match"))
        nextstatepair = iterate(fs, state)
    end
end

function checkmatch(fs::AbstractVector{<:QuantForecasts}; checkpred::Bool=false)
    first, state = iterate(fs, firstindex(fs))
    nextstatepair = iterate(fs, state)
    while !isnothing(nextstatepair)
        next, state = nextstatepair
        length(first) == length(next) || throw(ArgumentError("`QuantForecasts` have different lengths"))
        if checkpred
            npred(first) == npred(next) || throw(ArgumentError("`QuantForecasts` have different number of quantiles"))
            all(first.prob .≈ next.prob) || throw(ArgumentError("`QuantForecasts` have different quantile levels"))
        end
        all(first.obs .≈ next.obs) || throw(ArgumentError("`QuantForecasts` observations (elements of `obs` field) do not match"))
        all(first.id .≈ next.id) || throw(ArgumentError("`QuantForecasts` identifiers (elements of `id` field) do not match"))
        nextstatepair = iterate(fs, state)
    end
end

function checkmatch(fs::Vararg{T, N}; checkpred::Bool=false) where {T<:Union{PointForecasts, QuantForecasts}, N}
    v = Vector{T}(undef, N)
    v .= fs
    checkmatch(v; checkpred=checkpred)
end
