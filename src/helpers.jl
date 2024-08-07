"""
    equidistant(n::Integer, T::Type)
Return a vector of `n` equidistant probabilitiy values.
"""
function equidistant(n::Integer, T::Type = Float64)
    return [T(i/(n+1)) for i in 1:n]
end

"""
    cdf2quantiles!(output, cdf, y, prob)
Calculate quantiles from a tabular `cdf` evaluated at values `y`. 
Write quantiles corresponding to specified probabilities `prob`.
"""
function cdf2quantiles!(output::AbstractVector{<:AbstractFloat}, cdf::AbstractVector{<:AbstractFloat}, y::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})
    for i in eachindex(output)
        index = searchsortedfirst(cdf, prob[i] - eps(eltype(prob)))
        output[i] = y[min(lastindex(y), index)]
    end
end

"""
    arematching(S::AbstractVector{<:ForecastSeries}; checkpred::Bool=false)
Check if PointForecasts or QuantForecasts provided in vector `S` are matching, i.e.:
- all their `id`entifiers match
- all their `obs`ervations match
- their number of forecasts match (if `checkpred=true`)
- their `prob`abilities match (for QuantForecasts if `checkpred=true`).

Return nothing, throw an `ArgumentError` if any of the requirements above is not met.
"""
function arematching(S::AbstractVector{PointForecasts{F, I}}; checkpred::Bool=false) where {F, I}
    first, state = iterate(S, firstindex(S))
    nextstatepair = iterate(S, state)
    while !isnothing(nextstatepair)
        next, state = nextstatepair
        length(first) == length(next) || throw(ArgumentError("`PointForecasts` have different lengths"))
        (!checkpred || npred(first) == npred(next)) || throw(ArgumentError("`PointForecasts` have different sizes of forecast pools"))
        all(first.obs .≈ next.obs) || throw(ArgumentError("`PointForecasts` observations (elements of `obs` field) do not match"))
        all(first.id .≈ next.id) || throw(ArgumentError("`PointForecasts` identifiers (elements of `id` field) do not match"))
        nextstatepair = iterate(S, state)
    end
end

function arematching(S::AbstractVector{QuantForecasts{F, I}}; checkpred::Bool=false) where {F, I}
    first, state = iterate(S, firstindex(S))
    nextstatepair = iterate(S, state)
    while !isnothing(nextstatepair)
        next, state = nextstatepair
        length(first) == length(next) || throw(ArgumentError("`QuantForecasts` have different lengths"))
        if checkpred
            npred(first) == npred(next) || throw(ArgumentError("`QuantForecasts` have different number of quantiles"))
            all(first.prob .≈ next.prob) || throw(ArgumentError("`QuantForecasts` have different quantile levels"))
        end
        all(first.obs .≈ next.obs) || throw(ArgumentError("`QuantForecasts` observations (elements of `obs` field) do not match"))
        all(first.id .≈ next.id) || throw(ArgumentError("`QuantForecasts` identifiers (elements of `id` field) do not match"))
        nextstatepair = iterate(S, state)
    end
end

"""
    isunique(X::AbstractVector{<:Integer})
Return `true`` if vector `X` contains only unique values, otherwise return `false`.
"""
function isunique(X::AbstractVector{<:Integer})
    iterated = Set{eltype(X)}()
    for x in X
        if x ∈ iterated
            return false
        else
            push!(iterated, x)
        end
    end
    return true
end
