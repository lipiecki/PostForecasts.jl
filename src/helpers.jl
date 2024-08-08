# helper functions that are not exported with the package, inteded for internal use, lack argument validation

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
function cdf2quantiles!(output::AbstractVector{T}, cdf::AbstractVector{T}, y::AbstractVector{T}, prob::AbstractVector{T}) where {T<:AbstractFloat}
    for i in eachindex(output)
        @inbounds index = searchsortedfirst(cdf, prob[i] - eps(T))
        @inbounds output[i] = y[min(lastindex(y), index)]
    end
end

"""
    isunique(X::Vector{<:Integer})
Return `true`` if vector `X` contains only unique values, otherwise return `false`.
"""
function isunique(X::Vector{<:Integer})
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

function _point2prob(pf::PointForecasts{F, I}, window::Integer, model::ProbModel, prob::Vector{F}, first::Integer, last::Integer, recalibration::Integer) where {F, I}
    quantiles = zeros(F, last-first+1, length(prob))
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
        prob)
end