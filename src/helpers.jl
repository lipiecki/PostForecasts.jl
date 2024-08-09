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
