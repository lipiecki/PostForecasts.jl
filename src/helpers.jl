# helper functions that are not exported with the package, inteded for internal use, lack argument validation

"""
    equidistant(n::Integer, f::Type{F}=Float64) where {F<:AbstractFloat}
Return a `Vector{F}` of `n` equidistant quantile levels.
"""
function equidistant(n::Integer, type::Type{F}=Float64) where {F<:AbstractFloat}
    res = Vector{F}(undef, n)
    for i in 1:n
        res[i] = i/(n+1)
    end
    return res
end

"""
    isunique(X::Vector{<:Integer})
Return `true` if vector `X` contains only unique values, otherwise return `false`.
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
