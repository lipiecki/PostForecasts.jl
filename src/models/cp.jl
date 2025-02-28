"""
    CP([type::Type{F}=Float64,] n::Integer[; abs::Bool=true]) where {F<:AbstractFloat}
Creates a `CP{F}<:UniPostModel{F}<:PostModel{F}` model for conformal prediction that stores the non-conformity scores of `n` observations. Optional keyword argument `abs` specifies whether to use absolute errors.
"""
struct CP{F<:AbstractFloat} <: UniPostModel{F}
    scores::Vector{F}
    abs::Bool
    CP(::Type{F}, n::Integer; abs::Bool=true) where {F<:AbstractFloat} = new{F}(Vector{F}(undef, n), abs)
end

CP(n::Integer; abs::Bool=true) = CP(Float64, n; abs=abs)

getmodel(::Type{F}, ::Val{:cp}, params::Vararg) where {F<:AbstractFloat} = CP(F, params[1])

getmodel(::Type{F}, ::Val{:hs}, params::Vararg) where{F<:AbstractFloat} = CP(F, params[1], abs=false)

matchwindow(m::CP, window::Integer) = length(m.scores) == window

"""
    getscores(m::CP)
Return a vector of non-conformity score values from model `m`.
"""
function getscores(m::CP)
    return copy(m.scores)
end

function _train(m::CP, X::AbstractVecOrMat{<:Number}, Y::AbstractVector{<:Number})::Nothing
    for i in eachindex(m.scores)
        m.scores[i] = Y[i] - X[i]
    end
    if m.abs
        for i in eachindex(m.scores)
            m.scores[i] = abs(m.scores[i])
        end
    end
    sort!(m.scores)
    return nothing
end

function _predict(m::CP{F}, input::Number, prob::AbstractFloat) where {F<:AbstractFloat}
    if m.abs
        sgn::F = prob ≈ 0.5 ? 0.0 : (prob < 0.5 ? -1.0 : 1.0)
        return input + sgn*quantile(m.scores, (2prob - 1)sgn, sorted=true, alpha=1, beta=1)
    else
        return input + quantile(m.scores, prob, sorted=true, alpha=1, beta=1)
    end
end

function _predict(m::CP{F}, input::Number, prob::AbstractVector{<:AbstractFloat}) where {F<:AbstractFloat}
    output = Vector{F}(undef, length(prob))
    for j in eachindex(output)
        output[j] = _predict(m, input, prob[j])
    end
    return output
end

function _predict!(m::CP, output::AbstractVector{<:AbstractFloat}, input::Number, prob::AbstractVector{<:AbstractFloat})::Nothing
    for j in eachindex(output)
        output[j] = _predict(m, input, prob[j])
    end
    return nothing
end

function _predict!(m::CP, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})::Nothing
    for j in eachindex(output)
        output[j] = _predict(m, input[begin], prob[j])
    end
    return nothing
end
