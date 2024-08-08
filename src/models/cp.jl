"""
    CP
Structure for storing the non-conformity scores used for computing prediction intervals using [Conformal Prediction](https://doi.org/10.1016/j.ijforecast.2020.09.006).

CP(n, abs=true) creates a new structure for storing non-conformity scores (absolute errors if `abs=true`) of `n` observations.
"""
struct CP <: UniRegProbModel
    scores::Vector{Float64}
    abs::Bool
    CP(n::Integer; abs::Bool=true) = new(Vector{Float64}(undef, n), abs)
end

getmodel(::Val{:cp}, params::Vararg) = CP(params[1])
getmodel(::Val{:hs}, params::Vararg) = CP(params[1], abs=false)

matchwindow(m::CP, window::Integer) = length(m.scores) == window

"""
    getscores(m::CP)
Return the copy of non-conformity score values from CP model `m`.
"""
function getscores(m::CP)
    return copy(m.scores)
end

function _train(m::CP, X::AbstractVecOrMat{<:Number}, Y::AbstractVector{<:Number})
    m.scores .= Y - X
    if m.abs
        for i in eachindex(m.scores)
            m.scores[i] = abs(m.scores[i])
        end
    end
    sort!(m.scores)
end

function _predict(m::CP, input::Number, prob::AbstractFloat)
    if m.abs
        if prob ≈ 0.5
            return input
        else
            sgn = prob < 0.5 ? -1.0 : 1.0
            return input + sgn*quantile(m.scores, sgn*(2prob - 1), sorted=true, alpha=1, beta=1)
        end
    else
        return input + quantile(m.scores, prob, sorted=true, alpha=1, beta=1)
    end
end

function _predict!(m::CP, output::AbstractVector{<:AbstractFloat}, input::Number, prob::AbstractVector{<:AbstractFloat})
    for j in eachindex(output)
        output[j] = _predict(m, input, prob[j])
    end
end

function _predict!(m::CP, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})
    for j in eachindex(output)
        output[j] = _predict(m, input[1], prob[j])
    end
end