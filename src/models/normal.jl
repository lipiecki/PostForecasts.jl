"""
    Normal([type::Type{F}=Float64]; zeromean::Bool=false) where {F<:AbstractFloat}
Creates a `Normal{F}<:UniPostModel{F}<:PostModel{F}` model for normal error distribution. Optional keyword argument `zeromean` specifies wheter to assume a zero mean.
"""
struct Normal{F<:AbstractFloat} <: UniPostModel{F}
    μ::Base.RefValue{F}
    σ::Base.RefValue{F}
    zeromean::Bool

    Normal(::Type{F}; zeromean::Bool=false) where {F<:AbstractFloat} = new{F}(Ref{F}(0.0), Ref{F}(1.0), zeromean)
end

Normal(;zeromean::Bool=false) = Normal(Float64, zeromean=zeromean)

getmodel(::Type{F}, ::Val{:normal}, ::Vararg) where {F<:AbstractFloat} = Normal(F)

getmodel(::Type{F}, ::Val{:zeronormal}, ::Vararg) where {F<:AbstractFloat} = Normal(F, zeromean=true)

"""
    getmean(m::Normal)
Return the mean of the distribution from model `m`. 
"""
function getmean(m::Normal)
    return m.μ[]
end

"""
    getstd(m::Normal)
Return the standard deviation of the distribution from model `m`.
"""
function getstd(m::Normal)
    return m.σ[]
end

function _train(m::Normal, X::AbstractVecOrMat{<:Number}, Y::AbstractVector{<:Number})::Nothing
    n = length(Y)
    if m.zeromean
        m.μ[] = 0.0
        m.σ[] = sqrt(sum(abs2, Y - X)/n)
    else
        m.μ[] = mean(Y) - mean(X)
        m.σ[] = sqrt(sum(abs2, Y - X .- getmean(m))/(n - 1))
    end
    return nothing
end

function _predict(m::Normal{F}, input::Number, prob::AbstractFloat) where {F<:AbstractFloat}
    return input + getmean(m) + getstd(m)*(sqrt(2)*erfinv(2*prob - 1))
end

function _predict(m::Normal{F}, input::Number, prob::AbstractVector{<:AbstractFloat}) where {F<:AbstractFloat}
    output = Vector{F}(undef, length(prob))
    for j in eachindex(output)
        output[j] = _predict(m, input, prob[j])
    end
    return output
end

function _predict!(m::Normal, output::AbstractVector{<:AbstractFloat}, input::Number, prob::AbstractVector{<:AbstractFloat})::Nothing
    for j in eachindex(output)
        output[j] = _predict(m, input, prob[j])
    end
    return nothing
end

function _predict!(m::Normal, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})::Nothing
    for j in eachindex(output)
        output[j] = _predict(m, input[1], prob[j])
    end
    return nothing
end
