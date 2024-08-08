"""
    Normal
Structure for storing parameters of the normal distribution.
"""
struct Normal <: UniRegProbModel
    μ::Base.RefValue{Float64}
    σ::Base.RefValue{Float64}
    zeromean::Bool

    Normal(;zeromean::Bool=false) = new(Ref{Float64}(0.0), Ref{Float64}(1.0), zeromean)
end

getmodel(::Val{:normal}, ::Vararg) = Normal()
getmodel(::Val{:zeronormal}, ::Vararg) = Normal(zeromean=true)

"""
    getmean(m::Normal)
Return the mean of the distribution stored in `m`. 
"""
function getmean(m::Normal)
    return m.μ[]
end

"""
    getstd(m::Normal)
Return the standard deviation of the distribution stored in `m`.
"""
function getstd(m::Normal)
    return m.σ[]
end

function _train(m::Normal, X::AbstractVecOrMat{<:AbstractFloat}, Y::AbstractVector{<:AbstractFloat})
    n = length(Y)
    if m.zeromean
        m.μ[] = 0.0
        m.σ[] = sqrt(sum(abs2, Y - X)/n)
    else
        m.μ[] = mean(Y) - mean(X)
        m.σ[] = sqrt(sum(abs2, Y - X .- getmean(m))/(n - 1))
    end
end

function _predict(m::Normal, input::Number, prob::AbstractFloat)
    return input + getmean(m) + getstd(m)*sqrt(2)*erfinv(2*prob - 1)
end

function _predict!(m::Normal, output::AbstractVector{<:AbstractFloat}, input::Number, prob::AbstractVector{<:AbstractFloat})
    for j in eachindex(output)
        @inbounds output[j] = _predict(m, input, prob[j])
    end
end

function _predict!(m::Normal, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})
    for j in eachindex(output)
        @inbounds output[j] = _predict(m, input[1], prob[j])
    end
end
