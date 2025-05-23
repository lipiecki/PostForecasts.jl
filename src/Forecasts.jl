abstract type Forecasts{F<:AbstractFloat, I<:Integer} end

"""
    PointForecasts(pred::AbstractVecOrMat{F}, obs::AbstractVector{F}[, id::AbstractVector{I}]) where {F<:AbstractFloat, I<:Integer}
Create `PointForecasts{F, I}` for storing the series of point `pred`ictions, along with the `obs`ervations and `id`entifiers.

The shape of `pred` should be such that `pred[t, i]` is the prediction for time `t` from the forecaster `i`.

If `id` is not provided, it will default to `1:length(obs)`.
"""
struct PointForecasts{F<:AbstractFloat, I<:Integer} <: Forecasts{F, I}
    pred::Matrix{F}
    obs::Vector{F}
    id::Vector{I}

    function PointForecasts(pred::AbstractMatrix{F}, obs::AbstractVector{F}, id::AbstractVector{I}) where {F<:AbstractFloat, I<:Integer}
        size(pred, 1) == length(obs) || throw(ArgumentError("size of `pred` is $(size(pred)) while length of `obs` is $(length(obs))"))
        size(pred, 1) == length(id) || throw(ArgumentError("size of `pred` is $(size(pred)) while length of `id` is $(length(id))"))
        isunique(id) || throw(ArgumentError("`id` must contain only unique elements"))
        new{F, I}(Matrix{F}(pred), Vector{F}(obs), Vector{I}(id))
    end
end

PointForecasts(pred::AbstractVector{F}, obs::AbstractVector{F}, id::AbstractVector{I}) where {F<:AbstractFloat, I<:Integer} =
        PointForecasts(reshape(pred, length(pred), 1), obs, id)

PointForecasts(pred::AbstractMatrix{F}, obs::AbstractVector{F}) where {F<:AbstractFloat} =
    PointForecasts(pred, obs, 1:length(obs))

    PointForecasts(pred::AbstractVector{F}, obs::AbstractVector{F}) where {F<:AbstractFloat} =
        PointForecasts(reshape(pred, length(pred), 1), obs, 1:length(obs))

"""
    QuantForecasts(pred::AbstractMatrix{F}, obs::AbstractVector{F}[, id::AbstractVector{I}, prob::Union{F, AbstractVector{F}}]) where {F<:AbstractFloat, I<:Integer}
Create `QuantForecasts{F, I}` for storing the series of probabilistic `pred`ictions, represented as quantiles of predictive distribution at specified `prob`abilities, along with the `obs`ervations and `id`entifiers.

The shape of `pred` should be such that `pred[t, i]` is the prediction for time `t` of the `prob[i]`-quantile.

If `id` is not provided, it will default to `1:length(obs)`.
If `prob` is not provided, it will default to `size(pred, 2)` equidistant quantiles.
"""
struct QuantForecasts{F<:AbstractFloat, I<:Integer} <: Forecasts{F, I}
    pred::Matrix{F}
    obs::Vector{F}
    id::Vector{I}
    prob::Vector{F}

    # unsafe non-copying constructor
    function QuantForecasts(pred::Matrix{F}, obs::Vector{F}, id::Vector{I}, prob::Vector{F}, safe::Val{false}) where {F<:AbstractFloat, I<:Integer}
        size(pred, 1) == length(obs) || throw(ArgumentError("size of `pred` is $(size(pred)) while length of `obs` is $(length(obs))"))
        size(pred, 1) == length(id) || throw(ArgumentError("size of `pred` is $(size(pred)) while length of `id` is $(length(id))"))
        size(pred, 2) == length(prob) || throw(ArgumentError("size of `pred` is $(size(pred)) while length of `prob` is $(length(prob))"))
        issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
        (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
        isunique(id) || throw(ArgumentError("`id` must contain only unique elements"))
        for t in eachindex(@view(pred[:, begin]))
            if !issorted(@view(pred[t, :]))
                throw(ArgumentError("quantile `pred` passed to the constructor are decreasing"))
            end
        end
        new{F, I}(pred, obs, id, prob)
    end

    # safe copying constructor
    function QuantForecasts(pred::AbstractMatrix{F}, obs::AbstractVector{F}, id::AbstractVector{I}, prob::AbstractVector{F}, safe::Val{true}) where {F<:AbstractFloat, I<:Integer}
        QuantForecasts(Matrix{F}(pred), Vector{F}(obs), Vector{I}(id), Vector{F}(prob), Val(false))
    end
    
end

QuantForecasts(pred::AbstractMatrix{F}, obs::AbstractVector{F}, id::AbstractVector{I}, prob::AbstractVector{F}) where {F<:AbstractFloat, I<:Integer} = 
    QuantForecasts(pred, obs, id, prob, Val(true))

QuantForecasts(pred::AbstractMatrix{F}, obs::AbstractVector{F}, id::AbstractVector{I}) where {F<:AbstractFloat, I<:Integer} =
    QuantForecasts(pred, obs, id, equidistant(size(pred, 2), F))

QuantForecasts(pred::AbstractVector{F}, obs::AbstractVector{F}, id::AbstractVector{I}, prob::AbstractVector{F}) where {F<:AbstractFloat, I<:Integer} =
    QuantForecasts(reshape(pred, length(pred), 1), obs, id, prob)

QuantForecasts(pred::AbstractVector{F}, obs::AbstractVector{F}, id::AbstractVector{I}) where {F<:AbstractFloat, I<:Integer} =
    QuantForecasts(reshape(pred, length(pred), 1), obs, id)

QuantForecasts(pred::AbstractVecOrMat{F}, obs::AbstractVector{F}, prob::AbstractVector{F}) where {F<:AbstractFloat} =
    QuantForecasts(pred, obs, 1:length(obs), prob)

QuantForecasts(pred::AbstractVecOrMat{F}, obs::AbstractVector{F}) where {F<:AbstractFloat} =
    QuantForecasts(pred, obs, 1:length(obs))

QuantForecasts(pred::AbstractVecOrMat{F}, obs::AbstractVector{F}, id::AbstractVector{I}, prob::F) where {F<:AbstractFloat, I<:Integer} =
    QuantForecasts(pred, obs, id, [prob])

QuantForecasts(pred::AbstractVecOrMat{F}, obs::Vector{F}, prob::F) where {F<:AbstractFloat} =
    QuantForecasts(pred, obs, 1:length(obs), prob)

Base.show(io::IO, f::Forecasts) = println(io, typeof(f), " with a pool of ", npred(f),  " forecast(s) at ", length(f), " timesteps, between ", f.id[begin], " and ", f.id[end])

function Base.length(f::Forecasts)
    return length(f.obs)
end

function Base.getindex(pf::PointForecasts, t::Integer)
    return (pred = pf.pred[t, :],
            obs = pf.obs[t],
            id = pf.id[t])
end

function Base.getindex(pf::PointForecasts, T::AbstractVector{<:Integer})
    return PointForecasts(
            @view(pf.pred[T, :]),
            @view(pf.obs[T]),
            @view(pf.id[T]))
end

function Base.getindex(qf::QuantForecasts, t::Integer)
    return (pred = qf.pred[t, :],
            obs = qf.obs[t],
            id = qf.id[t],
            prob = copy(qf.prob))
end

function Base.getindex(qf::QuantForecasts, T::AbstractVector{<:Integer})
    return QuantForecasts(
            @view(qf.pred[T, :]),
            @view(qf.obs[T]),
            @view(qf.id[T]),
            qf.prob)
end

function Base.firstindex(f::Forecasts)
    return firstindex(f.obs)
end

function Base.lastindex(f::Forecasts)
    return lastindex(f.obs)
end

function Base.eachindex(f::Forecasts)
    return eachindex(f.obs)
end

"""
    findindex(f::Forecasts, i::Integer)
Return the index of `f`, for which the element of field `id` equals `i`.
"""
function findindex(f::Forecasts, i::Integer)
    t = findfirst(f.id .== i) 
    isnothing(t) && throw(error("field `id` of the provided series does not contain '$(i)'"))
    return t
end

function (f::Forecasts)(i::Integer) 
    f[findindex(f, i)]
end

function (f::Forecasts)(i::Integer, j::Integer) 
    f[findindex(f, i):findindex(f, j)]
end

function (f::Forecasts)(I::AbstractVector{<:Integer})  
    f[[findindex(f, i) for i in I]]
end

"""
    couple(fs::AbstractVector{<:T}) where T<:Union{PointForecasts, QuantForecasts}
Merge elements of `fs` into a single `Forecasts` object.
"""
function couple(fs::AbstractVector{<:PointForecasts})
    checkmatch(fs)
    PointForecasts(
        hcat(getpred.(fs)...),
        getobs(fs[begin]),
        getid(fs[begin]))
end

function couple(fs::AbstractVector{<:QuantForecasts})
    checkmatch(fs)
    QuantForecasts(
        hcat(getpred.(fs)...),
        getobs(fs[begin]),
        getid(fs[begin]),
        vcat(getprob.(fs)...))
end

"""
    decouple(f:<Forecasts})
Return a vector of `PointForecasts` or `QuantForecasts` objects, where each element contains an individual forecast series from `f`.
"""
function decouple(f::PointForecasts) 
    return [PointForecasts(@view(f.pred[:, i]), f.obs, f.id) for i in 1:npred(f)]
end

function decouple(f::QuantForecasts) 
    return [QuantForecasts(@view(f.pred[:, i]), f.obs, f.id, f.prob[i]) for i in 1:npred(f)]
end

"""
    npred(f::Forecasts)
Return the number of point forecasts in `f::PointForecasts` or the number of forecasted quantiles in `f::QuantForecasts`.
"""
function npred(f::Forecasts)
    return size(f.pred, 2)
end

"""
    setpred(f::Forecasts, t::Integer, i::Integer, val::AbstractFloat)
Set the element of field `f.pred` at indices `t, i` to `val`.
"""
function setpred(f::Forecasts, t::Integer, i::Integer, val::AbstractFloat)
    f.pred[t, i] = val
end

"""
    getpred(f::Forecasts[, T, I])
Return the copy of `pred`ictions from `f`. 

Provide optional argument `T::Union{Integer, AbstractVector{<:Integer}}` to get `pred`ictions at specified time indices.

Additionally, provide `I::Union{Integer, AbstractVector{<:Integer}}` to get `pred`icitons at specified forecast indices.
"""
function getpred(f::Forecasts)
    return copy(f.pred)
end

function getpred(f::Forecasts, T::Union{Integer, AbstractVector{<:Integer}})
    return f.pred[T, :]
end

function getpred(f::Forecasts, T::Union{Integer, AbstractVector{<:Integer}}, I::Union{Integer, AbstractVector{<:Integer}})
    return f.pred[T, I]
end

"""
    getobs(f::Forecasts[, T])
Return the copy of `obs`ervations from `f`. 

Provide optional argument `T::Union{Integer, AbstractVector{<:Integer}}` to get `obs`ervations at specified time indices.
"""
function getobs(f::Forecasts)
    return copy(f.obs)
end

function getobs(f::Forecasts, T::Union{Integer, AbstractVector{<:Integer}})
    return f.obs[T]
end

"""
    getid(f::Forecasts[, T])
Return the copy of `id`entifiers from `f`. 

Provide optional argument `T::Union{Integer, AbstractVector{<:Integer}}` to get `id`entifiers at specified time indices.
"""
function getid(f::Forecasts)
    return copy(f.id)
end

function getid(f::Forecasts, T::Union{Integer, AbstractVector{<:Integer}})
    return f.id[T]
end

"""
    getprob(qf::QuantForecasts[, I])
Return the copy of `prob`abilities from `qf`. 

Provide optional argument `I::Union{Integer, AbstractVector{<:Integer}}` to get `prob`abilities at specified forecast indices.
"""
function getprob(qf::QuantForecasts)
    return copy(qf.prob)
end

function getprob(qf::QuantForecasts, I::Union{Integer, AbstractVector{<:Integer}})
    return qf.prob[I]
end

"""
    viewpred(f::Forecasts[, T, I])
Return the view of `pred`ictions from `f`. 

Provide optional argument `T::Union{Integer, AbstractVector{<:Integer}}` to get `pred`ictions at specified time indices.

Additionally, provide `I::Union{Integer, AbstractVector{<:Integer}}` to get `pred`icitons at specified forecast indices.
"""
function viewpred(f::Forecasts)
    return @view(f.pred[:, :])
end

function viewpred(f::Forecasts, T::Union{Integer, AbstractVector{<:Integer}})
    return @view(f.pred[T, :])
end

function viewpred(f::Forecasts, T::Union{Integer, AbstractVector{<:Integer}}, I::Union{Integer, AbstractVector{<:Integer}})
    return @view(f.pred[T, I])
end

"""
    viewobs(f::Forecasts[, T])
Return the view of `obs`ervations from `f`. 

Provide optional argument `T::Union{Integer, AbstractVector{<:Integer}}` to get `obs`ervations at specified time indices.
"""
function viewobs(f::Forecasts)
    return @view(f.obs[:])
end

function viewobs(f::Forecasts, T::Union{Integer, AbstractVector{<:Integer}})
    return @view(f.obs[T])
end

"""
    viewid(f::Forecasts[, T])
Return the view of `id`entifiers from `f`. 

Provide optional argument `T::Union{Integer, AbstractVector{<:Integer}}` to get `id`entifiers at specified time indices.
"""
function viewid(f::Forecasts)
    return @view(f.id[:])
end

function viewid(f::Forecasts, T::Union{Integer, AbstractVector{<:Integer}})
    return @view(f.id[T])
end

"""
    viewprob(qf::QuantForecasts[, I])
Return the view of `prob`abilities from `qf`. 

Provide optional argument `I::Union{Integer, AbstractVector{<:Integer}}` to get `prob`abilities at specified forecast indices.
"""
function viewprob(qf::QuantForecasts)
    return @view(qf.prob[:])
end

function viewprob(qf::QuantForecasts, I::Union{Integer, AbstractVector{<:Integer}})
    return @view(qf.prob[I])
end
