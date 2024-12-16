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
Create `QuantForecasts{F, I}` for storing the series of probabilistic `pred`ictions, represented as quantiles of predictive distribution at specfied `prob`abilities, along with the `obs`ervations and `id`entifiers.

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

Base.show(io::IO, fs::Forecasts) = println(io, typeof(fs), " with a pool of ", npred(fs),  " forecast(s) at ", length(fs), " timesteps, between ", fs.id[begin], " and ", fs.id[end])

function Base.length(fs::Forecasts)
    return length(fs.obs)
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

function Base.firstindex(fs::Forecasts)
    return firstindex(fs.obs)
end

function Base.lastindex(fs::Forecasts)
    return lastindex(fs.obs)
end

function Base.eachindex(fs::Forecasts)
    return eachindex(fs.obs)
end

"""
    findindex(fs::Forecasts, i::Integer)
Return the index of `fs`, for which the element of field `id` equals `i`.
"""
function findindex(fs::Forecasts, i::Integer)
    t = findfirst(fs.id .== i) 
    isnothing(t) && throw(error("field `id` of the provided series does not contain '$(i)'"))
    return t
end

function (fs::Forecasts)(i::Integer) 
    fs[findindex(fs, i)]
end

function (fs::Forecasts)(i::Integer, j::Integer) 
    fs[findindex(fs, i):findindex(fs, j)]
end

function (fs::Forecasts)(I::AbstractVector{<:Integer})  
    fs[[findindex(fs, i) for i in I]]
end

"""
    decouple(pf::PointForecasts) 
Return `::Vector{PointForecasts}`, where each element contains an individual forecast series from `pf`.
"""
function decouple(pf::PointForecasts) 
    return [PointForecasts(@view(pf.pred[:, i]), pf.obs, pf.id) for i in 1:npred(pf)]
end

"""
    npred(fs::Forecasts)
Return the number of point forecasts in `fs::PointForecasts` or the number of forecasted quantiles in `fs::QuantForecasts`.
"""
function npred(fs::Forecasts)
    return size(fs.pred, 2)
end

"""
    setpred(fs::Forecasts, t::Integer, i::Integer, val::AbstractFloat)
Set the element of field `fs.pred` at indices `t, i` to `val`.
"""
function setpred(fs::Forecasts, t::Integer, i::Integer, val::AbstractFloat)
    fs.pred[t, i] = val
end

"""
    getpred(fs::Forecasts[, T, I])
Return the copy of `pred`ictions from `fs`. 

Provide optional argument `T::Union{Integer, AbstractVector{<:Integer}}` to get `pred`ictions at specified time indices.

Additionally, provide `I::Union{Integer, AbstractVector{<:Integer}}` to get `pred`icitons at specified forecast indices.
"""
function getpred(fs::Forecasts)
    return copy(fs.pred)
end

function getpred(fs::Forecasts, T::Union{Integer, AbstractVector{<:Integer}})
    return fs.pred[T, :]
end

function getpred(fs::Forecasts, T::Union{Integer, AbstractVector{<:Integer}}, I::Union{Integer, AbstractVector{<:Integer}})
    return fs.pred[T, I]
end

"""
    getobs(fs::Forecasts[, T])
Return the copy of `obs`ervations from `fs`. 

Provide optional argument `T::Union{Integer, AbstractVector{<:Integer}}` to get `obs`ervations at specified time indices.
"""
function getobs(fs::Forecasts)
    return copy(fs.obs)
end

function getobs(fs::Forecasts, T::Union{Integer, AbstractVector{<:Integer}})
    return fs.obs[T]
end

"""
    getid(fs::Forecasts[, T])
Return the copy of `id`entifiers from `fs`. 

Provide optional argument `T::Union{Integer, AbstractVector{<:Integer}}` to get `id`entifiers at specified time indices.
"""
function getid(fs::Forecasts)
    return copy(fs.id)
end

function getid(fs::Forecasts, T::Union{Integer, AbstractVector{<:Integer}})
    return fs.id[T]
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
    viewpred(fs::Forecasts[, T, I])
Return the view of `pred`ictions from `fs`. 

Provide optional argument `T::Union{Integer, AbstractVector{<:Integer}}` to get `pred`ictions at specified time indices.

Additionally, provide `I::Union{Integer, AbstractVector{<:Integer}}` to get `pred`icitons at specified forecast indices.
"""
function viewpred(fs::Forecasts)
    return @view(fs.pred[:, :])
end

function viewpred(fs::Forecasts, T::Union{Integer, AbstractVector{<:Integer}})
    return @view(fs.pred[T, :])
end

function viewpred(fs::Forecasts, T::Union{Integer, AbstractVector{<:Integer}}, I::Union{Integer, AbstractVector{<:Integer}})
    return @view(fs.pred[T, I])
end

"""
    viewobs(fs::Forecasts[, T])
Return the view of `obs`ervations from `fs`. 

Provide optional argument `T::Union{Integer, AbstractVector{<:Integer}}` to get `obs`ervations at specified time indices.
"""
function viewobs(fs::Forecasts)
    return @view(fs.obs[:])
end

function viewobs(fs::Forecasts, T::Union{Integer, AbstractVector{<:Integer}})
    return @view(fs.obs[T])
end

"""
    viewid(fs::Forecasts[, T])
Return the view of `id`entifiers from `fs`. 

Provide optional argument `T::Union{Integer, AbstractVector{<:Integer}}` to get `id`entifiers at specified time indices.
"""
function viewid(fs::Forecasts)
    return @view(fs.id[:])
end

function viewid(fs::Forecasts, T::Union{Integer, AbstractVector{<:Integer}})
    return @view(fs.id[T])
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
