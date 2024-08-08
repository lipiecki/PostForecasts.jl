"""
    PointForecasts
Structure for storing the series of point `pred`ictions, along with the `obs`ervations and `id`entifiers.
"""
struct PointForecasts{F<:AbstractFloat, I<:Integer}<:Forecasts
    pred::Matrix{<:F}
    obs::Vector{<:F}
    id::Vector{<:I}

    function PointForecasts(pred::Matrix{<:F}, obs::Vector{<:F}, id::Vector{<:I}) where {F<:AbstractFloat, I<:Integer}
        size(pred, 1) == length(obs) || throw(ArgumentError("size of `pred` is $(size(pred)) while length of `obs` is $(length(obs))"))
        size(pred, 1) == length(id) || throw(ArgumentError("size of `pred` is $(size(pred)) while length of `id` is $(length(id))"))
        isunique(id) || throw(ArgumentError("`id` must contain only unique elements"))
        new{F, I}(pred, obs, id)
    end

    function PointForecasts(pred::Matrix{<:F}, obs::AbstractVector{<:F}) where {F<:AbstractFloat}
        size(pred, 1) == length(obs) || throw(ArgumentError("size of `pred` is $(size(pred)) while length of `obs` is $(length(obs))"))
        new{F, Int64}(pred, obs, Vector(1:length(obs)))
    end

    function PointForecasts(pred::Vector{<:F}, obs::Vector{<:F}, id::Vector{<:I}) where {F<:AbstractFloat, I<:Integer}
        PointForecasts(reshape(pred, length(pred), 1), obs, id)
    end

    function PointForecasts(pred::Vector{<:F}, obs::Vector{<:F}) where {F<:AbstractFloat}
        PointForecasts(reshape(pred, length(pred), 1), obs)
    end
end

"""
    QuantForecasts
Structure for storing the series of probabilistic `pred`ictions, represented as quantiles of predictive distribution at specfied `prob`abilities, along with the `obs`ervations and `id`entifiers.
"""
struct QuantForecasts{F<:AbstractFloat, I<:Integer}<:Forecasts
    pred::Matrix{<:F}
    prob::Vector{<:F}
    obs::Vector{<:F}
    id::Vector{<:I}

    function QuantForecasts(pred::Matrix{<:F}, obs::Vector{<:F}, id::Vector{<:I}, prob::Vector{<:F}) where {F<:AbstractFloat, I<:Integer}
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
        new{F, I}(pred, prob, obs, id)
    end
    
    function QuantForecasts(pred::Matrix{<:F}, obs::Vector{<:F}, prob::Vector{<:F}) where {F<:AbstractFloat}
        QuantForecasts(pred, obs, Vector(1:length(obs)), prob)
    end

    function QuantForecasts(pred::Matrix{<:F}, obs::Vector{<:F}, id::Vector{<:I}) where {F<:AbstractFloat, I<:Integer}
        QuantForecasts(pred, obs, id, equidistant(size(pred, 2), F))
    end

    function QuantForecasts(pred::Matrix{<:F}, obs::Vector{<:F}) where {F<:AbstractFloat}
        QuantForecasts(pred, obs, Vector(1:length(obs)), equidistant(size(pred, 2), F))
    end

    function QuantForecasts(pred::Vector{<:F}, obs::Vector{<:F}, id::Vector{<:I}, prob::F) where {F<:AbstractFloat, I<:Integer}
        QuantForecasts(reshape(pred, length(obs), 1), obs, id, [prob])
    end

    function QuantForecasts(pred::Vector{<:F}, obs::Vector{<:F}, prob::F) where {F<:AbstractFloat}
        QuantForecasts(pred, obs, Vector(1:length(obs)), prob)
    end
end

function Base.length(s::Forecasts)
    return length(s.obs)
end

function Base.getindex(ps::PointForecasts, t::Integer)
    return (pred = ps.pred[t, :],
            obs = ps.obs[t],
            id = ps.id[t])
end

function Base.getindex(ps::PointForecasts, T::AbstractVector{<:Integer})
    return PointForecasts(
            ps.pred[T, :],
            ps.obs[T],
            ps.id[T])
end

function Base.getindex(qs::QuantForecasts, t::Integer)
    return (pred = qs.pred[t, :],
            obs = qs.obs[t],
            id = qs.id[t],
            prob = copy(qs.prob))
end

function Base.getindex(qs::QuantForecasts, T::AbstractVector{<:Integer})
    return QuantForecasts(
            qs.pred[T, :],
            qs.obs[T],
            qs.id[T],
            copy(qs.prob))
end

function Base.firstindex(s::Forecasts)
    return firstindex(s.obs)
end

function Base.lastindex(s::Forecasts)
    return lastindex(s.obs)
end

function Base.eachindex(s::Forecasts)
    return eachindex(s.obs)
end

"""
    findindex(s::Forecasts, i::Integer)
Return the index of `s`, for which the element of field `id` equals `i`.
"""
function findindex(s::Forecasts, i::Integer)
    t = findfirst(s.id .== i) 
    isnothing(t) && throw(error("field `id` of the provided series does not contain '$(i)'"))
    return t
end

function (s::Forecasts)(i::Integer) 
    s[findindex(s, i)]
end

function (s::Forecasts)(I::AbstractVector{<:Integer})  
    s[[findindex(s, i) for i in I]]
end

"""
    decouple(ps::PointForecasts) 
Return `::Vector{PointForecasts}`, where each element contains an individual forecast series from `ps`.
"""
function decouple(ps::PointForecasts) 
    return [PointForecasts(
            ps.pred[:, i],
            copy(ps.obs),
            copy(ps.id)) for i in 1:npred(ps)]
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
    getprob(qs::QuantForecasts[, I])
Return the copy of `prob`abilities from `qs`. 

Provide optional argument `I::Union{Integer, AbstractVector{<:Integer}}` to get `prob`abilities at specified forecast indices.
"""
function getprob(qs::QuantForecasts)
    return copy(qs.prob)
end

function getprob(qs::QuantForecasts, I::Union{Integer, AbstractVector{<:Integer}})
    return qs.prob[I]
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
function viewprob(qs::QuantForecasts)
    return @view(qs.prob[:])
end

"""
    viewprob(qs::QuantForecasts[, I])
Return the view of `prob`abilities from `qs`. 

Provide optional argument `I::Union{Integer, AbstractVector{<:Integer}}` to get `prob`abilities at specified forecast indices.
"""
function viewprob(qs::QuantForecasts, I::Union{Integer, AbstractVector{<:Integer}})
    return @view(qs.prob[I])
end
