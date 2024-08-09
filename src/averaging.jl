"""
    average(pf; agg::Symbol=:mean)
Average the pool of point pred from `pf`. Return `PointForecasts` containing averaged forecasts, keyword argument `agg` specifies whether to average using simple mean (`:mean`) or median (`:median`).

## Argument types
- `pf::PointForecasts` to average the pool of forecats in `pf`
- `pf::Vector{PointForecasts}` to average all individual forecasts from every `PointForecasts` in `pf`.
"""
function average(pf::PointForecasts; agg::Symbol=:mean)
    npred(pf) == 1 && return pf
    if agg == :mean
        aggpred = vec(mean(viewpred(pf), dims=2))
    elseif agg == :median
        aggpred = vec(median(viewpred(pf), dims=2))
    else
        throw(ArgumentError("$(agg) is not a viable aggregation scheme"))
    end
    return PointForecasts(
            aggpred,
            getobs(pf),
            getid(pf))
end

function average(PF::Vector{PointForecasts{F, I}}; agg::Symbol=:mean) where {F, I}
    checkmatch(PF)
    m = sum(npred(pf) for pf in PF) 
    m > 1 || return PF[begin]
    (agg == :mean || agg == :median) || throw(ArgumentError("$(agg) is not a viable aggregation scheme"))
    avergedpred = Vector{F}(undef, length(PF[begin]))
    auxiliary = Vector{F}(undef, m)
    for t in eachindex(PF[begin])
        f = 0
        for pf in PF
            for i in 1:npred(pf)
                auxiliary[f+=1] = getpred(pf, t, i)
            end
        end
        avergedpred[t] = agg == :mean ? mean(auxiliary) : median(auxiliary)
    end
    return PointForecasts(
        avergedpred,
        getobs(PF[begin]),
        getid(PF[begin]))
end

"""
    paverage(QF::Vector{QuantForecasts}, prob)
Average probabilistic pred from `QF` by averaging probabilities of the distributions.

Return `QuantForecasts` containing quantile pred at specified `prob`abilities:
- `prob::AbstractVector{<:AbstractFloat}`: vector of probabilities
- `prob::AbstractFloat`: a single probability value
- `prob::Integer`: number of equidistant probability values (e.g. 99 for percentiles).
"""
function paverage(QF::Vector{QuantForecasts{F, I}}, prob::AbstractVector{<:AbstractFloat}) where {F, I}
    Base.require_one_based_indexing(prob)
    checkmatch(QF)
    prob = Vector{F}(prob)
    quantiles = Matrix{F}(undef, length(QF[begin]), length(prob))
    y = Vector{F}(undef, sum(npred(qf) for qf in QF))
    cdf = Vector{F}(undef, length(y))
    order = Vector{Int}(undef, length(y))
    for t in eachindex(QF[begin])
        counter = 0
        for qf in QF
            for i in 1:npred(qf)
                y[counter += 1] = qf.pred[t, i]
                cdf[counter] = qf.prob[i] - ((i > 1) ? qf.prob[i-1] : F(0.0))
            end
        end
        sortperm!(order, y)
        cdf /= length(QF)
        p = 0.0
        for i in order
            p += cdf[i]
            cdf[i] = p
        end
        cdf2quantiles!(@view(quantiles[t, :]), @view(cdf[order]), @view(y[order]), prob)
    end
    return QuantForecasts(Val(:nocopy),
        quantiles,
        getobs(QF[begin]),
        getid(QF[begin]),
        prob)
end

paverage(QF::Vector{QuantForecasts{F, I}}, prob::AbstractFloat) where {F, I} = paverage(QF, [prob])

paverage(QF::Vector{QuantForecasts{F, I}}, prob::Integer) where {F, I} = paverage(QF, equidistant(prob, F))

"""
    qaverage(QF::Vector{QuantForecasts})
Average probabilistic pred from `QF::Vector{QuantForecasts}` by averaging quantiles of the distributions.

Return `QuantForecasts` containing quantile pred at the same prob as `QuantForecasts` stored in `QF`.
"""
function qaverage(QF::Vector{QuantForecasts{F, I}}) where {F, I}
    checkmatch(QF, checkpred=true)
    quantiles = zeros(F, length(QF[begin]), npred(QF[begin]))
    for t in eachindex(QF[begin])
        for i in 1:npred(QF[begin])
            for qf in QF
                quantiles[t, i] += qf.pred[t, i]
            end
        end
    end
    quantiles /= length(QF)
    return QuantForecasts(Val(:nocopy),
        quantiles,
        getobs(QF[begin]),
        getid(QF[begin]),
        getprob(QF[begin]))
end
