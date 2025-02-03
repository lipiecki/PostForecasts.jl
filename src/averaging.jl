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
    paverage(QF::Vector{QuantForecasts}; quantiles=99)
Average probabilistic pred from `QF` by averaging probabilities of the distributions.

Return `QuantForecasts` containing quantile pred at specified `quantiles`:
- `quantiles::AbstractVector{<:AbstractFloat}`: vector of probabilities
- `quantiles::AbstractFloat`: a single probability value
- `quantiles::Integer`: number of equidistant probability values (e.g. 99 for percentiles).
"""
function paverage(QF::Vector{QuantForecasts{F, I}}, quantiles=99) where {F, I}
    paverage(QF, quantiles)
end

function paverage(QF::Vector{QuantForecasts{F, I}}; quantiles::AbstractVector{<:AbstractFloat}) where {F, I}
    checkmatch(QF)
    prob = Vector{F}(quantiles)
    pred = Matrix{F}(undef, length(QF[begin]), length(prob))
    y = Vector{F}(undef, sum(npred(qf) for qf in QF))
    pdf = Vector{F}(undef, length(y))
    for t in eachindex(QF[begin])
        cdf::F = 0.0
        itr = 0
        for qf in QF
            for i in 1:npred(qf)
                y[itr += 1] = getpred(qf, t, i)
                pdf[itr] = getprob(qf, i) - ((i > 1) ? getprob(qf, i-1) : 0.0)
                cdf += pdf[itr]
            end
        end
        cdf /= length(QF)
        pdf /= length(QF)
        itr = lastindex(prob)
        for i in sortperm(y, rev=true)
            cdf -= pdf[i]
            pred[t, itr] = y[i]
            if !(cdf >= prob[itr] || cdf ≈ prob[itr])
                itr -= 1
                itr == 0 && break
            end
        end
    end
    return QuantForecasts(
        pred,
        getobs(QF[begin]),
        getid(QF[begin]),
        prob,
        Val(false))
end

paverage(QF::Vector{QuantForecasts{F, I}}; quantiles::AbstractFloat) where {F, I} = paverage(QF, [quantiles])

paverage(QF::Vector{QuantForecasts{F, I}}; quantiles::Integer) where {F, I} = paverage(QF, equidistant(quantiles, F))

"""
    qaverage(QF::Vector{QuantForecasts})
Average probabilistic pred from `QF::Vector{QuantForecasts}` by averaging quantiles of the distributions.

Return `QuantForecasts` containing quantile pred at the same prob as `QuantForecasts` stored in `QF`.
"""
function qaverage(QF::Vector{QuantForecasts{F, I}}) where {F, I}
    checkmatch(QF, checkpred=true)
    pred = zeros(F, length(QF[begin]), npred(QF[begin]))
    for t in eachindex(QF[begin])
        for i in 1:npred(QF[begin])
            for qf in QF
                pred[t, i] += qf.pred[t, i]
            end
        end
    end
    pred /= length(QF)
    return QuantForecasts(
        pred,
        getobs(QF[begin]),
        getid(QF[begin]),
        getprob(QF[begin]), 
        Val(false))
end
