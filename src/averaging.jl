"""
    average(pf; agg::Symbol=:mean)
Average the pool of point pred from `pf`. Return `PointForecasts` containing averaged pred, keyword argument `agg` specifies whether to average using simple mean (`:mean`) or median (`:median`).

If `pf` is a vector of `PointForecasts`, all individual pred from passed PointForecasts are averaged.
"""
function average(pf::PointForecasts; agg::Symbol=:mean)
    npred(pf) == 1 && return pf
    if agg == :mean
        aggpred = vec(mean(pf.pred, dims=2))
    elseif agg == :median
        aggpred = vec(median(pf.pred, dims=2))
    else
        throw(ArgumentError("$(agg) is not a viable aggregation scheme"))
    end
    return PointForecasts(
            aggpred,
            getobs(pf),
            getid(pf))
end

function average(PF::AbstractVector{PointForecasts{F, I}}; agg::Symbol=:mean) where {F, I}
    arematching(PF)
    m = sum(npred(pf) for pf in PF) 
    m > 1 || return PF[begin]
    if agg == :mean
        aggf = mean
    elseif agg == :median
        aggf = median
    else
        throw(ArgumentError("$(agg) is not a viable aggregation scheme"))
    end
    avergedpred = Vector{F}(undef, length(PF[begin]))
    auxiliary = Vector{F}(undef, m)
    for t in eachindex(PF[begin])
        f = 0
        for pf in PF
            for i in 1:npred(pf)
                auxiliary[f+=1] = pf.pred[t, i]
            end
        end
        avergedpred[t] = aggf(auxiliary)
    end
    return PointForecasts(
        avergedpred,
        getobs(PF[begin]),
        getid(PF[begin]))
end

"""
    paverage(QF::AbstractVector{QuantForecasts}, prob)
Average probabilistic pred from `QF` by averaging probabilities of the distributions.

Return `ProbcastsSeries` containing quantile pred at specified probabilities `prob` (vector of probabilities `::AbstractVector{<:AbstractFloat}`, single probability value `::AbstractFloat` or the number of equidistant probability values `::Integer`).
"""
function paverage(QF::AbstractVector{QuantForecasts{F, I}}, prob::AbstractVector{F}) where {F, I}
    arematching(QF)
    quantiles = Matrix{F}(undef, length(QF[begin]), length(prob))
    y = Vector{F}(undef, sum(npred(qf) for qf in QF))
    cdf = Vector{F}(undef, length(y))
    order = Vector{Int}(undef, length(y))
    for t in eachindex(QF[begin])
        counter = 0
        for qf in QF
            for i in 1:npred(qf)
                y[counter += 1] = qf.pred[t, i]
                cdf[counter] = qf.prob[i] - ((i > 1) ? qf.prob[i-1] : 0.0)
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
    return QuantForecasts(
        quantiles,
        getobs(QF[begin]),
        getid(QF[begin]),
        Vector(prob))
end

paverage(QF::AbstractVector{QuantForecasts{F, I}}, prob::F) where {F, I} = paverage(QF, [prob])

paverage(QF::AbstractVector{QuantForecasts{F, I}}, prob::Integer) where {F, I} = paverage(QF, equidistant(prob, F))

"""
    qaverage(QF::AbstractVector{QuantForecasts})
Average probabilistic pred from `QF::AbstractVector{QuantForecasts}` by averaging quantiles of the distributions.

Return `ProbcastsSeries` containing quantile pred at the same prob as `QuantForecasts` stored in `QF`.
"""
function qaverage(QF::AbstractVector{QuantForecasts{F, I}}) where {F, I}
    arematching(QF, checkpred=true)
    quantiles = zeros(F, length(QF[begin]), npred(QF[begin]))
    for t in eachindex(QF[begin])
        for i in 1:npred(QF[begin])
            for qf in QF
                quantiles[t, i] += qf.pred[t, i]
            end
        end
    end
    quantiles /= length(QF)
    return QuantForecasts(
        quantiles,
        getobs(QF[begin]),
        getid(QF[begin]),
        getprob(QF[begin]))
end
