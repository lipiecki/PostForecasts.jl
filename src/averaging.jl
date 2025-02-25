"""
    average(pf; agg::Symbol=:mean)
Average the pool of point pred from `pf`. Return `PointForecasts` containing averaged forecasts, keyword argument `agg` specifies whether to average using simple mean (`:mean`) or median (`:median`).

## Argument types
- `pf::PointForecasts` to average the pool of forecasts in `pf`
- `pf::PointForecasts...` to average all individual forecasts from every `PointForecasts` in `pf`.
- `pf::AbstractVector{<:PointForecasts}` to average all individual forecasts from every `PointForecasts` in `pf`.
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

function average(pfs::AbstractVector{<:PointForecasts}; agg::Symbol=:mean)
    checkmatch(pfs)
    m = sum(npred.(pfs)) 
    m > 1 || return pfs[begin]
    (agg == :mean || agg == :median) || throw(ArgumentError("$(agg) is not a viable aggregation scheme"))
    avepred = Vector{Float64}(undef, length(pfs[begin]))
    auxiliary = Vector{Float64}(undef, m)
    for t in eachindex(pfs[begin])
        i = 0
        for pf in pfs
            for j in 1:npred(pf)
                auxiliary[i+=1] = getpred(pf, t, j)
            end
        end
        avepred[t] = agg == :mean ? mean(auxiliary) : median(auxiliary)
    end
    return PointForecasts(
        avepred,
        getobs(pfs[begin]),
        getid(pfs[begin]))
end

function average(pfs::Vararg{PointForecasts, N}; agg::Symbol=:mean) where N
    v = Vector{PointForecasts}(undef, N)
    v .= pfs
    average(v; agg=agg)
end

"""
    paverage(qfs[; quantiles])
Average probabilistic predictions from a collection `qfs` by averaging the distributions across probability.

Return `QuantForecasts` containing predictions of specified `quantiles`:
- `quantiles::AbstractVector{<:AbstractFloat}`: vector of probabilities
- `quantiles::AbstractFloat`: a single probability value
- `quantiles::Integer`: number of equidistant probability values (e.g. 99 for percentiles).

If `quantiles` argument is not provided, the function will default to the quantiles of the first `QuantForecasts` in `qfs`.
"""
paverage(qfs::AbstractVector{<:QuantForecasts}; quantiles=getprob(first(qfs))) = paverage(qfs, quantiles)

function paverage(qfs::Vararg{QuantForecasts, N}; quantiles) where N
    v = Vector{QuantForecasts}(undef, N)
    v .= qfs
    paverage(v, quantiles) 
end

function paverage(qfs::AbstractVector{<:QuantForecasts}, quantiles::AbstractVector{<:AbstractFloat})
    checkmatch(qfs)
    issorted(quantiles) || throw(ArgumentError("`quantiles` vector has to be sorted"))
    (quantiles[begin] > 0.0 && quantiles[end] < 1.0) || throw(ArgumentError("elements of `quantiles` must belong to an open (0, 1) interval"))
    prob = Vector{Float64}(quantiles)
    pred = Matrix{Float64}(undef, length(qfs[begin]), length(prob))
    y = Vector{Float64}(undef, sum(npred(qf) for qf in qfs))
    Δcdf = Vector{Float64}(undef, length(y))
    for t in eachindex(qfs[begin])
        i = 1
        for qf in qfs
            for j in 1:npred(qf)
                y[i] = getpred(qf, t, j)
                Δcdf[i] = getprob(qf, j) - ((j > 1) ? getprob(qf, j-1) : 0.0)
                i += 1
            end
        end
        Δcdf /= length(qfs)
        i = 1
        cdf = 0.0
        ỹ = -Inf
        for j in sortperm(y)
            cdf += Δcdf[j]
            ỹ = y[j]
            while cdf >= prob[i] || cdf ≈ prob[i]
                pred[t, i] = ỹ
                i == lastindex(prob) && @goto loopend
                i += 1
            end
        end
        pred[t, i:end] .= ỹ
        @label loopend
    end
    return QuantForecasts(
        pred,
        getobs(qfs[begin]),
        getid(qfs[begin]),
        prob,
        Val(false))
end

paverage(qfs::AbstractVector{<:QuantForecasts}, quantiles::AbstractFloat) = paverage(qfs, [quantiles])

paverage(qfs::AbstractVector{<:QuantForecasts}, quantiles::Integer) = paverage(qfs, equidistant(quantiles))

"""
    qaverage(qfs)
Average probabilistic predictions from a collection `qfs` by averaging the quantiles.

Return `QuantForecasts` containing quantile predictions at the same quantile levels as `QuantForecasts` in `qfs`.
"""
function qaverage(qfs::AbstractVector{<:QuantForecasts})
    checkmatch(qfs; checkpred=true)
    pred = zeros(length(qfs[begin]), npred(qfs[begin]))
    for t in eachindex(qfs[begin])
        for i in 1:npred(qfs[begin])
            for qf in qfs
                pred[t, i] += getpred(qf, t, i)
            end
        end
    end
    pred /= length(qfs)
    return QuantForecasts(
        pred,
        getobs(qfs[begin]),
        getid(qfs[begin]),
        getprob(qfs[begin]), 
        Val(false))
end

function qaverage(qfs::Vararg{QuantForecasts, N}) where N
    v = Vector{QuantForecasts}(undef, N)
    v .= qfs
    qaverage(v)
end
