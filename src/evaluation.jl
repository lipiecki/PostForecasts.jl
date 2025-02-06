"""
    mae(pf::PointForecasts)
Calculate Mean Absolute Error of pred from `pf`. Return the vector of MAE corresponding to each forecaster.
"""
function mae(pf::PointForecasts)
    loss = zeros(npred(pf))
    for i in 1:npred(pf)
        for t in eachindex(pf)
            loss[i] += abs(getobs(pf, t) -  getpred(pf, t, i))
        end
    end
    loss /= length(pf)
    return loss
end

"""
    rmse(pf::PointForecasts)
Calculate Root Mean Squared Error of pred from `pf`. Return the vector of RMSE corresponding to each forecaster.
"""
function rmse(pf::PointForecasts)
    loss = zeros(npred(pf))
    for i in 1:npred(pf)
        for t in eachindex(pf)
            loss[i] += (getobs(pf, t) - getpred(pf, t, i))^2
        end
    end
    loss /= length(pf)
    return sqrt.(loss)
end

"""
    pinball(qf::QuantForecasts)
Calculate the Pinball Loss over all quantiles in `qf`. Return the vector of Pinball Loss values corresponding to each quantile.
See [Gneiting 2011](httqf://doi.org/10.1016/j.ijforecast.2009.12.015) for more details about Pinball Loss. 

## Note
Average Pinball Loss over equidistant quantiles approximates Continuous Ranked Probability Score.
"""
function pinball(qf::QuantForecasts)
    loss = zeros(npred(qf))
    for i in 1:npred(qf)
        for t in eachindex(qf)
            if getobs(qf, t) < getpred(qf, t, i)
                loss[i] += (getpred(qf, t, i) - getobs(qf, t))*(1-getprob(qf, i))
            else
                loss[i] += (getobs(qf, t) - getpred(qf, t, i))*getprob(qf, i)
            end
        end
    end
    loss /= length(qf)
    return loss
end

"""
    crps(qf::QuantForecasts)
Approximate the Continous Ranked Probability Score using the Pinball Loss of quantile forecasts in `qf`, with `2mean(pinball(qf))`.

## Note
Approximating CRPS with the average Pinball Loss requires a dense grid of equidistant quantiles.
"""
function crps(qf::QuantForecasts{F, I}) where {F, I}
    all(equidistant(npred(qf), F) .≈ viewprob(qf)) || @warn "improper CRPS approximation: quantile grid of `qf` is not uniform"
    return 2mean(pinball(qf))
end

"""
    coverage(qf::QuantForecasts)
Calculate empirical coverage of quantile pred in `qf`. Return the vector of coverage corresponding to each quantile. 
"""
function coverage(qf::QuantForecasts)
    cover = zeros(UInt, npred(qf))
    for i in 1:npred(qf)
        for t in eachindex(qf)
            if getobs(qf, t) <= getpred(qf, t, i)
                cover[i] += 1
            end
        end
    end
    return cover/length(qf)
end
