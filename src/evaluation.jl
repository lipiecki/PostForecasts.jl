"""
    mae(pf::PointForecasts)
Calculate Mean Absolute Error of pred from `pf`. Return the vector of MAE corresponding to each forecaster.
"""
function mae(pf::PointForecasts)
    loss = zeros(npred(pf))
    for i in 1:npred(pf)
        for t in eachindex(pf)
            loss[i] += abs(getobs(pf, t) - getpred(pf, t, i))
        end
    end
    loss /= length(pf)
    return loss
end

"""
    mape(pf::PointForecasts; eps=1e-9)
Calculate Mean Absolute Percentage Error of pred from `pf`. Return the vector of MAPE corresponding to each forecaster.
"""
function mape(pf::PointForecasts; eps::AbstractFloat=1e-9)
    loss = zeros(npred(pf))
    for i in 1:npred(pf)
        for t in eachindex(pf)
            loss[i] += abs(getobs(pf, t) - getpred(pf, t, i))/max(abs(getobs(pf, t)), eps)
        end
    end
    loss *= 100.0/length(pf)
    return loss
end

"""
    smape(pf::PointForecasts; eps=1e-9)
Calculate Symmetric Mean Absolute Percentage Error of pred from `pf`. Return the vector of SMAPE corresponding to each forecaster.
"""
function smape(pf::PointForecasts; eps::AbstractFloat=1e-9)
    loss = zeros(npred(pf))
    for i in 1:npred(pf)
        for t in eachindex(pf)
            loss[i] += 2abs(getobs(pf, t) - getpred(pf, t, i))/(max(abs(getobs(pf, t)), eps) + max(abs(getpred(pf, t, i)), eps))
        end
    end
    loss *= 100.0/length(pf)
    return loss
end


"""
    mse(pf::PointForecasts)
Calculate Mean Squared Error of pred from `pf`. Return the vector of MSE corresponding to each forecaster.
"""
function mse(pf::PointForecasts)
    loss = zeros(npred(pf))
    for i in 1:npred(pf)
        for t in eachindex(pf)
            loss[i] += (getobs(pf, t) - getpred(pf, t, i))^2
        end
    end
    loss /= length(pf)
    return loss
end

"""
    pinball(qf::QuantForecasts)
Calculate Pinball Loss over all quantiles in `qf`. Return the vector of Pinball Loss values corresponding to each quantile.
See [Gneiting 2011](https://doi.org/10.1016/j.ijforecast.2009.12.015) for more details about Pinball Loss. 

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
Approximate Continous Ranked Probability Score using the Pinball Loss of quantile forecasts in `qf`, with `2mean(pinball(qf))`.

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
