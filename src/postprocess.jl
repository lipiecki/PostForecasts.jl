"""
    point2prob(pf::PointForecasts, window::Integer, modelname::Symbol, prob[; first = window + 1, last = length(pf), recalibration = 1])
Compute probabilistic forecast based on point forecasts `pf`. Probabilistic forecast will be calculated for observations between the index `first` and `last` of `pf`. The model is calibrated on the last `window` observations, and recalibrated every `recalibration` steps.

Available options for `model`:
- `:qr` for Quantile Regression Averaging
- `:cp` for Conformal Prediction
- `:hs` for Conformal Prediction with non-symmetric errors (a.k.a. Historical Simulation)
- `:idr` for Isotonic Distributional Regression
- `:normal` for Normal distribution of errors
- `:zeronormal` for Normal distribution of errors with fixed mean equal to 0.

`QR` supports multiple regressors; `IDR` partially supports multiple regressors - one isotonic regression is fitted to each forecast and the final predictive distribution is an average of individual distributions; `CP` and `Normal` do not support multiple regressors.

Return QuantForecasts containing quantile forecasts at specified probabilities `prob` (vector of probabilities `::AbstractVector{<:AbstractFloat}`, single probability value `::AbstractFloat` or the number of equidistant probability values `::Integer`).
"""
function point2prob(pf::PointForecasts{F, I}, window::Integer, modelname::Symbol, prob::Vector{<:AbstractFloat}; first::Integer=window+firstindex(pf), last::Integer=lastindex(pf), recalibration::Integer=1) where {F, I}
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    prob = Vector{F}(prob)
    model = getmodel(Val(modelname), window, npred(pf), prob)
    pf = (nreg(model) == 1 && npred(pf) > 1) ? average(pf) : pf
    _point2prob(pf, window, model, prob, first, last, recalibration)
end

function point2prob(pf::PointForecasts{F, I}, window::Integer, modelname::Symbol, prob::AbstractFloat; first::Integer=window+firstindex(pf), last::Integer=lastindex(pf), recalibration::Integer=1) where {F, I}
    (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
    prob = [F(prob)]
    model = getmodel(Val(modelname), window, npred(pf), prob)
    pf = (nreg(model) == 1 && npred(pf) > 1) ? average(pf) : pf
    _point2prob(pf, window, model, prob, first, last, recalibration)
end
    
function point2prob(pf::PointForecasts{F, I}, window::Integer, modelname::Symbol, prob::Integer; first::Integer=window+firstindex(pf), last::Integer=lastindex(pf), recalibration::Integer=1) where {F, I}
    prob > 0 || throw(ArgumentError("number of quantiles must be greater than 0"))   
    prob = equidistant(prob, F)
    model = getmodel(Val(modelname), window, npred(pf), prob)
    pf = (nreg(model) == 1 && npred(pf) > 1) ? average(pf) : pf
    _point2prob(pf, window, model, prob, first, last, recalibration)
end

"""
    conformalize(qf::QuantForecasts{F, I}, window::Integer[; first::Integer, last::Integer)
Perform [conformalization](https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf) of quantile forecasts provided in `ps`.
Conformalized quantiles will be calculated for observations between the index `first` and `last` of `qf`. The model is calibrated on the last `window` observations.

Return `QuantForecasts` with conformalized quantiles.
"""
function conformalize(qf::QuantForecasts{F, I}, window::Integer; first::Integer=window+firstindex(qf), last::Integer=lastindex(qf)) where {F, I} 
    quantiles = getpred(qf, first:last)
    model = CP(window, abs=false)
    for t in first:last
        for i in 1:npred(qf)
            train(model, viewpred(qf, t-window:t-1, i), viewobs(qf, t-window:t-1))
            quantiles[t - first + 1, i] = _predict(model, quantiles[t - first + 1, i], 1.0 - getprob(qf, i))
        end
    end
    sort!(quantiles, dims = 2)
    return QuantForecasts(
        quantiles,
        getobs(qf, first:last),
        getid(qf, first:last),
        getprob(qf))
end

"""
    conformalize!(qf::QuantForecasts{F, I}, window::Integer[; first::Integer, last::Integer)
In-place version of conformalize, that mutates `qf` instead of creating a new `QuantForecasts`.
"""
function conformalize!(qf::QuantForecasts{F, I}, window::Integer; first::Integer=window+firstindex(qf), last::Integer=lastindex(qf)) where {F, I}
    model = CP(window, abs=false) 
    for t in last:-1:first
        for i in 1:npred(qf)
            train(model, viewpred(qf, t-window:t-1, i), viewobs(qf, t-window:t-1))
            setpred(qf, t, i, _predict(model, getpred(qf, t, i), 1.0 - getprob(qf, i)))
        end
        sort!(viewpred(qf, t))
    end
end
