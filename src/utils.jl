"""
    getmodel(::Val, params...)
Helper function that dispatches the model based on the model name passed as `Val`.

## Available methods
- `getmodel(Val(:qra, n, m, prob), ` for Quantile Regression Averaging
- `getmodel(Val(:cp), n)` for Conformal Prediction
- `getmodel(Val(:hs), n)` for Conformal Prediction Prediction with non-symmetric errors (a.k.a. Historical Simulation)
- `getmodel(Val(:idr), n, m)` for Isotonic Distributional Regression
- `getmodel(Val(:normal))` for Normal distribution of errors
- `getmodel(Val(:zeronormal))` for Normal distribution of errors with fixed mean equal to 0

where `n` is the length of the calibration window, `m` is the number of regressors and `prob` is the probability (scalar value or vector).

Return an appropriate `ProbModel`.
"""
function getmodel(::Val, ::Vararg)
    throw(ArgumentError("provided model name not recognized"))
end

"""
    nreg(m::ProbModel)
Return the number of regressors of model `m`.
"""
nreg(::UniRegProbModel) = 1

"""
    matchwindow(m::ProbModel, window::Integer)
Return `true` if `window` matches the specification of model `m`, otherwise return `false`.
"""
matchwindow(::ProbModel, ::Integer) = true