"""
    train(m, X, Y)
Calibrate the model `m` on the covariates `X` and responses `Y`.

In general, `X` should be a matrix, which columns correspond to respective regressors.
The number of regressors must match the specification of the model.

For `m::UniPostModel`, `X` can be a vector, if it is a matrix with multiple columns, they will be averaged before training.
"""
function train(m::UniPostModel, X::AbstractVector{<:Number}, Y::AbstractVector{<:Number})::Nothing
    Base.require_one_based_indexing(X, Y)
    length(X) == length(Y) || throw(ArgumentError("lengths of `X` and `Y` do not match"))
    matchwindow(m, size(X, 1)) || throw(ArgumentError("length of `X` and `Y` does not match model specification"))
    _train(m, X, Y)
end

function train(m::UniPostModel, X::AbstractMatrix{<:Number}, Y::AbstractVector{<:Number})::Nothing
    Base.require_one_based_indexing(X, Y)
    size(X, 1) == length(Y) || throw(ArgumentError("lengths of `X` and `Y` do not match"))
    size(X, 2) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(size(X, 2)) were provided"))
    matchwindow(m, size(X, 1)) || throw(ArgumentError("length of `X` and `Y` does not match model specification"))
    _train(m, X, Y)
end

function train(m::MultiPostModel, X::AbstractVector{<:Number}, Y::AbstractVector{<:Number})::Nothing
    Base.require_one_based_indexing(X, Y)
    length(X) == length(Y) || throw(ArgumentError("lengths of `X` and `Y` do not match"))
    matchwindow(m, size(X, 1)) || throw(ArgumentError("length of `X` and `Y` does not match model specification"))
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    _train(m, X, Y)
end

function train(m::MultiPostModel, X::AbstractMatrix{<:Number}, Y::AbstractVector{<:Number})::Nothing
    Base.require_one_based_indexing(X, Y)
    size(X, 1) == length(Y) || throw(ArgumentError("lengths of `X` and `Y` do not match"))
    matchwindow(m, size(X, 1)) || throw(ArgumentError("length of `X` and `Y` does not match model specification"))
    nreg(m) == size(X, 2) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(size(X, 2)) were provided"))
    _train(m, X, Y)
end
