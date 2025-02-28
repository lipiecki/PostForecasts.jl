"""
    predict(m, input, quantiles)
Predict the specified `quantiles` of the predictive distribution from model `m::PostModel{F}` conditional on `input`.

## Argument types 
- `input` can be of type `Number` or `AbstractVector{<:Number}`
- `quantiles` can be of type `AbstractFloat` (to return a single prediction) or `AbstractVector{<:AbstractFloat}` (to return a vector of predictions)

## Note
For `m::QR`, the `quantiles` argument can be ommited in the function call to return all quantiles specified in model `m`.
"""
function predict(m::UniPostModel{F}, input::Number, quantiles::AbstractFloat)::F where {F<:AbstractFloat}
    Base.require_one_based_indexing(quantiles)
    (quantiles > 0.0 && quantiles < 1.0) || throw(ArgumentError("`quantiles` must belong to an open (0, 1) interval"))
    _predict(m, input, quantiles)
end

function predict(m::UniPostModel{F}, input::Number, quantiles::AbstractVector{<:AbstractFloat})::Vector{F} where {F<:AbstractFloat}
    Base.require_one_based_indexing(quantiles)
    issorted(quantiles) || throw(ArgumentError("`quantiles` vector has to be sorted"))
    (quantiles[begin] > 0.0 && quantiles[end] < 1.0) || throw(ArgumentError("elements of `quantiles` must belong to an open (0, 1) interval"))
    _predict(m, input, quantiles)
end

function predict(m::UniPostModel{F}, input::AbstractVector{<:Number}, quantiles::AbstractFloat)::F where {F<:AbstractFloat}
    Base.require_one_based_indexing(input)
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    (quantiles > 0.0 && quantiles < 1.0) || throw(ArgumentError("`quantiles` must belong to an open (0, 1) interval"))
    _predict(m, input[1], quantiles)
end

function predict(m::UniPostModel{F}, input::AbstractVector{<:Number}, quantiles::AbstractVector{<:AbstractFloat})::Vector{F} where {F<:AbstractFloat}
    Base.require_one_based_indexing(input, quantiles)
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    issorted(quantiles) || throw(ArgumentError("`quantiles` vector has to be sorted"))
    (quantiles[begin] > 0.0 && quantiles[end] < 1.0) || throw(ArgumentError("elements of `quantiles` must belong to an open (0, 1) interval"))
    _predict(m, input[1], quantiles)
end

function predict(m::MultiPostModel{F}, input::AbstractVector{<:Number}, quantiles::AbstractFloat)::F where {F<:AbstractFloat}
    Base.require_one_based_indexing(input)
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    (quantiles > 0.0 && quantiles < 1.0) || throw(ArgumentError("`quantiles` must belong to an open (0, 1) interval"))
    _predict(m, input, quantiles)
end

function predict(m::MultiPostModel{F}, input::AbstractVector{<:Number}, quantiles::AbstractVector{<:AbstractFloat})::Vector{F} where {F<:AbstractFloat}
    Base.require_one_based_indexing(input, quantiles)
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    issorted(quantiles) || throw(ArgumentError("`quantiles` vector has to be sorted"))
    (quantiles[begin] > 0.0 && quantiles[end] < 1.0) || throw(ArgumentError("elements of `quantiles` must belong to an open (0, 1) interval"))
    _predict(m, input, quantiles)
end

function predict(m::MultiPostModel{F}, input::Number, quantiles::AbstractFloat)::F where {F<:AbstractFloat}
    Base.require_one_based_indexing(input)
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    (quantiles > 0.0 && quantiles < 1.0) || throw(ArgumentError("`quantiles` must belong to an open (0, 1) interval"))
    _predict(m, [input], quantiles)
end

function predict(m::MultiPostModel{F}, input::Number, quantiles::AbstractVector{<:AbstractFloat})::Vector{F} where {F<:AbstractFloat}
    Base.require_one_based_indexing(input, quantiles)
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    issorted(quantiles) || throw(ArgumentError("`quantiles` vector has to be sorted"))
    (quantiles[begin] > 0.0 && quantiles[end] < 1.0) || throw(ArgumentError("elements of `quantiles` must belong to an open (0, 1) interval"))
    _predict(m, [input], quantiles)
end

function predict(m::QR{F}, input::AbstractVector{<:Number})::Vector{F} where {F<:AbstractFloat}
    Base.require_one_based_indexing(input)
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    _predict(m, input)
end

function predict(m::QR{F}, input::Number)::Vector{F} where {F<:AbstractFloat}
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    _predict(m, [input])
end

"""
    predict!(m, output, input, quantiles)
In-place version of `predict` that stores the results in the `output::AbstractVector{<:AbstractFloat}` vector.

## Argument types 
- `input` can be of type `Number` or `AbstractVector{<:Number}`
- `quantiles` needs to be of type `AbstractVector{<:AbstractFloat}`

## Note
For `m::QR`, `quantiles` argument will be ignored and can be ommited in the function call.
"""
function predict!(m::UniPostModel, output::AbstractVector{<:AbstractFloat}, input::Number, quantiles::AbstractVector{<:AbstractFloat})::Nothing
    Base.require_one_based_indexing(output, quantiles)
    length(output) == length(quantiles) || throw(ArgumentError("length of `quantiles` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    issorted(quantiles) || throw(ArgumentError("`quantiles` vector has to be sorted"))
    (quantiles[begin] > 0.0 && quantiles[end] < 1.0) || throw(ArgumentError("elements of `quantiles` must belong to an open (0, 1) interval"))
    _predict!(m, output, input, quantiles)
end

function predict!(m::UniPostModel, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, quantiles::AbstractVector{<:AbstractFloat})::Nothing
    Base.require_one_based_indexing(output, input, quantiles)
    length(output) == length(quantiles) || throw(ArgumentError("length of `quantiles` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    issorted(quantiles) || throw(ArgumentError("`quantiles` vector has to be sorted"))
    (quantiles[begin] > 0.0 && quantiles[end] < 1.0) || throw(ArgumentError("elements of `quantiles` must belong to an open (0, 1) interval"))
    _predict!(m, output, input, quantiles)
end

function predict!(m::MultiPostModel, output::AbstractVector{<:AbstractFloat}, input::Number, quantiles::AbstractVector{<:AbstractFloat})::Nothing
    Base.require_one_based_indexing(output, quantiles)
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    length(output) == length(quantiles) || throw(ArgumentError("length of `quantiles` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    issorted(quantiles) || throw(ArgumentError("`quantiles` vector has to be sorted"))
    (quantiles[begin] > 0.0 && quantiles[end] < 1.0) || throw(ArgumentError("elements of `quantiles` must belong to an open (0, 1) interval"))
    _predict!(m, output, [input], quantiles)
end

function predict!(m::MultiPostModel, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, quantiles::AbstractVector{<:AbstractFloat})::Nothing
    Base.require_one_based_indexing(output, quantiles)
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    length(output) == length(quantiles) || throw(ArgumentError("length of `quantiles` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    issorted(quantiles) || throw(ArgumentError("`quantiles` vector has to be sorted"))
    (quantiles[begin] > 0.0 && quantiles[end] < 1.0) || throw(ArgumentError("elements of `quantiles` must belong to an open (0, 1) interval"))
    _predict!(m, output, input, quantiles)
end

function predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number})::Nothing
    Base.require_one_based_indexing(output, input)
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    length(m.prob) == length(output) || throw(ArgumentError("size of the output vector ($(length(output))) does not match the model specification ($(length(m.prob)))"))
    _predict!(m, output, input)
end

function predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::Number)::Nothing
    Base.require_one_based_indexing(output)
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    length(m.prob) == length(output) || throw(ArgumentError("size of the output vector ($(length(output)) does not match the model specification ($(length(m.prob)))"))
    _predict!(m, output, [input])
end

predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::Number, ::AbstractVector{<:AbstractFloat}) = predict!(m, output, input)
predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, ::AbstractVector{<:AbstractFloat}) = predict!(m, output, input)
