"""
    predict(m, input, prob)
Predict quantiles at specified `prob`ability with model `m::PostModel{F}` conditional on `input`.

## Argument types 
- `input` can be of type `Number` or `AbstractVector{<:Number}`
- `prob` can be of type `AbstractFloat` (to return `F`) or `AbstractVector{<:AbstractFloat}` (to return `Vector{F}`)

## Note
For `m::QR`, `prob` argument can be ommited in the function call to return all quantiles specified in model `m`.
"""
function predict(m::UniPostModel{F}, input::Number, prob::AbstractFloat)::F where {F<:AbstractFloat}
    Base.require_one_based_indexing(prob)
    (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
    _predict(m, input, prob)
end

function predict(m::UniPostModel{F}, input::Number, prob::AbstractVector{<:AbstractFloat})::Vector{F} where {F<:AbstractFloat}
    Base.require_one_based_indexing(prob)
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict(m, input, prob)
end

function predict(m::UniPostModel{F}, input::AbstractVector{<:Number}, prob::AbstractFloat)::F where {F<:AbstractFloat}
    Base.require_one_based_indexing(input)
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
    _predict(m, input[1], prob)
end

function predict(m::UniPostModel{F}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})::Vector{F} where {F<:AbstractFloat}
    Base.require_one_based_indexing(input, prob)
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict(m, input[1], prob)
end

function predict(m::MultiPostModel{F}, input::AbstractVector{<:Number}, prob::AbstractFloat)::F where {F<:AbstractFloat}
    Base.require_one_based_indexing(input)
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
    _predict(m, input, prob)
end

function predict(m::MultiPostModel{F}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})::Vector{F} where {F<:AbstractFloat}
    Base.require_one_based_indexing(input, prob)
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict(m, input, prob)
end

function predict(m::MultiPostModel{F}, input::Number, prob::AbstractFloat)::F where {F<:AbstractFloat}
    Base.require_one_based_indexing(input)
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
    _predict(m, [input], prob)
end

function predict(m::MultiPostModel{F}, input::Number, prob::AbstractVector{<:AbstractFloat})::Vector{F} where {F<:AbstractFloat}
    Base.require_one_based_indexing(input, prob)
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict(m, [input], prob)
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
    predict!(m, output, input, prob)
In-place version of `predict` that stores the results in the `output::AbstractVector{<:AbstractFloat}` vector.

## Argument types 
- `input` can be of type `Number` or `AbstractVector{<:Number}`
- `prob` needs to be of type `AbstractVector{<:AbstractFloat}`

## Note
For `m::QR`, `prob` argument will be ignored and can be ommited in the function call.
"""
function predict!(m::UniPostModel, output::AbstractVector{<:AbstractFloat}, input::Number, prob::AbstractVector{<:AbstractFloat})::Nothing
    Base.require_one_based_indexing(output, prob)
    length(output) == length(prob) || throw(ArgumentError("length of `prob` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict!(m, output, input, prob)
end

function predict!(m::UniPostModel, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})::Nothing
    Base.require_one_based_indexing(output, input, prob)
    length(output) == length(prob) || throw(ArgumentError("length of `prob` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict!(m, output, input, prob)
end

function predict!(m::MultiPostModel, output::AbstractVector{<:AbstractFloat}, input::Number, prob::AbstractVector{<:AbstractFloat})::Nothing
    Base.require_one_based_indexing(output, prob)
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    length(output) == length(prob) || throw(ArgumentError("length of `prob` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict!(m, output, [input], prob)
end

function predict!(m::MultiPostModel, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})::Nothing
    Base.require_one_based_indexing(output, prob)
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    length(output) == length(prob) || throw(ArgumentError("length of `prob` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict!(m, output, input, prob)
end

function predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number})::Nothing
    Base.require_one_based_indexing(output, input)
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    length(m.prob) == length(output) || throw(ArgumentError("size of the output vector ($(length(output))) does not match the model specification ($(nquant(m)))"))
    _predict!(m, output, input)
end

function predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::Number)::Nothing
    Base.require_one_based_indexing(output)
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    length(m.prob) == length(output) || throw(ArgumentError("size of the output vector ($(length(output)) does not match the model specification ($(nquant(m)))"))
    _predict!(m, output, [input])
end

predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::Number, ::AbstractVector{<:AbstractFloat}) = predict!(m, output, input)
predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, ::AbstractVector{<:AbstractFloat}) = predict!(m, output, input)
