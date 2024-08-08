"""
    predict(m, input, prob)
Predict quantiles at specified `prob`abilbities using model `m::ProbModel` with `input`.

## Argument types 
- `input` can be of type `Number` or `AbstractVector{<:Number}`
- `prob` can be of type `AbstractFloat` (to return `Float64` value) or `AbstractVector{<:AbstractFloat}` (to return `Vector{Float64}`)

## Note
For `m::QR`, calling `predict` with `prob::AbstractFloat` will match the probability to one of the quantiles of the model and return an approriate value,
but `prob::AbstractVector{<:AbstractFloat}` will be ignored.
"""
function predict(m::ProbModel, input::Number, prob::AbstractFloat)::Float64
    Base.require_one_based_indexing(prob)
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
    _predict(m, input, prob)
end

function predict(m::ProbModel, input::Number, prob::AbstractVector{<:AbstractFloat})::Vector{Float64}
    Base.require_one_based_indexing(prob)
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    output = Vector{Float64}(undef, length(prob))
    _predict!(m, output, input, prob)
    return output
end

function predict(m::UniRegProbModel, input::AbstractVector{<:Number}, prob::AbstractFloat)::Float64
    Base.require_one_based_indexing(input)
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
    _predict(m, input[1], prob)
end

function predict(m::MultiRegProbModel, input::AbstractVector{<:Number}, prob::AbstractFloat)::Float64
    Base.require_one_based_indexing(input)
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
    _predict(m, input, prob)
end

function predict(m::UniRegProbModel, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})::Vector{Float64}
    Base.require_one_based_indexing(input, prob)
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    output = Vector{Float64}(undef, length(prob))
    _predict!(m, output, input[1], prob)
    return output
end

function predict(m::MultiRegProbModel, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})::Vector{Float64}
    Base.require_one_based_indexing(input, prob)
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    output = Vector{Float64}(undef, length(prob))
    _predict!(m, output, input, prob)
    return output
end

function predict(m::QR, input::AbstractVector{<:Number})::Vector{Float64}
    Base.require_one_based_indexing(input)
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    output = Vector{Float64}(undef, nquant(m))
    _predict!(m, output, input)
    return output
end

function predict(m::QR, input::Number)::Vector{Float64}
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    output = Vector{Float64}(undef, nquant(m))
    _predict!(m, output, input)
    return output
end

"""
    predict!(m, output, input[, prob])
In-place version of `predict` that stores the results in the `output::AbstractVector{<:AbstractFloat}` vector.

## Argument types 
- `input` can be of type `Number` or `AbstractVector{<:Number}`
- `prob` needs to be of type `AbstractVector{<:AbstractFloat}`

## Note
For `m::QR`, `prob` will be ignored.
"""
function predict!(m::ProbModel, output::AbstractVector{<:AbstractFloat}, input::Number, prob::AbstractVector{<:AbstractFloat})
    Base.require_one_based_indexing(output, prob)
    length(output) == length(prob) || throw(ArgumentError("length of `prob` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict!(m, output, input, prob)
end

function predict!(m::UniRegProbModel, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})
    Base.require_one_based_indexing(output, input, prob)
    length(output) == length(prob) || throw(ArgumentError("length of `prob` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict!(m, output, input[1], prob)
end

function predict!(m::MultiRegProbModel, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})
    Base.require_one_based_indexing(output, input, prob)
    length(output) == length(prob) || throw(ArgumentError("length of `prob` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict!(m, output, input, prob)
end

function predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number})
    Base.require_one_based_indexing(output, input)
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    _predict!(m, output, input)
end

function predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::Number)
    Base.require_one_based_indexing(output)
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    _predict!(m, output, input)
end
