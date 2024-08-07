"""
    predict(m, input, prob)
Predict quantiles at specified `prob::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}` using model `m::ProbModel` conditional on `input::Union{Number, AbstractVector{<:Number}}`.

**Note:** for `m::QR` ...
"""
function predict(m::UniRegProbModel, input::Number, prob::AbstractFloat)
    (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
    _predict(m, input, prob)
end

function predict(m::MultiRegProbModel, input::AbstractVector{<:Number}, prob::AbstractFloat)
    (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
    output = [0.0]
    _predict!(m, output, input, [prob])
    output[1]
end

function predict(m::UniRegProbModel, input::Number, prob::AbstractVector{<:AbstractFloat})
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    output = similar(prob)
    _predict!(m, output, input, prob)
    return output
end

function predict(m::MultiRegProbModel, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(_nreg(m)) regressors, but $(length(input)) were provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    output = similar(prob)
    _predict!(m, output, input, prob)
    return output
end

function predict(m::UniRegProbModel, input::AbstractVector{<:Number}, prob::Union{AbstractFloat, AbstractVector{<:AbstractFloat}})
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    predict(m, input[1], prob)
end

function predict(m::MultiRegProbModel, input::Number, prob::Union{AbstractVector{<:AbstractFloat}, AbstractFloat})
    predict(m, [input], prob)
end

"""
    predict!(m, output, input[, prob])
In-place version of `predict` that stores the results in `output` vector.
"""
function predict!(m::UniRegProbModel, output::AbstractVector{<:AbstractFloat}, input::Number, prob::AbstractVector{<:AbstractFloat})
    length(prob) == length(output) || throw(ArgumentError("length of `prob` does not match the length of `output`"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict!(m, output, input, prob)
end

function predict!(m::MultiRegProbModel, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})
    length(prob) == length(output) || throw(ArgumentError("length of `prob` does not match the length of `output`"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict!(m, output, input, prob)
end

function predict!(m::UniRegProbModel, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    predict!(m, output, input[1], prob)
end

function predict!(m::MultiRegProbModel, output::AbstractVector{<:AbstractFloat}, input::Number, prob::AbstractVector{<:AbstractFloat})
    predict!(m, output, [input], prob)
end