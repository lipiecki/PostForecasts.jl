"""
    predict(m, input, prob)
Predict quantiles at specified `prob::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}` using model `m::ProbModel` conditional on `input::Union{Number, AbstractVector{<:Number}}`.

**Note:** for `m::QR` ...
"""
function predict(m::ProbModel, input::Number, prob::AbstractFloat)::AbstractFloat
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
    _predict(m, input, prob)
end

function predict(m::ProbModel, input::Number, prob::AbstractVector{<:AbstractFloat})::AbstractVector{<:AbstractFloat}
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    output = Vector{Float64}(undef, length(prob))
    _predict!(m, output, input, prob)
    return output
end

function predict(m::UniRegProbModel, input::AbstractVector{<:Number}, prob::AbstractFloat)::AbstractFloat
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
    _predict(m, input[1], prob)
end

function predict(m::MultiRegProbModel, input::AbstractVector{<:Number}, prob::AbstractFloat)::AbstractFloat
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
    _predict(m, input, prob)
end

function predict(m::UniRegProbModel, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})::AbstractVector{<:AbstractFloat}
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    output = Vector{Float64}(undef, length(prob))
    _predict!(m, output, input[1], prob)
    return output
end

function predict(m::MultiRegProbModel, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})::AbstractVector{<:AbstractFloat}
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    output = Vector{Float64}(undef, length(prob))
    _predict!(m, output, input, prob)
    return output
end

function predict(m::QR, input::AbstractVector{<:Number})::AbstractVector{<:AbstractFloat}
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    output = Vector{Float64}(undef, nquantiles(m))
    _predict!(m, output, input, prob)
    return output
end

function predict(m::QR, input::Number)::AbstractVector{<:AbstractFloat}
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    output = Vector{Float64}(undef, nquantiles(m))
    _predict!(m, output, input)
    return output
end

"""
    predict!(m, output, input[, prob])
In-place version of `predict` that stores the results in `output` vector.
"""
function predict!(m::ProbModel, output::AbstractVector{<:AbstractFloat}, input::Number, prob::AbstractVector{<:AbstractFloat})
    length(output) == length(prob) || throw(ArgumentError("length of `prob` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict!(m, output, input, prob)
end

function predict!(m::UniRegProbModel, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})
    length(output) == length(prob) || throw(ArgumentError("length of `prob` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    length(input) == 1 || throw(ArgumentError("model `m` requires a single regressor, but $(length(input)) were provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict!(m, output, input[1], prob)
end

function predict!(m::MultiRegProbModel, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})
    length(output) == length(prob) || throw(ArgumentError("length of `prob` ($(length(output))) does not match the length of `output` ($(length(output)))"))
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
    (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
    _predict!(m, output, input, prob)
end

function predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number})
    nreg(m) == length(input) || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but $(length(input)) were provided"))
    _predict!(m, output, input)
end

function predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::Number)
    nreg(m) == 1 || throw(ArgumentError("model `m` requires $(nreg(m)) regressors, but one was provided"))
    _predict!(m, output, input)
end
