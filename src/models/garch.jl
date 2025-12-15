"""
    GARCH([type::Type{F}=Float64,] n::Integer[; filter::Bool=false]) where {F<:AbstractFloat}
Creates a `GARCH{F}<:UniPostModel{F}<:PostModel{F}` model for Generalized Autoregressive Conditional Heteroskedasticity model with filtered empirical distribution, trained on `n` observations.
"""
struct GARCH{F<:AbstractFloat} <: UniPostModel{F}
    errors::Vector{F}
    scores::Vector{F}
    scale::Base.RefValue{F}
    params::Vector{F}
    σ::Base.RefValue{F}
    optimizer::Opt
    filter::Bool
    function GARCH(::Type{F}, n::Integer; filter::Bool=false) where {F<:AbstractFloat} 
        optimizer = NLopt.Opt(:LD_MMA, 2)
        NLopt.lower_bounds!(optimizer, [0.0, 0.0])
        NLopt.upper_bounds!(optimizer, [1.0, 1.0])
        NLopt.xtol_rel!(optimizer, 1e-8)
        NLopt.nlopt_set_maxeval(optimizer, 100_000)
        function variance_targeting(x::Vector, grad::Vector)
            if length(grad) > 0
                grad[1] = 1.0
                grad[2] = 1.0
            end
            return x[1] + x[2] - 1.0
        end
        NLopt.inequality_constraint!(optimizer, (x, g) -> variance_targeting(x, g), 1e-8)
        new{F}(
            Vector{F}(undef, n),
            Vector{F}(undef, n),
            Ref{F}(1.0),
            Vector{F}(undef, 2),
            Ref{F}(1.0),
            optimizer,
            filter
        )
    end
end

getmodel(::Type{F}, ::Val{:garch}, params::Vararg) where {F<:AbstractFloat} = GARCH(F, params[1])

getmodel(::Type{F}, ::Val{:hsgarch}, params::Vararg) where {F<:AbstractFloat} = GARCH(F, params[1], filter=true)

matchwindow(m::GARCH, window::Integer) = length(m.errors) == window

function _objective(params::Vector, errors::Vector{<:AbstractFloat})
    α, β = params
    ω = 1.0 - α - β
    loss = 0.0
    variance = 1.0
    for i in eachindex(errors)
        squared_error = abs2(errors[i])
        loss += squared_error/variance + log(variance)
        variance = ω + α*variance + β*squared_error
    end
    return loss/length(errors)
end

function _autodiff(f::Function)
    function nlopt_fn(x::Vector, grad::Vector)
        if length(grad) > 0
            ForwardDiff.gradient!(grad, f, x)
        end
        return f(x)
    end
end

function _filter!(m::GARCH)
    α, β = m.params
    ω = 1.0 - α - β
    variance = 1.0
    for i in eachindex(m.errors)
        m.scores[i] = m.errors[i]/sqrt(variance)
        squared_error = abs2(m.errors[i])
        variance = ω + α*variance + β*squared_error
    end
    sort!(m.scores)
    m.σ[] = sqrt(variance)*m.scale[]
    return nothing
end

function _train(m::GARCH, X::AbstractVecOrMat{<:Number}, Y::AbstractVector{<:Number})::Nothing
    m.params[1] = 0.7
    m.params[2] = 0.15
    for i in eachindex(m.errors)
        m.errors[i] = Y[i] - X[i]
    end
    m.scale[] = sqrt(sum(abs2, m.errors)/length(m.errors))
    m.errors .= m.errors./m.scale[]
    f(u) = _objective(u, m.errors)
    NLopt.min_objective!(m.optimizer, _autodiff(f))
    NLopt.optimize!(m.optimizer, m.params)
    _filter!(m)
    return nothing
end

function _predict(m::GARCH{F}, input::Number, prob::AbstractFloat) where {F<:AbstractFloat}
    if m.filter
        return input + m.σ[]*quantile(m.scores, prob, sorted=true, alpha=1, beta=1)
    else
        return input + m.σ[]*(sqrt(2)*erfinv(2*prob - 1))
    end
end

function _predict(m::GARCH{F}, input::Number, prob::AbstractVector{<:AbstractFloat}) where {F<:AbstractFloat}
    output = Vector{F}(undef, length(prob))
    for j in eachindex(output)
        output[j] = _predict(m, input, prob[j])
    end
    return output
end

function _predict!(m::GARCH, output::AbstractVector{<:AbstractFloat}, input::Number, prob::AbstractVector{<:AbstractFloat})::Nothing
    for j in eachindex(output)
        output[j] = _predict(m, input, prob[j])
    end
    return nothing
end

function _predict!(m::GARCH, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})::Nothing
    for j in eachindex(output)
        output[j] = _predict(m, input[begin], prob[j])
    end
    return nothing
end
