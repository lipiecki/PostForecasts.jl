"""
    GARCH([type::Type{F}=Float64,] n::Integer; kwargs...) where {F<:AbstractFloat}
Creates a `GARCH{F}<:UniPostModel{F}<:PostModel{F}` model for Generalized Autoregressive Conditional Heteroskedasticity model with filtered empirical distribution, trained on `n` observations.

## Optional keyword arguments
- `filter::Bool` specifies whether to use filtered empirical distribution of standardized residuals for prediction. The default value is `false`.
- `abs::Bool` specifies whether to use absolute values of standardized residuals for prediction. The default value is `false`. Relevant only if `filter` is set to `true`.
- `abstol::Float64` specifies the absolute tolerance for the optimization algorithm. The default value is set by the hyperparameter `:abstol`.
- `reltol::Float64` specifies the relative tolerance for the optimization algorithm. The default value is set by the hyperparameter `:reltol`.
- `maxeval::Int` specifies the maximum number of evaluations for the optimization algorithm. The default value is set by the hyperparameter `:maxeval`.
- `nloptalg::Symbol` specifies the optimization algorithm to be used. The default value is set by the hyperparameter `:garch_solver`.
"""
struct GARCH{F<:AbstractFloat} <: UniPostModel{F}
    errors::Vector{F}
    scores::Vector{F}
    scale::Base.RefValue{F}
    params::Vector{F}
    σ::Base.RefValue{F}
    opt::Opt
    filter::Bool
    abs::Bool
    function GARCH(::Type{F}, n::Integer; 
            filter::Bool=false, abs::Bool=false, 
            abstol::Float64=get_hyperparam(:abstol), reltol::Float64=get_hyperparam(:reltol), 
            maxeval::Int=get_hyperparam(:maxeval), nloptalg::Symbol=get_hyperparam(:garch_solver)) where {F<:AbstractFloat}
        opt = NLopt.Opt(nloptalg, 2)
        NLopt.lower_bounds!(opt, zeros(2) .+ abstol)
        NLopt.upper_bounds!(opt, ones(2) .- abstol)
        NLopt.xtol_abs!(opt, abstol)
        NLopt.xtol_rel!(opt, reltol)
        NLopt.nlopt_set_maxeval(opt, maxeval)
        NLopt.inequality_constraint!(opt, (x, g) -> _variance_targeting(x, g) + abstol)
        new{F}(
            Vector{F}(undef, n),
            Vector{F}(undef, n),
            Ref{F}(1.0),
            Vector{F}(undef, 2),
            Ref{F}(1.0),
            opt,
            filter,
            abs
        )
    end
end

GARCH(n::Integer; kwargs...) = GARCH(Float64, n; kwargs...)

getmodel(::Type{F}, ::Val{:garch}, params::Vararg) where {F<:AbstractFloat} = GARCH(F, params[1])

getmodel(::Type{F}, ::Val{:fhs}, params::Vararg) where {F<:AbstractFloat} = GARCH(F, params[1], filter=true)

getmodel(::Type{F}, ::Val{:sfhs}, params::Vararg) where {F<:AbstractFloat} = GARCH(F, params[1], filter=true, abs=true)

matchwindow(m::GARCH, window::Integer) = length(m.errors) == window

"""
    getparams(m::GARCH)
Returns the parameters of the GARCH model `m` as a tuple `(α, β, ω)`, where ``\\sigma_{t}^{2} = \\alpha \\sigma_{t-1}^{2} + \\beta \\epsilon_{t-1}^{2} + \\omega``.
"""
function getparams(m::GARCH)
    return (m.params[1], m.params[2], abs2(m.scale[])*(1.0 - m.params[1] - m.params[2]))
end

function advance!(m::GARCH, ::Vararg{Union{Number, AbstractVector{<:Number}}})::Nothing
    variance = abs2(m.σ[])*(m.params[1] + m.params[2])
    variance += 1.0 - m.params[1] - m.params[2]
    m.σ[] = sqrt(variance)
    return nothing
end

function _objective_garch(params::Vector, errors::Vector{<:AbstractFloat})
    loss = 0.0
    variance = 1.0
    epsilon = eps(eltype(params))
    for i in eachindex(errors)
        squared_error = abs2(errors[i]) + epsilon
        loss += squared_error/variance + log(variance)
        variance *= params[1]
        variance += params[2]*squared_error
        variance += 1.0 - params[1] - params[2]
    end
    return loss/length(errors)
end

function _forward_pass!(m::GARCH)
    variance = 1.0
    epsilon = eps(eltype(m.params))
    for i in eachindex(m.errors)
        m.scores[i] = m.errors[i]/sqrt(variance)
        squared_error = abs2(m.errors[i]) + epsilon
        variance *= m.params[1]
        variance += m.params[2]*squared_error
        variance += 1.0 - m.params[1] - m.params[2]
    end
    if m.abs
        m.scores .= abs.(m.scores)
    end
    sort!(m.scores)
    m.σ[] = sqrt(variance)
    return nothing
end

function _train(m::GARCH, X::AbstractVecOrMat{<:Number}, Y::AbstractVector{<:Number})::Nothing
    m.params[1] = 0.8
    m.params[2] = 0.1
    for i in eachindex(m.errors)
        m.errors[i] = Y[i] - X[i]
    end
    m.scale[] = sqrt(sum(abs2, m.errors)/length(m.errors))
    if m.scale[] ≈ 0.0
        m.scale[] = 0.0
        m.σ[] = 0.0
        return nothing
    end
    m.errors .= m.errors./m.scale[]
    f(u) = _objective_garch(u, m.errors)
    NLopt.min_objective!(m.opt, _autodiff(f))
    _, _, ret = NLopt.optimize!(m.opt, m.params)
    _nlopt_check_success(ret, m.params)
    _forward_pass!(m)
    return nothing
end

function _predict(m::GARCH{F}, input::Number, prob::AbstractFloat) where {F<:AbstractFloat}
    σ = m.σ[]*m.scale[]
    if m.filter 
        if m.abs
            sgn::F = prob ≈ 0.5 ? 0.0 : (prob < 0.5 ? -1.0 : 1.0)
			output = input + σ*sgn*quantile(m.scores, (2prob - 1)sgn, sorted=true, alpha=1, beta=1)
        else
			output = input + σ*quantile(m.scores, prob, sorted=true, alpha=1, beta=1)
        end
    else
        output = input + σ*(sqrt(2)*erfinv(2*prob - 1))
    end
    return output
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
