"""
    LassoQR([type::Type{F}=Float64,] n::Integer, r::Integer, prob::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}; kwargs...) where {F<:AbstractFloat}
Creates a `LassoQR{F}<:MultiPostModel{F}<:PostModel{F}` model for lasso-estimated quantile regression to be trained on `n` observations with `r` forecasts (regressors), fitting quantiles at probabilities specified by `prob`.
## Optional keyword arguments 
- `nlambdas::Integer` specifies the number of regularization strength values to be considered in the regularization path. The default value is set by the hyperparameter `:nlambdas`.
- `minlambda::AbstractFloat` specifies the relative minimum regularization strength. The default value is set by the hyperparameter `:minlambda`.
"""
struct LassoQR{F<:AbstractFloat} <: MultiPostModel{F}
    prob::Vector{F} # vector of probabilities for which quantile regressions are fitted
    W::Matrix{F} # weights of quantile regressions

    # z-score parameters
    zmean::Vector{F}
    zstd::Vector{F}

    # variables for constructing a linear programming problem
    solutions::Vector{F}
    h::Vector{F}
    H::Matrix{F}
    lambda_path::Vector{F}
    minlambda::Base.RefValue{F}
    optimal_lambda::Vector{F}
    lpmodel::GenericModel{F}

    function LassoQR(::Type{F}, n::Integer, r::Integer, prob::AbstractVector{<:AbstractFloat};
        nlambdas::Integer=get_hyperparam(:nlambdas), minlambda::AbstractFloat=get_hyperparam(:minlambda)) where {F<:AbstractFloat}
        issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
        (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
        lpmodel = GenericModel{F}(HiGHS.Optimizer, add_bridges=false)
        _config_solver_threads(lpmodel)
        set_silent(lpmodel)
        set_string_names_on_creation(lpmodel, false)
        new{F}(convert(Vector{F}, prob), 
            Matrix{F}(undef, r + 1, length(prob)),
            Vector{F}(undef, r + 1),
            Vector{F}(undef, r + 1),
            Vector{F}(undef, 2(r + 1 + n)),
            Vector{F}(undef, 2(r + 1 + n)),
            Matrix{F}(undef, n, 2(r + 1 + n)),
            Vector{F}(undef, nlambdas),
            Base.RefValue{F}(minlambda),
            Vector{F}(undef, length(prob)),
            lpmodel)
    end
    LassoQR(::Type{F}, n::Integer, r::Integer, prob::AbstractFloat; kwargs...) where {F<:AbstractFloat} = LassoQR(F, n, r, [prob]; kwargs...)
    LassoQR(n::Integer, r::Integer, prob::Union{AbstractFloat, Vector{<:AbstractFloat}}; kwargs...) = LassoQR(Float64, n, r, prob; kwargs...)
end

getmodel(::Type{F}, ::Val{:lassoqr}, params::Vararg) where {F<:AbstractFloat} = LassoQR(F, params[1], params[2], params[3])

matchwindow(m::LassoQR, window::Integer) = size(m.H, 1) == window

function getweights(m::LassoQR)
    return copy(m.W)
end

function getlambdas(m::LassoQR)
    return copy(m.optimal_lambda)
end

function getquantprob(m::LassoQR)
    return copy(m.prob)
end

function nreg(m::LassoQR)
    return size(m.W, 1) - 1 # -1 to discount the intercept
end

function _train(m::LassoQR{F}, X::AbstractVecOrMat{<:Number}, Y::AbstractVector{<:Number})::Nothing where {F<:AbstractFloat}
    H, h = m.H, m.h
    n, d = ndims(X) > 1 ? size(X) : (length(X), 1)
    for i in 1:d
        m.zmean[i] = mean(@views(X[:, i]))
        m.zstd[i] = sqrt(sum(abs2, @views(X[:, i]) .- m.zmean[i])/(n-1))
    end
    m.zmean[end] = mean(Y)
    m.zstd[end] = sqrt(sum(abs2, Y .- m.zmean[end])/(n-1))

    d += 1 # for the intercept
    fill!(H, 0.0)
    fill!(h, 0.0)# 
    empty!(m.lpmodel)
    for i in 1:n
        H[i, d] = 1.0
        H[i, 2d] = -1.0
        H[i, 2d+i] = 1.0
        H[i, 2d+n+i] = -1.0
        for j in 1:d-1
            z = (X[i, j] - m.zmean[j]) / m.zstd[j]
            H[i, j] = z
            H[i, d+j] = -z
        end
    end
    @variable(m.lpmodel, x[axes(H, 2)] >= 0)
    @constraint(m.lpmodel, [j in 1:n], sum(H[j, i]*x[i] for i in axes(H, 2)) == (Y[j]-m.zmean[end])/m.zstd[end])
    for (p, α) in enumerate(m.prob)
        h[2d+1:2d+n] .= α
        h[2d+n+1:2d+2n] .= 1.0 - α
        bic = Inf
        
        # construct a regularization path
        maxlambda = -Inf
        sample_quantile = quantile(Y, α, alpha=1, beta=1)
        @views for j in 1:d-1
            maxlambda = max(maxlambda, abs(sum(H[i, j]*(α - (Y[i] <= sample_quantile)) for i in 1:n)))
        end

        m.solutions .= 0.0
        q = (sample_quantile - m.zmean[end])/m.zstd[end]
        if q > 0
            m.solutions[d] = q
        else
            m.solutions[2d] = -q
        end
    
        if length(m.lambda_path) == 1
            m.lambda_path[1] = maxlambda*m.minlambda[]
        else
            m.lambda_path .= exp.(range(log(maxlambda), log(m.minlambda[]*maxlambda), length=length(m.lambda_path)))
        end

        for λ in m.lambda_path
            h[1:d-1] .= λ
            h[d+1:2d-1] .= λ
            if p > 1
                foreach(i -> set_start_value(x[i], m.solutions[i]), eachindex(m.solutions))
            end
            @objective(m.lpmodel, Min, sum(h.*x)) 
            JuMP.optimize!(m.lpmodel)
            current_bic = log(abs(sum(JuMP.value(x[i])*h[i] for i in 2d+1:2d+2n))) + log(d)*(sum(abs(JuMP.value(x[i]-x[d+i])) > eps(zero(F)) for i in 1:d-1)+1)*log(n)/(2n)
            if current_bic < bic
                bic = current_bic
                m.optimal_lambda[p] = λ
                m.solutions .= JuMP.value(x)
                for i in 1:d
                    m.W[i, p] = m.solutions[i] - m.solutions[d+i]
                end
            end
        end
    end
    return nothing
end

function _predict(m::LassoQR{F}, input::AbstractVector{<:Number}, prob::AbstractFloat) where {F<:AbstractFloat}
    j = findfirst(p -> p ≈ prob, m.prob)
    isnothing(j) && throw(ArgumentError("cannot match the model quantile to the provided probability ($(prob))"))
    return (m.W[end, j] + dot(@view(m.W[1:end-1, j]), (input .- @view(m.zmean[1:end-1]))./@view(m.zstd[1:end-1]))) * m.zstd[end] + m.zmean[end]
end

function _predict(m::LassoQR{F}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat}) where {F<:AbstractFloat}
    output = Vector{F}(undef, length(prob))
    for j in eachindex(output)
        output[j] = _predict(m, input, prob[j])
    end
    sort!(output)
    return output
end

function _predict(m::LassoQR{F}, input::AbstractVector{<:Number}) where {F<:AbstractFloat}
    output = Vector{F}(undef, length(m.prob))
    _predict!(m, output, input)
    return output
end

function _predict!(m::LassoQR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number})::Nothing
    for j in eachindex(output)
        output[j] = (m.W[end, j] + dot(@view(m.W[1:end-1, j]), (input .- @view(m.zmean[1:end-1]))./@view(m.zstd[1:end-1]))) * m.zstd[end] + m.zmean[end]
    end
    sort!(output)
    return nothing
end

_predict!(m::LassoQR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, ::AbstractVector{<:AbstractFloat}) = _predict!(m, output, input)
