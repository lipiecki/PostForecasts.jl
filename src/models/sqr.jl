"""
    SQR([type::Type{F}=Float64,] n::Integer, r::Integer, prob::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}) where {F<:AbstractFloat}
Creates a `SQR{F}<:MultiPostModel{F}<:PostModel{F}` model for smoothing quantile regression to be trained on `n` observations with `r` forecasts (regressors), fitting quantiles at probabilities specified by `prob`.
"""
struct SQR{F<:AbstractFloat} <: MultiPostModel{F}
    prob::Vector{F} # vector of probabilities for which quantile regressions are fitted
    W::Matrix{F} # weights of quantile regressions

    # z-score parameters
    zmean::Vector{F}
    zstd::Vector{F}

    residuals::Vector{F}

    # variables for constructing a linear programming problem
    solutions::Vector{F}
    h::Vector{F}
    H::Matrix{F}
    bounds::Vector{F}
    lpmodel::GenericModel{F}

    # variables for nonlinear optimization of the smoothing step
    params::Vector{F}
    optimizer::Opt
    tol::Float64
    
    function SQR(::Type{F}, n::Integer, r::Integer, prob::AbstractVector{<:AbstractFloat}, tol::Float64=get_hyperparam(:tol), maxeval::Int=get_hyperparam(:maxeval), nloptalg::Symbol=get_hyperparam(:nloptalg)) where {F<:AbstractFloat}
        issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
        (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
        lpmodel = GenericModel{F}(HiGHS.Optimizer, add_bridges=false)
        _config_solver_threads(lpmodel)
        set_silent(lpmodel)
        set_string_names_on_creation(lpmodel, false)
        
        optimizer = NLopt.Opt(nloptalg, r + 1)
        NLopt.xtol_abs!(optimizer, tol)
        NLopt.nlopt_set_maxeval(optimizer, maxeval)

        new{F}(convert(Vector{F}, prob), 
            Matrix{F}(undef, r + 1, length(prob)), 
            Vector{F}(undef, r + 1),
            Vector{F}(undef, r + 1),
            Vector{F}(undef, n),
            Vector{F}(undef, r + 1 + 2n),
            Vector{F}(undef, r + 1 + 2n),
            Matrix{F}(undef, n, r + 1 + 2n),
            convert(Vector{F}, [-Inf.*ones(r + 1); zeros(2n)]),
            lpmodel,
            Vector{F}(undef, r + 1),
            optimizer,
            tol)
    end

    SQR(::Type{F}, n::Integer, r::Integer, prob::AbstractFloat) where {F<:AbstractFloat} = SQR(F, n, r, [prob])

    SQR(n::Integer, r::Integer, prob::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}) = SQR(Float64, n, r, prob)
end

"""
    iSQR(args...)
Creates an isotonic smoothing quantile regression model (see [Lipiecki & Uniejewski (2025)](https://arxiv.org/abs/2507.15079)), constraining the weights to be non-negative. The arguments `args...` are the same as for `SQR`.
"""
function iSQR(args...)
    isqr = SQR(args...)
    isqr.bounds[1:nreg(isqr)] .= 0.0
    NLopt.lower_bounds!(isqr.optimizer, [zeros(nreg(isqr)); -Inf])
    return isqr
end

getmodel(::Type{F}, ::Val{:sqr}, params::Vararg) where {F<:AbstractFloat} = SQR(F, params[1], params[2], params[3])

getmodel(::Type{F}, ::Val{:isqr}, params::Vararg) where {F<:AbstractFloat} = iSQR(F, params[1], params[2], params[3])

matchwindow(m::SQR, window::Integer) = size(m.H, 1) == window

function getweights(m::SQR)
    return copy(m.W)
end

function getquantprob(m::SQR)
    return copy(m.prob)
end

function nreg(m::SQR)
    return size(m.W, 1) - 1 # -1 to discount the intercept
end

function _objective_sqr(params::Vector, level::Number, bandwidth::Number, X::AbstractVecOrMat{<:Number}, Y::AbstractVector{<:Number})
    loss = 0.0
    for i in eachindex(Y)
        @views residual = Y[i] - dot(X[i, :], params[1:end-1]) - params[end]
        loss += bandwidth*normal_pdf(-residual/bandwidth) + residual*(level - normal_cdf(-residual/bandwidth))
    end
    return loss/length(Y)
end

function _train(m::SQR, X::AbstractVecOrMat{<:Number}, Y::AbstractVector{<:Number})::Nothing
    H, h = m.H, m.h
    n, d = ndims(X) > 1 ? size(X) : (length(X), 1)
    for i in 1:d
        m.zmean[i] = mean(@views(X[:, i]))
        m.zstd[i] = sqrt(sum(abs2, @views(X[:, i]) .- m.zmean[i])/(n-1))
    end
    d += 1 # for the intercept
    m.zmean[end] = mean(Y)
    m.zstd[end] = sqrt(sum(abs2, Y .- m.zmean[end])/(n-1))
    targets = (Y .- m.zmean[end]) ./ m.zstd[end]
    fill!(H, 0.0)
    fill!(h, 0.0)
    empty!(m.lpmodel)
    for i in 1:n
        H[i, d] = 1.0
        H[i, d+i] = 1.0
        H[i, d+n+i] = -1.0
        for j in 1:d-1
            z = (X[i, j] - m.zmean[j]) / m.zstd[j]
            H[i, j] = z
        end
    end
    @variable(m.lpmodel, x[i=axes(H, 2)] >= m.bounds[i])
    @constraint(m.lpmodel, [j in 1:n], sum(H[j, i]*x[i] for i in axes(H, 2)) == targets[j])
    for (p, α) in enumerate(m.prob)
        # standard quantile regression
        h[d+1:d+n] .= α
        h[d+n+1:d+2n] .= 1.0 - α
        if p > 1
            foreach(i -> set_start_value(x[i], m.solutions[i]), eachindex(m.solutions))
        end
        @objective(m.lpmodel, Min, sum(h.*x))
        JuMP.optimize!(m.lpmodel)
        m.solutions .= JuMP.value(x)
        for i in 1:d
            m.W[i, p] = m.solutions[i]
        end
        
        # smoothing step
        m.residuals .= targets
        @views foreach(i -> m.residuals[i] -= dot(H[i, 1:d-1], m.W[1:d-1, p]), 1:n)
        m.residuals .-= m.W[end, p]
        sigma_res = sqrt(sum(abs2, m.residuals .- mean(m.residuals))/(n-1))
        sigma_res = min(
            sigma_res, 
            (quantile(m.residuals, 0.75) - quantile(m.residuals, 0.25))/1.34898
        )
        bandwidth = 0.9*sigma_res*n^(-1/5)
        f(u) = _objective_sqr(u, m.prob[p], bandwidth, @views(H[:, 1:d-1]), targets)
        m.params .= m.W[:, p]
        m.params[1:end-1] .+= 2m.tol
        NLopt.min_objective!(m.optimizer, _autodiff(f))
        NLopt.optimize!(m.optimizer, m.params)
        m.W[:, p] .= m.params
    end
    return nothing
end

function _predict(m::SQR{F}, input::AbstractVector{<:Number}, prob::AbstractFloat) where {F<:AbstractFloat}
    j = findfirst(p -> p ≈ prob, m.prob)
    isnothing(j) && throw(ArgumentError("cannot match the model quantile to the provided probability ($(prob))"))
    return (m.W[end, j] + dot(@view(m.W[1:end-1, j]), (input .- @view(m.zmean[1:end-1]))./@view(m.zstd[1:end-1]))) * m.zstd[end] + m.zmean[end]
end

function _predict(m::SQR{F}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat}) where {F<:AbstractFloat}
    output = Vector{F}(undef, length(prob))
    for j in eachindex(output)
        output[j] = _predict(m, input, prob[j])
    end
    sort!(output)
    return output
end

function _predict(m::SQR{F}, input::AbstractVector{<:Number}) where {F<:AbstractFloat}
    output = Vector{F}(undef, length(m.prob))
    _predict!(m, output, input)
    return output
end

function _predict!(m::SQR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number})::Nothing
    for j in eachindex(output)
        output[j] = (m.W[end, j] + dot(@view(m.W[1:end-1, j]), (input .- @view(m.zmean[1:end-1]))./@view(m.zstd[1:end-1]))) * m.zstd[end] + m.zmean[end]
    end
    sort!(output)
    return nothing
end

_predict!(m::SQR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, ::AbstractVector{<:AbstractFloat}) = _predict!(m, output, input)
