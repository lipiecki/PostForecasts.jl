"""
    QR([type::Type{F}=Float64,] n::Integer, r::Integer, prob::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}) where {F<:AbstractFloat}
Creates a `QR{F}<:MultiPostModel{F}<:PostModel{F}` model for quantile regression to be trained on `n` observations with `r` forecasts (regressors), fitting quantiles at probabilities specified by `prob`.
"""
struct QR{F<:AbstractFloat} <: MultiPostModel{F}
    prob::Vector{F} # vector of probabilities for which quantile regressions are fitted
    W::Matrix{F} # weights of quantile regressions

    # variables for constructing a linear programming problem
    h::Vector{F}
    H::Matrix{F}
    bounds::Vector{F}
    lpmodel::GenericModel{F}

    function QR(::Type{F}, n::Integer, r::Integer, prob::AbstractVector{<:AbstractFloat}) where {F<:AbstractFloat}
        issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
        (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
        lpmodel = GenericModel{F}(HiGHS.Optimizer, add_bridges=false)
        set_silent(lpmodel)
        set_string_names_on_creation(lpmodel, false)
        new{F}(convert(Vector{F}, prob), 
            Matrix{F}(undef, r + 1, length(prob)), 
            Vector{F}(undef, r + 1 + 2n),
            Matrix{F}(undef, n, r + 1 + 2n),
            convert(Vector{F}, [-Inf.*ones(r + 1); zeros(2n)]),
            lpmodel)
    end

    QR(::Type{F}, n::Integer, r::Integer, prob::AbstractFloat) where {F<:AbstractFloat} = QR(F, n, r, [prob])

    QR(n::Integer, r::Integer, prob::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}) = QR(Float64, n, r, prob)
end

"""
    iQR(args...)
    Create an isotonic quantile regression model, constraining the weights to be non-negative. The arguments `args...` are the same as for `QR`.
"""
function iQR(args...)
    iqr = QR(args...)
    iqr.bounds[1:nreg(iqr)] .= 0.0
    return iqr
end

getmodel(::Type{F}, ::Val{:qr}, params::Vararg) where {F<:AbstractFloat} = QR(F, params[1], params[2], params[3])

getmodel(::Type{F}, ::Val{:iqr}, params::Vararg) where {F<:AbstractFloat} = iQR(F, params[1], params[2], params[3])

matchwindow(m::QR, window::Integer) = size(m.H, 1) == window

"""
    getweights(m::QR)
Return a copy of the weight matrix from model `m`.
"""
function getweights(m::QR)
    return copy(m.W)
end

"""
   getquantprob(m::QR)
Return a copy of the vector of probabilities corresponding to the quantiles from model `m`.
"""
function getquantprob(m::QR)
    return copy(m.prob)
end

function nreg(m::QR)
    return size(m.W, 1) - 1 # -1 to discount the intercept
end

function _train(m::QR, X::AbstractVecOrMat{<:Number}, Y::AbstractVector{<:Number})::Nothing
    H, h = m.H, m.h
    n, d = ndims(X) > 1 ? size(X) : (length(X), 1)
    d += 1 # for the intercept
    for (p, α) in enumerate(m.prob)
        empty!(m.lpmodel)  
        fill!(H, 0.0)
        fill!(h, 0.0)
        
        for i in 1:n
            H[i, d] = 1.0
            H[i, d+i] = 1.0
            H[i, d+n+i] = -1.0
            for j in 1:d-1
                H[i, j] = X[i, j]
            end
        end
        
        h[d+1:d+n] .= α
        h[d+n+1:d+2n] .= 1.0 - α
    
        @variable(m.lpmodel, x[i=axes(H, 2)] >= m.bounds[i])
        @objective(m.lpmodel, Min, sum(h.*x))
        @constraint(m.lpmodel, H*x == Y)
        JuMP.optimize!(m.lpmodel)
        for i in 1:d
            m.W[i, p] = JuMP.value(x[i])
        end
    end
    return nothing
end

function _predict(m::QR{F}, input::AbstractVector{<:Number}, prob::AbstractFloat) where {F<:AbstractFloat}
    j = findfirst(p -> p ≈ prob, m.prob)
    isnothing(j) && throw(ArgumentError("cannot match the model quantile to the provided probability ($(prob))"))
    return m.W[end, j] + dot(@view(m.W[1:end-1, j]), input)
end

function _predict(m::QR{F}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat}) where {F<:AbstractFloat}
    output = Vector{F}(undef, length(prob))
    for j in eachindex(output)
        output[j] = _predict(m, input, prob[j])
    end
    return output
end

function _predict(m::QR{F}, input::AbstractVector{<:Number}) where {F<:AbstractFloat}
    output = Vector{F}(undef, length(m.prob))
    _predict!(m, output, input)
    return output
end

function _predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number})::Nothing
    for j in eachindex(output)
        output[j] = m.W[end, j] + dot(@view(m.W[1:end-1, j]), input)
    end
    sort!(output)
    return nothing
end

_predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, ::AbstractVector{<:AbstractFloat}) = _predict!(m, output, input)
