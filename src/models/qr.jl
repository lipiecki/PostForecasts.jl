"""
    QR([type::Type{F}=Float64,] n::Integer, r::Integer, prob::Union{AbstractFloat, Vector{<:AbstractFloat}}) where {F<:AbstractFloat}
Creates a `QR{F}<:MultiPostModel{F}<:PostModel{F}` model for quantile regression to be trained on `n` observations with `r` forecasts (regressors), fitting quantiles at probabilities specified by `prob`.
"""
struct QR{F<:AbstractFloat} <: MultiPostModel{F}
    prob::Vector{F} # vector of probabilities for which quantile regressions are fitted
    W::Matrix{F} # weights of quantile regressions

    # variables for constructing a linear programming problem
    h::Vector{F}
    H::Matrix{F}
    lpmodel::GenericModel{F}

    function QR(::Type{F}, n::Integer, r::Integer, prob::Vector{<:AbstractFloat}) where {F<:AbstractFloat}
        issorted(prob) || throw(ArgumentError("`prob` vector has to be sorted"))
        (prob[begin] > 0.0 && prob[end] < 1.0) || throw(ArgumentError("elements of `prob` must belong to an open (0, 1) interval"))
        lpmodel = GenericModel{F}(HiGHS.Optimizer, add_bridges=false)
        set_silent(lpmodel)
        set_string_names_on_creation(lpmodel, false)
        new{F}(convert(Vector{F}, prob), 
            Matrix{F}(undef, r + 1, length(prob)), 
            Vector{F}(undef, 2(r + 1 + n)),
            Matrix{F}(undef, n, 2(r + 1 + n)),
            lpmodel)
    end

    function QR(::Type{F}, n::Integer, r::Integer, prob::AbstractFloat) where {F<:AbstractFloat}
        (prob > 0.0 && prob < 1.0) || throw(ArgumentError("`prob` must belong to an open (0, 1) interval"))
        lpmodel = GenericModel{F}(HiGHS.Optimizer, add_bridges=false)
        set_silent(lpmodel)
        set_string_names_on_creation(lpmodel, false)
        new{F}([convert(F, prob)], 
            Matrix{F}(undef, r + 1, length(prob)), 
            Vector{F}(undef, 2(r + 1 + n)),
            Matrix{F}(undef, n, 2(r + 1 + n)),
            lpmodel)
    end
end

QR(n::Integer, r::Integer, prob::Union{AbstractFloat, Vector{<:AbstractFloat}}) = QR(Float64, n, r, prob)

getmodel(::Type{F}, ::Val{:qr}, params::Vararg) where {F<:AbstractFloat} = QR(F, params[1], params[2], params[3])

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
    for (p, α) in enumerate(m.prob)
        empty!(m.lpmodel)  
        n, d = ndims(X) > 1 ? size(X) : (length(X), 1)
        d += 1 # for the intercept

        fill!(H, 0.0)
        for i in 1:n
            H[i, d] = 1.0
            H[i, 2d] = -1.0
            H[i, 2d+i] = 1.0
            H[i, 2d+n+i] = -1.0
            for j in 1:d-1
                H[i, j] = X[i, j]
                H[i, d+j] = -X[i, j]
            end
        end
        
        h[1:2d] .= 0.0
        h[2d+1:2d+n] .= α
        h[2d+n+1:2d+2n] .= 1.0 - α
    
        @variable(m.lpmodel, x[axes(H, 2)] >= 0)
        @objective(m.lpmodel, Min, sum(h[i]*x[i] for i in axes(H, 2)))
        @constraint(m.lpmodel, [j in 1:n], sum(H[j, i]*x[i] for i in axes(H, 2)) == Y[j])
        JuMP.optimize!(m.lpmodel)
        for i in 1:d
            m.W[i, p] = JuMP.value(x[i] - x[i+d])
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
