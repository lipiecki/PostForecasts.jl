"""
    QR
Struct for storing variables necessary for calibration and prediction using quantile regression.

QR(n, m, prob) creates a new structure for QR to be calibrated on `n` observations of `m` forecasts (regressors) for predicting quantiles at probabilties `prob`.
"""
struct QR <: MultiRegProbModel
    prob::Vector{Float64} # vector of probabilities for which quantile regressions are fitted
    W::Matrix{Float64} # weights of quantile regressions

    # variables for reusing the memory in consecutive trainings using Linear Programming optimization
    h::Vector{Float64}
    H::Matrix{Float64}
    lpmodel::GenericModel{Float64}

    function QR(n::Integer, m::Integer, prob::AbstractVector{<:AbstractFloat})
        lpmodel = Model(HiGHS.Optimizer)
        set_silent(lpmodel)
        set_string_names_on_creation(lpmodel, false)
        new(Float64.(vec(prob)), Matrix{Float64}(undef, m + 1, length(prob)), 
            Vector{Float64}(undef, 2(m + 1 + n)),
            Matrix{Float64}(undef, n, 2(m + 1 + n)),
            lpmodel)
    end
    function QR(n::Integer, m::Integer, prob::AbstractFloat)
        QR(n, m, [prob])
    end
end

getmodel(::Val{:qr}, params::Vararg) = QR(params[1], params[2], params[3])

matchwindow(m::QR, window::Integer) = size(m.H, 1) == window

"""
    getweights(m::QR)
Return the copy of weights matrix of the QR model `m`.
"""
function getweights(m::QR)
    return copy(m.W)
end

"""
    nquantiles(m::QR)
Return the number of quantiles matching the specification of QR model `m`.
"""
function nquantiles(m::QR)
    return length(m.prob)
end

function nreg(m::QR)
    return size(m.W, 1) - 1 # -1 to discount the intercept
end

function _train(m::QR, X::AbstractVecOrMat{<:AbstractFloat}, Y::AbstractVector{<:AbstractFloat})
    H, h, lpmodel = m.H, m.h, m.lpmodel

    for (p, α) in enumerate(m.prob)
        empty!(lpmodel)  
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

        nA, dA = size(H)
        rowindices = 1:nA
        colindices = 1:dA

        @variable(lpmodel, x[colindices] >= 0)
        @objective(lpmodel, Min, sum(h[i]*x[i] for i in colindices))
        @constraint(lpmodel, constraint[j in rowindices], sum(H[j,i]*x[i] for i in colindices) == Y[j])
        JuMP.optimize!(lpmodel)

        for i in 1:d
            m.W[i, p] = JuMP.value(x[i]) - JuMP.value(x[i+d])
        end
    end
end

function _predict(m::QR, input::Number, prob::AbstractFloat)
    j = findfirst(p -> p ≈ prob, m.prob)
    isnothing(j) && throw(ArgumentError("cannot match the model quantile to the provided probability ($(prob))"))
    return m.W[end, j] + m.W[1, j]*input
end

function _predict(m::QR, input::AbstractVector{<:Number}, prob::AbstractFloat)
    j = findfirst(p -> p ≈ prob, m.prob)
    isnothing(j) && throw(ArgumentError("cannot match the model quantile to the provided probability ($(prob))"))
    output = m.W[end, j]
    for i in eachindex(input)
        output += m.W[i, j]*input[i]
    end
    return output
end

function _predict(m::QR, input::Union{Number, AbstractVector{<:Number}})
    output = Vector{Float64}(undef, nquantiles(m))
    _predict!(m, output, input)
end

function _predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, ::Vararg{AbstractVector{<:AbstractFloat}})
    nquantiles(m) == length(output) || throw(ArgumentError("size of the output vector ($(length(prob))) does not match the model specification ($(nquantiles(m)))"))
    for j in eachindex(output)
        output[j] = m.W[end, j]
        for i in eachindex(input)
            output[j] += m.W[i, j]*input[i]
        end
    end
    sort!(output)
end

function _predict!(m::QR, output::AbstractVector{<:AbstractFloat}, input::Number, ::Vararg{AbstractVector{<:AbstractFloat}})
    nquantiles(m) == length(output) || throw(ArgumentError("size of the output vector ($(length(prob))) does not match the model specification ($(nquantiles(m)))"))
    for j in eachindex(output)
        output[j] = m.W[end, j] + m.W[1, j]*input
    end
    sort!(output)
end
