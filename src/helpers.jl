# helper functions that are not exported with the package, inteded for internal use, lack argument validation

"""
    equidistant(n::Integer, f::Type{F}=Float64) where {F<:AbstractFloat}
Return a `Vector{F}` of `n` equidistant quantile levels.
"""
function equidistant(n::Integer, type::Type{F}=Float64) where {F<:AbstractFloat}
    res = Vector{F}(undef, n)
    for i in 1:n
        res[i] = i/(n+1)
    end
    return res
end

"""
    isunique(X::Vector{<:Integer})
Return `true` if vector `X` contains only unique values, otherwise return `false`.
"""
function isunique(X::AbstractVector{<:Integer})
    iterated = Set{eltype(X)}()
    for x in X
        if x ∈ iterated
            return false
        else
            push!(iterated, x)
        end
    end
    return true
end

function _autodiff(f::Function)
    function nlopt_fn(x::Vector, grad::Vector)
        if length(grad) > 0
            ForwardDiff.gradient!(grad, f, x)
        end
        return f(x)
    end
end

function _config_solver_threads(lpmodel::GenericModel)
    Threads.threadid() > 1 && @warn "configuring solver parallelization is not thread-safe, construct the model in the main thread to avoid issues"
    if get_hyperparam(:parsol) & Threads.nthreads() > 1
        Highs_resetGlobalScheduler(1)
        set_attribute(lpmodel, MOI.NumberOfThreads(), Threads.nthreads())
    elseif !get_hyperparam(:parsol)
        Highs_resetGlobalScheduler(1)
        set_attribute(lpmodel, MOI.NumberOfThreads(), 1)
    end
end

function normal_cdf(x::Number)
    return 0.5 * (1 + erf(x / sqrt(2)))
end

function normal_pdf(x::Number)
    return exp(-0.5 * x^2) / sqrt(2 * π)
end
