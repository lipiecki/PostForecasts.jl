"""
    shapley(fs::AbstractVector{<:T}, agg::Function, payoff::Function[, ∅::AbstractFloat]) where T<:Union{PointForecasts, QuantForecasts}
Calculate Shapley values of forecasters in `fs`, using specified `agg`regation function and `payoff` function.

Optional argument `∅` is the payoff value for an empty coalition. If not provided, empty coalition is excluded from calculations.

Return a vector of Shapley values correspinding to each forecaster in `fs`.
"""
function shapley(fs::AbstractVector{<:T}, agg::Function, payoff::Function, ∅::Union{Nothing, AbstractFloat}=nothing) where T<:Union{PointForecasts, QuantForecasts}
    m = length(fs)
    vals = zeros(m)
    
    if ∅ !== nothing
        for i in 1:m
            vals[i] += (payoff(fs[getindex(eachindex(fs), i)]) - ∅)/binomial(m-1, 0)
        end
    end
    
    for s in 1:m-1
        for coalition in combinations(1:m, s)
            coalition_payoff = payoff(agg(fs[getindex(eachindex(fs), coalition)]))
            for i in 1:m
                if i ∉ coalition
                    vals[i] += (payoff(agg(fs[getindex(eachindex(fs), [coalition; i])])) - coalition_payoff)/binomial(m-1, s)
                end
            end
        end
    end
    
    vals /= m
    return vals
end
