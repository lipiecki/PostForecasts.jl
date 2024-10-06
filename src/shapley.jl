"""
    shapley(FS::Vector{<:Forecasts{F, I}}, agg::Function, payoff::Function[, null_payoff::AbstractFloat])
Calculate Shapley values of forecasters in `FS`, using specified `agg`regation function and `payoff` function.

Optional argument `null_payoff` is the payoff value for an empty coalition. If not provided, empty coalition is excluded from calculations.

Return a vector of Shapley values correspinding to each forecaster in `FS`.
"""
function shapley(FS::Vector{<:Forecasts{F, I}}, agg::Function, payoff::Function) where {F, I}
    M = length(FS)
    vals = zeros(Float64, M)
    for s in 1:M-1
        for coalition in combinations(1:M, s)
            coalition_payoff = payoff(agg(FS[coalition]))
            for i in 1:M
                if i ∉ coalition
                    vals[i] += (payoff(agg(FS[vcat(coalition, i)])) - coalition_payoff)/binomial(M-1, s)
                end
            end
        end
    end
    vals /= M
    return vals
end

function shapley(FS::Vector{<:Forecasts{F, I}}, agg::Function, payoff::Function, null_payoff::AbstractFloat) where {F, I}
    M = length(FS)
    vals = zeros(Float64, M)
    for i in 1:M
        vals[i] += (payoff(FS[i]) - null_payoff)/binomial(M-1, 0)
    end
    for s in 1:M-1
        for coalition in combinations(1:M, s)
            coalition_payoff = payoff(agg(FS[coalition]))
            for i in 1:M
                if i ∉ coalition
                    vals[i] += (payoff(agg(FS[vcat(coalition, i)])) - coalition_payoff)/binomial(M-1, s)
                end
            end
        end
    end
    vals /= M
    return vals
end
