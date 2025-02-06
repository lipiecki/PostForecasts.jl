"""
    IDR([type::Type{F}=Float64,] n::Integer, r::Integer) where {F<:AbstractFloat}
Creates an `IDR{F}<:MultiPostModel{F}<:PostModel{F}` model for isotonic distributional regression to be trained on `n` observations with `r` forecasts (regressors).
"""
struct IDR{F<:AbstractFloat} <: MultiPostModel{F}
    cdf::Array{F, 3} # Array for storing estimated conditional CDFs
    domain::Matrix{F} # Matrix for storing the domain of conditional CDFs, i.e. unique values of regressors and response
    domainsize::Vector{Int} # Vector for storing the number of unique values for each domain dimension
    predstate::Vector{F} # Vector for storing CDF conditioned on input regressors
    
    # variables for running Abridged Pool-Adjacent-Violators Algorithm (see https://doi.org/10.1007/s11009-022-09937-2)
    P::Vector{Int}
    P_:: Vector{Int}
    W::Vector{Int}
    W_::Vector{Int}
    M::Vector{F}
    M_::Vector{F}
    z::Vector{F}
    w::Vector{Int}
    order::Vector{Int}
    order2::Vector{Int}
    keys::Vector{Int}
    
    function IDR(::Type{F}, n::Integer, r::Integer) where {F<:AbstractFloat} 
        new{F}(
            Array{F}(undef, n, n, r),
            Matrix{F}(undef, n, r+1),
            Vector{Int}(undef, r+1),
            Vector{F}(undef, n),
            Vector{Int}(undef, n),
            Vector{Int}(undef, n),
            Vector{Int}(undef, n),
            Vector{Int}(undef, n),
            Vector{Int}(undef, n),
            Vector{Int}(undef, n),
            Vector{Int}(undef, n),
            Vector{Int}(undef, n),
            Vector{Int}(undef, n),
            Vector{Int}(undef, n),
            Vector{Int}(undef, n))
    end
end

IDR(n::Integer, r::Integer) = IDR(Float64, n, r)

getmodel(::Type{F}, ::Val{:idr}, params::Vararg) where {F<:AbstractFloat} = IDR(F, params[1], params[2])

matchwindow(m::IDR, window::Integer) = size(m.cdf, 1) == window

"""
    getx(m::IDR [, r])
Return a vector of regressor values from model `m` on which cumulative dsitribution function is defined. Optional argument `r::Integer = 1` corresponds to the regressor index.
"""
function getx(m::IDR, r::Integer = 1)
    (r < 1 || r >= length(m.domainsize)) && throwDomainError("`r` must be between 1 and $(m.domainsize-1)")
    return m.domain[1:m.domainsize[r], r]
end

"""
    gety(m::IDR)
Return a vector of response values from model `m` on which cumulative dsitribution function is defined.
"""
function gety(m::IDR)
    return m.domain[1:m.domainsize[end], end]
end

"""
    getcdf(m::IDR [, r])
Return a vector of cumulative distribution function values from model `m`. Optional argument `r::Integer = 1` corresponds to the regressor index.
"""
function getcdf(m::IDR, r::Integer = 1)
    (r < 1 || r >= length(m.domainsize)) && throwDomainError("`r` must be between 1 and $(m.domainsize-1)")
    return m.cdf[1:m.domainsize[r], 1:m.domainsize[end], r]
end

function nreg(m::IDR)
    return length(m.domainsize)-1
end

function _train(m::IDR, X::AbstractVecOrMat{<:Number}, Y::AbstractVector{<:Number})::Nothing
    cdf, domain, domainsize, P, P_, W, W_, M, M_, z, w, order, order2, keys = 
        m.cdf, m.domain, m.domainsize, m.P, m.P_, m.W, m.W_, m.M, m.M_, m.z, m.w, m.order, m.order2, m.keys

    n = length(Y)

    fill!(cdf, 1.0)

    for r in axes(X, 2)
        fill!(w, 0)
        fill!(z, 0.0)
        fill!(P, 0)
        fill!(P_, 0)
        fill!(W, 0)
        fill!(W_, 0)
        fill!(M, 0.0)
        fill!(M_, 0.0)

        order .= 1:n
        order2 .= order
        sort!(order, by=(i -> X[i, r]))
        sort!(order2, by=(i -> Y[order[i]]))

        if r == 1
            domainsize[end] = 1
            domain[1, end] = Y[order[order2[1]]]
            for i in 2:n
                if !((Y[order[order2[i]]]) ≈ Y[order[order2[i-1]]])
                    domain[domainsize[end]+=1, end] = Y[order[order2[i]]]
                end
            end
        end

        domainsize[r] = 1
        domain[1, r] = X[order[1], r]
        w[1] = 1
        keys[1] = 1
        for i in 2:n
            if !(X[order[i], r] ≈ X[order[i-1], r])
                domain[domainsize[r]+=1, r] = X[order[i], r]
            end
            w[domainsize[r]] += 1
            keys[i] = domainsize[r]
        end

        # intialize for the 0th order statistics
        P[1] = domainsize[r]
        P_[1] = P[1]
        W[1] = n
        W_[1] = W[1] 
        d_ = 1
        k = 1
        t = 1

        # calculate for consecutive kth order statistics
        while !(Y[order[order2[t]]] ≈ Y[order[order2[end]]])
            idx = keys[order2[t]]
            z[idx] += 1.0/w[idx]
            s = 1
            while P_[s] < idx
                s += 1
            end
            P[s] = idx
            W[s] = 0
            M[s] = 0.0
            for i in (s > 1 ? P_[s-1] + 1 : 1):idx
                W[s] += w[i]
                M[s] += w[i]*z[i]
            end
            M[s] /= W[s]
            d = s
            
            # initial pooling
            while d > 1 && (M[d-1] <= M[d] || M[d-1] ≈ M[d])
                P[d-1] = P[d]
                M[d-1] = (W[d]*M[d] + W[d-1]*M[d-1]) / (W[d] + W[d-1])
                W[d-1] = W[d] + W[d-1]
                d -= 1
            end

            # induction step
            nextindex = P[d]
            while nextindex < P_[s]
                nextindex = P[d] + 1
                d += 1
                P[d] = nextindex
                W[d] = w[nextindex]
                M[d] = z[nextindex]
                while nextindex < P_[s] && z[nextindex + 1] ≈ M[d]
                    nextindex += 1
                    P[d] = nextindex
                    W[d] += w[nextindex]
                end
                while d > 1 && (M[d-1] <= M[d] || M[d-1] ≈ M[d])
                    P[d-1] = P[d]
                    M[d-1] = (W[d]*M[d] + W[d-1]*M[d-1]) / (W[d] + W[d-1])
                    W[d-1] = W[d] + W[d-1]
                    d -= 1
                end
            end

            # finalization step
            if P_[s] < domainsize[r]
                for i in 1:(d_ - s)
                    P[d+i] = P_[s+i]
                    W[d+i] = W_[s+i]
                    M[d+i] = M_[s+i]
                end
                d = d + d_ - s
            end
            for i in 1:d
                P_[i] = P[i]
                W_[i] = W[i]
                M_[i] = M[i]
            end
            last = 1
            if !(Y[order[order2[t]]] ≈ Y[order[order2[t+1]]])
                for i in 1:d
                    for j in last:P[i]
                        cdf[j, k, r] = M[i]
                    end
                    last = P[i] + 1
                end
                k += 1
            end
            t += 1
            d_ = d
        end
    end
    return nothing
end

function _setpredstate(m::IDR, input::AbstractVector{<:Number})::Nothing
    fill!(m.predstate, 0.0)
    for (f, inp) in enumerate(input)
        i = searchsortedfirst(@view(m.domain[1:m.domainsize[f], f]), inp)
        if i > m.domainsize[f]
            m.predstate .= m.predstate .+ @view(m.cdf[m.domainsize[f], :, f])
        elseif i == 1
            m.predstate .= m.predstate .+ @view(m.cdf[i, :, f])
        else
            val = (inp - m.domain[i-1, f])/(m.domain[i, f] - m.domain[i-1, f])
            m.predstate .= m.predstate .+ @view(m.cdf[i-1, :, f]).*(1-val)
            m.predstate .= m.predstate .+ @view(m.cdf[i, :, f]).*(val)
        end
    end
    m.predstate .= m.predstate./length(input)
    return nothing
end

function _predict(m::IDR, input::AbstractVector{<:Number}, prob::AbstractFloat)
    _setpredstate(m, input)
    for j in 1:m.domainsize[end]
        if m.predstate[j] > prob - 1e-9
            return m.domain[j, end]
        end
    end
end

function _predict(m::IDR{F}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat}) where {F<:AbstractFloat}
    output = Vector{F}(undef, length(prob))
    _setpredstate(m, input)
    itr = 1
    for i in eachindex(output)
        for j in itr:m.domainsize[end]
            if m.predstate[j] > prob[i] - 1e-9
                output[i] = m.domain[j, end]
                itr = j
                break
            end
        end
    end
    return output
end

function _predict!(m::IDR, output::AbstractVector{<:AbstractFloat}, input::AbstractVector{<:Number}, prob::AbstractVector{<:AbstractFloat})::Nothing
    _setpredstate(m, input)
    itr = 1
    for i in eachindex(output)
        for j in itr:m.domainsize[end]
            if m.predstate[j] > prob[i] - 1e-9
                output[i] = m.domain[j, end]
                itr = j
                break
            end
        end
    end
    return nothing
end
