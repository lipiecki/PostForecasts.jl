"""
    IDR
Struct for storing variables necessary for calibration and prediction using [Isotonic Distributional Regression](https://doi.org/10.1111/rssb.12450)

IDR(n, m) creates a new structure for IDR to be calibrated on `n` observations of `m` forecasts (regressors).
"""
struct IDR <: MultiRegProbModel
    cdf::Array{Float64, 3} # Array storing estimated conditional `CDF`s, where `F[i, j, k]` is the `CDF(y[j])` given that `k`-th regressor value is `x[i, k]`.
    x::Matrix{Float64} # Matrix storing regressor values, where `x[1:lastx[k], k]` are values of the `k`-th regressor
    y::Vector{Float64} # Vector storing response values
    lastx::Vector{UInt16} # Vector of the numbers of unique values of each regressor
    lasty::Base.RefValue{UInt16} # Number of unique response values

    # variables for reusing the memory in consecutive trainings using Abridged Pool-Adjacent-Violators Algorithm (see https://doi.org/10.1007/s11009-022-09937-2)
    P::Vector{UInt16}
    Pprev:: Vector{UInt16}
    W::Vector{UInt16}
    Wprev::Vector{UInt16}
    M::Vector{Float64}
    Mprev::Vector{Float64}
    z::Vector{Float64}
    w::Vector{UInt16}
    order¹::Vector{UInt16}
    order²::Vector{UInt16}
    keys::Vector{UInt16}
    
    IDR(n::Integer, m::Integer) = new(
        Array{Float64}(undef, n, n, m),
        Matrix{Float64}(undef, n, m),
        Vector{Float64}(undef, n),
        Vector{UInt16}(undef, m),
        Ref{UInt16}(),
        Vector{UInt16}(undef, n),
        Vector{UInt16}(undef, n),
        Vector{UInt16}(undef, n),
        Vector{UInt16}(undef, n),
        Vector{UInt16}(undef, n),
        Vector{UInt16}(undef, n),
        Vector{UInt16}(undef, n),
        Vector{UInt16}(undef, n),
        Vector{UInt16}(undef, n),
        Vector{UInt16}(undef, n),
        Vector{UInt16}(undef, n))
end

getmodel(::Val{:idr}, params::Vararg) = IDR(params[1], params[2])

matchwindow(m::IDR, window::Integer) = size(m.cdf, 1) == window

"""
    getx(m [, r])
Return the copy of regressor values from `m::IDR` on which cumulative dsitribution function is specified. Optional argument `r::Integer = 1` corresponds to the regressor index.
"""
function getx(m::IDR, r::Integer = 1)
    return m.x[1:m.lastx[r], r]
end

"""
    gety(m)
Return the copy of response values from `m::IDR` on which cumulative dsitribution function is specified.
"""
function gety(m::IDR)
    return m.y[1:m.lasty[]]
end

"""
    getcdf(m [, r])
Return the copy of cumulative distribution function from `m::IDR`. Optional argument `r::Integer = 1` corresponds to the regressor index.
"""
function getcdf(m::IDR, r::Integer = 1)
    return m.cdf[1:m.lastx[r], 1:m.lasty[], r]
end

function nreg(m::IDR)
    return size(m.cdf, 3)
end

function _train(m::IDR, X::AbstractVecOrMat{<:AbstractFloat}, Y::AbstractVector{<:AbstractFloat})
    cdf, x, y, lastx, lasty, P, Pprev, W, Wprev, M, Mprev, z, w, order¹, order², keys = 
        m.cdf, m.x, m.y, m.lastx, m.lasty, m.P, m.Pprev, m.W, m.Wprev, m.M, m.Mprev, m.z, m.w, m.order¹, m.order², m.keys

    npreds = size(cdf, 3)
    n = length(y)

    fill!(cdf, 1.0)

    for p in 1:npreds
        
        fill!(w, 0)
        fill!(z, 0.0)
        fill!(P, 0)
        fill!(Pprev, 0)
        fill!(W, 0)
        fill!(Wprev, 0)
        fill!(M, 0.0)
        fill!(Mprev, 0.0)

        sortperm!(order¹, @view(X[:, p]))
        sortperm!(order², @view(Y[order¹])) 

        if p == 1
            lasty[] = 1
            y[1] = Y[order¹[order²[1]]]
            for i in 2:n
                if !((Y[order¹[order²[i]]]) ≈ Y[order¹[order²[i-1]]])
                    y[lasty[]+=1] = Y[order¹[order²[i]]]
                end
            end
        end

        lastx[p] = 1
        x[1, p] = X[order¹[1], p]
        w[1] = 1
        keys[1] = 1
        for i in 2:n
            if !(X[order¹[i], p] ≈ X[order¹[i-1], p])
                x[lastx[p]+=1, p] = X[order¹[i], p]
            end
            w[lastx[p]] += 1
            keys[i] = lastx[p]
        end

        # intialize for 0-th statistic
        P[1] = lastx[p]
        Pprev[1] = P[1]
        W[1] = n
        Wprev[1] = W[1]
        
        d₋ = 1
        k = 1
        t = 1
        
        while Y[order¹[order²[t]]] < Y[order¹[order²[end]]]
            iₓ = keys[order²[t]]
            z[iₓ] += 1.0/w[iₓ]
            
            s = 1
            while Pprev[s] < iₓ
                s += 1
            end
            
            P[s] = iₓ
            W[s] = 0
            M[s] = 0.0

            for i in (s > 1 ? Pprev[s-1] + 1 : 1):iₓ
                W[s] += w[i]
                M[s] += w[i]*z[i]
            end

            M[s] /= W[s]
            d = s
            
            # Initial pooling:
            while d > 1 && (M[d-1] <= M[d] || M[d-1] ≈ M[d])
                P[d-1] = P[d]
                M[d-1] = (W[d]*M[d] + W[d-1]*M[d-1]) / (W[d] + W[d-1])
                W[d-1] = W[d] + W[d-1]
                d -= 1
            end

            # Induction:
            nextindex = P[d]
            while nextindex < Pprev[s]
                nextindex = P[d] + 1
                d += 1
                P[d] = nextindex
                W[d] = w[nextindex]
                M[d] = z[nextindex]
                while nextindex < Pprev[s] && z[nextindex + 1] ≈ M[d]
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

            # Finalization:
            if Pprev[s] < lastx[p]
                for i in 1:(d₋ - s)
                    P[d+i] = Pprev[s+i]
                    W[d+i] = Wprev[s+i]
                    M[d+i] = Mprev[s+i]
                end
                d = d + d₋ - s
            end
            
            for i in 1:d
                Pprev[i] = P[i]
                Wprev[i] = W[i]
                Mprev[i] = M[i]
            end

            last = 1
            if Y[order¹[order²[t]]] < Y[order¹[order²[t+1]]]
                for i in 1:d
                    for j in last:P[i]
                        cdf[j, k, p] = M[i]
                    end
                    last = P[i] + 1
                end
                k += 1
            end
            t += 1
            d₋ = d
        end
    end
end

function _predict(m::IDR, input::Number, prob::AbstractFloat)
    y = @view(m.y[1:m.lasty[]])
    for j in 1:m.lasty[]
        p = 0.0
        x = @view(m.x[1:m.lastx[1], 1])
        i = searchsortedfirst(x, input)
        if i > m.lastx[1]
            p += m.cdf[m.lastx[1], j, 1]
        elseif i == 1
            p += m.cdf[i, j, 1]
        else
            p += m.cdf[i-1, j, 1] + (m.cdf[i, j, 1] - m.cdf[i-1, j, 1])*(input - x[i-1])/(x[i] - x[i-1])
        end
        if p >= prob - eps(typeof(prob))
            return y[j]
        end
    end
    return y[end]
end

function _predict(m::IDR, input::AbstractVector{<:Number}, prob::AbstractFloat)
    y = @view(m.y[1:m.lasty[]])
    for j in 1:m.lasty[]
        p = 0.0
        f = 0
        for val in input
            f += 1
            x = @view(m.x[1:m.lastx[f], f])
            i = searchsortedfirst(x, val)
            if i > m.lastx[f]
                p += m.cdf[m.lastx[f], j, f]
            elseif i == 1
                p += m.cdf[i, j, f]
            else
                p += m.cdf[i-1, j, f] + (m.cdf[i, j, f] - m.cdf[i-1, j, f])*(val - x[i-1])/(x[i] - x[i-1])
            end
        end
        if p/f >= prob - eps(typeof(prob))
            return y[j]
        end
    end
    return y[end]
end

function _predict!(m::IDR, output::AbstractVector{<:AbstractFloat}, input::Union{Number, AbstractVector{<:Number}}, prob::AbstractVector{<:AbstractFloat})
    for j in eachindex(output)
        output[j] = _predict(m, input, prob[j])
    end
end
