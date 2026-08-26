module PostForecasts

abstract type PostModel{F<:AbstractFloat} end
abstract type UniPostModel{F<:AbstractFloat} <: PostModel{F} end
abstract type MultiPostModel{F<:AbstractFloat} <: PostModel{F} end

# constants for dataset handling
const EPEX = Dict(Symbol("epex$(H)") => "epex_hour$(H).csv" for H in 1:24)
const PANGU = Dict((Symbol("pangu$(H)u10") => ("pangu_lead$(H).csv", 2, 7) for H in 0:6:186)...,
    (Symbol("pangu$(H)v10") => ("pangu_lead$(H).csv", 3, 8) for H in 0:6:186)...,
    (Symbol("pangu$(H)t2m") => ("pangu_lead$(H).csv", 4, 9) for H in 0:6:186)...,
    (Symbol("pangu$(H)t850") => ("pangu_lead$(H).csv", 5, 10) for H in 0:6:186)...,
    (Symbol("pangu$(H)z500") => ("pangu_lead$(H).csv", 6, 11) for H in 0:6:186)...)

# hyperparameter constants
const HYPERPARAMS = Dict{Symbol, Union{Vector{<:Any}, Ref{<:Any}}}(
    :highs_nthreads => Ref{Int}(0),
    :garch_solver => Ref{Symbol}(:LD_CCSAQ),
    :sqr_solver => Ref{Symbol}(:LD_SLSQP),
    :minlambda => Ref{Float64}(1e-5),
    :nlambdas => Ref{Int}(20),
    :reltol => Ref{Float64}(0.0),
    :abstol => Ref{Float64}(1e-7),
    :maxeval => Ref{Int}(1_000)
)

import Base: getindex, firstindex, lastindex, eachindex, length, show
using Combinatorics: combinations
using DelimitedFiles
using HDF5
using HiGHS
using JuMP
using NLopt
using ForwardDiff
using LinearAlgebra
using SpecialFunctions: erf, erfinv
using Statistics: mean, median, quantile

include(joinpath("models", "cp.jl"))
include(joinpath("models", "idr.jl"))
include(joinpath("models", "normal.jl"))
include(joinpath("models", "qr.jl"))
include(joinpath("models", "lassoqr.jl"))
include(joinpath("models", "sqr.jl"))
include(joinpath("models", "garch.jl"))
include("Forecasts.jl")
include("helpers.jl")
include("utils.jl")
include("train.jl")
include("predict.jl")
include("point2quant.jl")
include("conformalize.jl")
include("evaluation.jl")
include("averaging.jl")
include("shapley.jl")
include("loadsave.jl")

export
    # Forecasts API
    PointForecasts,
    QuantForecasts,
    Forecasts,
    show,
    length,
    getindex,
    firstindex,
    lastindex,
    eachindex,
    findindex,
    couple,
    decouple,
    npred,
    setpred,
    getpred,
    getobs,
    getid,
    getprob,
    viewpred,
    viewobs,
    viewid,
    viewprob,

    # Averaging
    average,
    qaverage,
    paverage,

    # PostModel
    PostModel,
    UniPostModel,
    MultiPostModel,
    train,
    predict,
    predict!,
    nregs,

    # Conformal Prediction
    CP,
    getscores,

    # Isotonic Distributional Regression
    IDR,
    getx,
    gety,
    getcdf,

    # Normal
    Normal,
    getmean,
    getstd,

    # Quantile Regression
    QR,
    iQR,
    LassoQR,
    SQR,
    iSQR,
    getweights,
    getlambdas,
    getquantprob,

    # GARCH
    GARCH,
    getparams,

    # Postprocessing
    point2quant,
    conformalize,
    conformalize!,

    # Evaluation
    mae,
    mape,
    smape,
    mse,
    pinball,
    crps,
    coverage,
    shapley,

    # Utilities
    getmodel,
    nreg,
    matchwindow,
    checkmatch,
    set_hyperparam,
    get_hyperparam,
    advance!,
    
    # Data
    loaddata,
    loaddlm,
    saveforecasts,
    loadforecasts,
    loadpointf,
    loadquantf
end
