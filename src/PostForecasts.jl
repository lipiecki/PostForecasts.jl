module PostForecasts

abstract type Forecasts end
abstract type ProbModel end
abstract type UniRegProbModel <: ProbModel end
abstract type MultiRegProbModel <: ProbModel end

const EPEX = Dict(Symbol("epex$(H)") => "epex_hour$(H).csv" for H in 0:23)
const PANGU_U10 = Dict(Symbol("pangu$(H)u10") => ("pangu_lead$(H).csv", 2, 7) for H in 0:6:186)
const PANGU_V10 = Dict(Symbol("pangu$(H)v10") => ("pangu_lead$(H).csv", 3, 8) for H in 0:6:186)
const PANGU_T2M = Dict(Symbol("pangu$(H)t2m") => ("pangu_lead$(H).csv", 4, 9) for H in 0:6:186)
const PANGU_T850 = Dict(Symbol("pangu$(H)t850") => ("pangu_lead$(H).csv", 5, 10) for H in 0:6:186)
const PANGU_Z500 = Dict(Symbol("pangu$(H)z500") => ("pangu_lead$(H).csv", 5, 10) for H in 0:6:186)
const PANGU = Dict(PANGU_U10..., PANGU_V10..., PANGU_T2M..., PANGU_T850..., PANGU_Z500...)

import Base: getindex, firstindex, lastindex, eachindex, length
using DelimitedFiles
using HDF5
using HiGHS
using JuMP
using SpecialFunctions: erfinv
using Statistics: mean, median, quantile

include(joinpath("models", "cp.jl"))
include(joinpath("models", "idr.jl"))
include(joinpath("models", "normal.jl"))
include(joinpath("models", "qr.jl"))
include("Forecasts.jl")
include("utils.jl")
include("helpers.jl")
include("train.jl")
include("predict.jl")
include("postprocess.jl")
include("evaluation.jl")
include("averaging.jl")
include("loadnsave.jl")


export
    # Forecasts API
    PointForecasts,
    QuantForecasts,
    Forecasts,   
    length,
    getindex,
    firstindex,
    lastindex,
    eachindex,
    findindex,
    decouple,
    npred,
    setpred!,
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

    # ProbModel
    ProbModel,
    UniRegProbModel,
    MultiRegProbModel,
    train,
    predict,
    predict!,
    nregs,
    getmodel,
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
    getweights,
    nquantiles,

    # Postprocessing
    point2prob,
    conformalize,
    conformalize!,

    # Evaluation
    mae,
    rmse,
    pinball,
    coverage,

    # Utilities
    getmodel,
    nreg,
    matchwindow,
    arematching,

    # Data
    loaddata,
    loaddlmdata,
    loadpointf,
    loadquantf,
    save,

    # Statistics package methods
    mean,
    median,
    quantile
end