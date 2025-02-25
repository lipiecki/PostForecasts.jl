"""
    loaddata(dataset::Union{Symbol, AbstractString})
Create a `PointForecasts` object from the `dataset` provided with the package, availabe options include:
- `epexH`, where `H` is an integer between 1 and 24
- `pangu'H'u10`, where `H` is an integer between 0 and 186, divisible by 6.
- `pangu'H'v10`, where `H` is an integer between 0 and 186, divisible by 6.
- `pangu'H't2m`, where `H` is an integer between 0 and 186, divisible by 6.
- `pangu'H't850`, where `H` is an integer between 0 and 186, divisible by 6.
- `pangu'H'z500`, where `H` is an integer between 0 and 186, divisible by 6.

Details of the datasets are available in documentation.
"""
function loaddata(dataset::Symbol)
    if dataset ∈ keys(PANGU)
        loaddlmdata(joinpath(@__DIR__, "..", "data", "pangu", PANGU[dataset][1]), idcol=1, predcol = PANGU[dataset][2], obscol = PANGU[dataset][3], colnames=true)
    elseif dataset ∈ keys(EPEX)
        loaddlmdata(joinpath(@__DIR__, "..", "data", "epex", EPEX[dataset]), idcol=1, obscol=2, colnames=true)
    else
        throw(ArgumentError("$(dataset) is not a valid dataset name"))
    end
end

loaddata(dataset::AbstractString) = loaddata(Symbol(dataset))

"""
    loaddlmdata(filepath::AbstractString; kwargs...)
Create a `PointForecasts` object from delimited file at `filepath`.
## Keyword Arguments
- `delim=','`: Specifies the delimitter
- `obscol=1: Specifies which column is used for observations
- `predcol=nothing`: Specifies which columns are used for pred (omit to use all remaining columns)
- `idcol=nothing`: Specifies which column is used for timestamps (omit to generate timestamps automatically)
- `colnames=false` If true, omit the first row of the file.
"""
function loaddlmdata(filepath::AbstractString; delim::Char=',', obscol::Integer=1, idcol::Union{Nothing, Integer}=nothing,  predcol::Union{Nothing, Integer, AbstractVector{<:Integer}}=nothing, colnames::Bool=false)
    data = readdlm(filepath, delim)[(colnames ? 2 : 1):end, :]
    l, m = size(data)
    return PointForecasts(
        collect(Float64, data[:, (isnothing(predcol) ? [i for i in 1:m if i ∉ [idcol, obscol]] : predcol)]),
        collect(Float64, data[:, obscol]),
        isnothing(idcol) ? Int.(1:l) : Int.(data[:, idcol]))
end

"""
    saveforecasts(f::Forecasts, filepath::AbstractString)
Save `f` to a HDF5 file at `filepath` with `.pointf` extension for `PointForecasts` and `.quantf` extension for `QuantForecasts` (extension is added if missing).
"""
function saveforecasts(pf::PointForecasts, filepath::AbstractString)
    ext = (splitext(filepath)[2] == ".pointf") ? "" : ".pointf"
    h5open(filepath*ext, "w") do fid
        fid["pred"] = pf.pred
        fid["obs"] = pf.obs
        fid["id"] = pf.id
    end
end

function saveforecasts(qf::QuantForecasts, filepath::AbstractString)
    ext = (splitext(filepath)[2] == ".quantf") ? "" : ".quantf"
    h5open(filepath*ext, "w") do fid
        fid["pred"] = qf.pred
        fid["obs"] = qf.obs
        fid["id"] = qf.id
        fid["prob"] = qf.prob
    end
end

"""
    loadforecasts(filepath::AbstractString)::Forecasts
Load `PointForecasts` or `QuantForecasts` from `filepath` (`.pointf` or `.quantf` extension is required).
"""
function loadforecasts(filepath::AbstractString)::Forecasts
    ext = splitext(filepath)[2] 
    if ext == ".pointf"
        loadpointf(filepath)
    elseif ext == ".quantf"
        loadquantf(filepath)
    else
        throw(ArgumentError("`.pointf` or `.quantf` extension required"))
    end
end

"""
    loadpointf(filepath::AbstractString)::PointForecasts
Load `PointForecasts` from `filepath` (`.pointf` extension is added if missing).
"""
function loadpointf(filepath::AbstractString)::PointForecasts
    ext = (splitext(filepath)[2] == ".pointf") ? "" : ".pointf"
    h5open(filepath*ext, "r") do fid
        PointForecasts(
            read(fid, "pred"),
            read(fid, "obs"),
            read(fid, "id"))
    end
end

"""
    loadquantf(filepath::AbstractString)::QuantForecasts
Load `QuantForecasts` from `filepath` (`.quantf` extension is added if missing).
"""
function loadquantf(filepath::AbstractString)::QuantForecasts
    ext = (splitext(filepath)[2] == ".quantf") ? "" : ".quantf"
    h5open(filepath*ext, "r") do fid
        QuantForecasts(
            read(fid, "pred"),
            read(fid, "obs"),
            read(fid, "id"),
            read(fid, "prob"))
    end
end
