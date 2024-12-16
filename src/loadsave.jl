"""
    loaddata(dataset::Symbol)
Create a `PointForecasts` object from the `dataset` provided with the package, availabe options include:
- `:epexH`, where `H` is an integer between 1 and 24
- `:pangu'H'u10`, where `H` is an integer between 0 and 186, divisible by 6.
- `:pangu'H'v10`, where `H` is an integer between 0 and 186, divisible by 6.
- `:pangu'H't2m`, where `H` is an integer between 0 and 186, divisible by 6.
- `:pangu'H't850`, where `H` is an integer between 0 and 186, divisible by 6.
- `:pangu'H'z500`, where `H` is an integer between 0 and 186, divisible by 6.

Details of the datasets are avaiable in documentation.
"""
function loaddata(dataset::Symbol)
    if dataset ∈ keys(PANGU)
        loaddlmdata(joinpath(@__DIR__, "..", "data", "pangu", PANGU[dataset][1]), predcol = PANGU[dataset][2], obscol = PANGU[dataset][3], colnames=true)
    elseif dataset ∈ keys(EPEX)
        loaddlmdata(joinpath(@__DIR__, "..", "data", "epex", EPEX[dataset]), colnames=true)
    else
        throw(ArgumentError("$(dataset) is not a valid dataset name"))
    end
end

"""
    loaddlmdata(filepath::String; kwargs...)
Create a `PointForecasts` object from delimited file at `filepath`.
## Keyword Arguments
- `delim=','`: Specifies the delimitter
- `idcol=1`: Specifies which column is used for timestamps (`0` to generate timestamps automatically)
- `obscol=2`: Specifies which column is used for observations
- `predcol=0`: Specifies which columns are used for pred (`0` to use all remaining columns)
- `colnames=false` If true, omit the first row of the file.
"""
function loaddlmdata(filepath::String; delim::Char=',', idcol::Integer=1, obscol::Integer=2, predcol::Union{AbstractVector{Integer}, Integer}=0, colnames::Bool=false)
    data = readdlm(filepath, delim)[(colnames ? 2 : 1):end, :]
    l, m = size(data)
    predcol = (ndims(predcol) == 0 && predcol == 0) ? [i for i in 1:m if i ∉ [idcol, obscol]] : predcol
    return PointForecasts(
        collect(Float64, data[:, predcol]),
        collect(Float64, data[:, obscol]),
        idcol == 0 ? Int.(1:l) : Int.(data[:, idcol]))
end

"""
    save(f::Forecasts, filepath::String, groupname::String="forecasts")
Save `f` to a HDF5 file at `filepath` (`.pointf` or `.quantf` extension is added if missing). Optionally specify the name of the group, in which the data will be saved. 
"""
function save(pf::PointForecasts, filepath::String, groupname::String="forecasts")
    ext = filepath[end-6:end] == ".pointf" ? "" : ".pointf"
    h5open(filepath*ext, "w") do fid
        g = create_group(fid, groupname)
        g["pred"] = pf.pred
        g["obs"] = pf.obs
        g["id"] = pf.id
    end
end

function save(qf::QuantForecasts, filepath::String, groupname::String="forecasts")
    ext = filepath[end-6:end] == ".quantf" ? "" : ".quantf"
    h5open(filepath*ext, "w") do fid
        g = create_group(fid, groupname)
        g["pred"] = qf.pred
        g["obs"] = qf.obs
        g["id"] = qf.id
        g["levels"] = qf.prob
    end
end

"""
    loadpointf(filepath::String, groupname::String="forecasts")
Load `PointForecasts` from a HDF5 file at `filepath` (`.pointf` extension is added if missing). Optionally specify the name of the group, from which the data will be loaded.
"""
function loadpointf(filepath::String, groupname::String="forecasts")
    ext = filepath[end-6:end] == ".pointf" ? "" : ".pointf"
    h5open(filepath*ext, "r") do fid
        g = fid[groupname]
        PointForecasts(
            read(g, "pred"),
            read(g, "obs"),
            read(g, "id"))
    end
end

"""
    loadquantf(filepath::String, groupname::String="forecasts")
Load `QuantForecasts` from a HDF5 file at `filepath` (`.quantf` extension is added if missing). Optionally specify the name of the group, from which the data will be loaded.
"""
function loadquantf(filepath::String, groupname::String="forecasts")
    ext = filepath[end-6:end] == ".quantf" ? "" : ".quantf"
    h5open(filepath*ext, "r") do fid
        g = fid[groupname]
        QuantForecasts(
            read(g, "pred"),
            read(g, "obs"),
            read(g, "id"),
            read(g, "levels"))
    end
end
