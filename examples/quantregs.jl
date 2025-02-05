using PostForecasts

using PostForecasts
pf = loaddata(Symbol(:epex, 15))
first = findindex(pf, 20210101)
last = findindex(pf, 20211231)
pf = pf[first-182:last]
qf = Dict()

# QRA
qf["QRA"] = point2quant(pf, model=:qr, window=182, quantiles=9)

# QRM
qf["QRM"] = point2quant(average(pf), model=:qr, window=182, quantiles=9)

# QRF
qf["QRF"] = paverage(point2quant.(decouple(pf), model=:qr, window=182, quantiles=9))

# QRQ
qf["QRQ"] = qaverage(point2quant.(decouple(pf), model=:qr, window=182, quantiles=9))

for key in keys(qfs)
    println(key, "\t CRPS: ", round(crps(qfs[key]), digits=3))
end
