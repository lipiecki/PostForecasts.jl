using PostForecasts
pf = loaddata(:epex20)
pf = pf(20200101, 20210101)
qf = Dict()

qf["QRA"] = point2quant(pf, method=:qr, window=365, quantiles=9)

qf["QRM"] = point2quant(average(pf), method=:qr, window=365, quantiles=9)

qf["QRF"] = paverage(point2quant.(decouple(pf), method=:qr, window=365, quantiles=9))

qf["QRQ"] = qaverage(point2quant.(decouple(pf), method=:qr, window=365, quantiles=9))

println("Method \t| CRPS ")
println("-"^20)
for method in ["QRA", "QRM", "QRF", "QRQ"]
    println(method, "\t| ", round(crps(qf[method]), digits=3))
end
