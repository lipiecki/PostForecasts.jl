using PostForecasts

pf = loaddata(:epex20)
pf = pf(20200101, 20211231)
qf = Dict()

qf["QRA"] = point2quant(pf, method=:qr, window=365, quantiles=9)
qf["QRM"] = point2quant(average(pf), method=:qr, window=365, quantiles=9)
qf["QRF"] = paverage(point2quant.(decouple(pf), method=:qr, window=365, quantiles=9))
qf["QRQ"] = qaverage(point2quant.(decouple(pf), method=:qr, window=365, quantiles=9))

println("Method \t| CRPS ")
println("-"^20)
for m in ["QRA", "QRM", "QRF", "QRQ"]
    println(m, "\t| ", round(crps(qf[m]), digits=3))
end
