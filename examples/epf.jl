using PostForecasts

year = 2023
methods = [:idr, :cp, :qr]

qf = Dict((m => Vector{QuantForecasts}(undef, 24) for m in methods)...)

for h in 1:24
    pf = loaddata(Symbol(:epex, h))
    firstday = findindex(pf, year*10_000 + 0101)
    lastday = findindex(pf, year*10_000 + 1231)
    for m in methods
        qf[m][h] = point2quant(pf, method=m, window=182, quantiles=9, first=firstday, last=lastday)
    end
end

qf[:ave] = Vector{QuantForecasts}(undef, 24)
for h in 1:24
    qf[:ave][h] = paverage([qf[m][h] for m in methods])
end

for m in [methods..., :ave]
    println(m, "\t CRPS: ", round(sum(crps.(qf[m]))/24, digits=3))
end
