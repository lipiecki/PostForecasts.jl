using PostForecasts

window = 182
nquantiles = 9
hour = 12
year = 2022

pf = loaddata(Symbol(:epex, hour))
first = findindex(pf, year*10_000 + 0101)
last = findindex(pf, year*10_000 + 1231)
pf = pf[first-window:last]

# QRA
qfQRA = point2prob(pf, window, :qr, 9)

# QRM
qfQRM = point2prob(average(pf), window, :qr, nquantiles)

# QRF
qfQRF = paverage([point2prob(ipf, window, :qr, nquantiles) for ipf in decouple(pf)], nquantiles)

# QRQ
qfQRQ = qaverage([point2prob(ipf, window, :qr, nquantiles) for ipf in decouple(pf)])

println("Year $(year), hour $(hour):00")
println("Calibration window of $(window) days")
println("-"^30)
println("Model\t| Average Pinball Loss")
println("-"^30)
println("QRA\t|", round(mean(pinball(qfQRA)), digits=3))
println("QRM\t|", round(mean(pinball(qfQRM)), digits=3))
println("QRF\t|", round(mean(pinball(qfQRQ)), digits=3))
println("QRQ\t|", round(mean(pinball(qfQRF)), digits=3))
println("-"^30)