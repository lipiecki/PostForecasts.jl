using PostForecasts, Plots
#theme(:vibrant)
variable = :u10 # u10, c10, t2m, t850 or z500
leadtime = 24 # between 0 and 186, divisible by 6

fs = loaddata(Symbol(:pangu, leadtime, variable))
println("$(uppercase(string(variable))) forecasts with lead time of $(leadtime) hours")

qf = point2quant(fs, method=:idr, window=364, quantiles=9)

println("\t", "-"^73)
println("\t| \t\t\t Coverage of α-quantiles \t\t\t|")
println("-"^81)
println("Model\t| α=0.1\t| α=0.2\t| α=0.3\t| α=0.4\t| α=0.5\t| α=0.6\t| α=0.7\t| α=0.8\t| α=0.9\t|")
println("-"^81)
print("IDR\t|")
for cov in coverage(qf)
    print(" ", round(cov, digits=3), "\t|")
end
println()
conformalize!(qf, window=182)

println("-"^81)
print("CIDR\t|")
for cov in coverage(qf)
    print(" ", round(cov, digits=3), "\t|")
end
println()
println("-"^81)
