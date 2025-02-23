using PostForecasts, Plots

variable = :u10 # u10, c10, t2m, t850 or z500
leadtime = 24 # between 0 and 186, divisible by 6

pf = loaddata("pangu$(leadtime)$(variable)")
plot(xlabel="Quantile level (%)", ylabel="Miscoverage (%)", framestyle=:grid, xticks = 10:10:90)

qf = point2quant(pf, method=:idr, window=365, quantiles=9)
bar!(getprob(qf).*100, (coverage(qf)-getprob(qf)).*100, linewidth=0, color=colorant"#bcbddc", label="IDR") 

conformalize!(qf, window=182)
bar!(getprob(qf).*100, (coverage(qf)-getprob(qf)).*100, linewidth=0, color=colorant"#756bb1", label="Conformalized IDR")
