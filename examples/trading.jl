using PostForecasts, Plots

fsBUY = loaddata(Symbol(:epex, 4))      # buy at 3:00
fsSELL = loaddata(Symbol(:epex, 20))    # sell at 19:00

qfBUY = point2quant(fsBUY, method=:idr, window=182, quantiles=9, start=20230408, stop=20230421)
qfSELL = point2quant(fsSELL, method=:idr, window=182, quantiles=9, start=20230408, stop=20230421)

#theme(:dark)
plot(legend = :bottom, xlabel = "Day", ylabel = "Price (€/MWh)", xticks = 1:14, framestyle = :box)
plot!(viewpred(qfSELL, eachindex(qfBUY), 5), linealpha = 0.5, color=colorant"#FE4365", lw=3, label="Sell price")
plot!(viewpred(qfBUY, eachindex(qfBUY), 5), linealpha = 0.5, color=colorant"#3f9778", lw=3, label="Buy price")

for i in 1:4
    plot!(viewpred(qfSELL, eachindex(qfBUY), 5-i), lw = 0, fillrange = viewpred(qfSELL, eachindex(qfBUY), 5+i), fillalpha = 0.1, color = colorant"#FE4365", label = nothing)
    plot!(viewpred(qfBUY, eachindex(qfBUY), 5-i), lw = 0, fillrange = viewpred(qfBUY, eachindex(qfBUY), 5+i), fillalpha = 0.1, color = colorant"#3f9778", label = nothing)
end

plot!(viewobs(qfSELL), color = colorant"#FE4365", st=:scatter, markerstrokewidth=0, label=nothing)
plot!(viewobs(qfBUY), color = colorant"#3f9778", st=:scatter, markerstrokewidth=0, label=nothing)
