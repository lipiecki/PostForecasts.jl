using PostForecasts, Plots

fsBUY = loaddata(Symbol(:epex, 4))      # buy at 3am
fsSELL = loaddata(Symbol(:epex, 20))    # sell at 7pm

qfBUY = point2quant(fsBUY, method=:idr, window=182, quantiles=9, start=20230408, stop=20230421)
qfSELL = point2quant(fsSELL, method=:idr, window=182, quantiles=9, start=20230408, stop=20230421)

plot(legend=:bottom, xlabel="Days", ylabel="Price (€/MWh)", xticks=1:14, framestyle=:box) 
# plot forecasts of the median price
plot!(viewpred(qfBUY, eachindex(qfBUY), 5), linealpha=0.5, color=theme_palette(:dark)[3], lw=3, label="Buy price")
plot!(viewpred(qfSELL, eachindex(qfBUY), 5), linealpha=0.5, color=theme_palette(:dark)[1], lw=3, label="Sell price")
# plot prediction intervals constructed from quantiles forecasts
for i in 1:4
    plot!(viewpred(qfBUY, eachindex(qfBUY), 5-i), lw=0, fillrange=viewpred(qfBUY, eachindex(qfBUY), 5+i), fillalpha=0.1, color=theme_palette(:dark)[3], label=nothing)
    plot!(viewpred(qfSELL, eachindex(qfBUY), 5-i), lw=0, fillrange=viewpred(qfSELL, eachindex(qfBUY), 5+i), fillalpha=0.1, color=theme_palette(:dark)[1], label=nothing)    
end
# plot observed prices
plot!(viewobs(qfBUY), color=theme_palette(:dark)[3], st=:scatter, markerstrokewidth=0, label=nothing)
plot!(viewobs(qfSELL), color=theme_palette(:dark)[1], st=:scatter, markerstrokewidth=0, label=nothing)
