using PostForecasts, Plots

firstdate = 20230408
lastdate = 20230421

buyhour = 3
sellhour = 19

fsBUY = loaddata(Symbol(:epex, buyhour))
fsSELL = loaddata(Symbol(:epex, sellhour))

first= findindex(fsBUY, firstdate)
last = findindex(fsBUY, lastdate)

qfBUY = point2prob(fsBUY, 182, :idr, 9, first=first, last=last)
qfSELL = point2prob(fsSELL, 182, :idr, 9, first=first, last=last)

theme(:dark)
plot(legend = :bottom, xlabel = "Days", ylabel = "Price (€/MWh)", xticks = 1:14, framestyle = :box)
plot!(viewpred(qfBUY, eachindex(qfBUY), 5), linealpha = 0.5, color=3, lw=3, label="Buy hour ($(buyhour):00)")
plot!(viewpred(qfSELL, eachindex(qfBUY), 5), linealpha = 0.5, color=1, lw=3, label="Sell hour ($(sellhour):00)")

for i in 1:4
    plot!(viewpred(qfBUY, eachindex(qfBUY), 5-i), lw = 0, fillrange = viewpred(qfBUY, eachindex(qfBUY), 5+i), fillalpha = 0.1, color = 3, label = nothing)
    plot!(viewpred(qfSELL, eachindex(qfBUY), 5-i), lw = 0, fillrange = viewpred(qfSELL, eachindex(qfBUY), 5+i), fillalpha = 0.1, color = 1, label = nothing)    
end

plot!(viewobs(qfBUY), color = 3, st=:scatter, markerstrokewidth=0, label=nothing)
plot!(viewobs(qfSELL), color = 1, st=:scatter, markerstrokewidth=0, label=nothing)
