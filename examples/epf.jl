using PostForecasts, Statistics

year = 2020
window = 182
models = [:idr, :cp, :qr]

losses = Dict((model => zeros(24) for model in models)...)

Threads.@threads for hour in 0:23
    fs = loaddata(Symbol(:epex, hour))
    first = findindex(fs, year*10_000 + 0101)
    last = findindex(fs, year*10_000 + 1231)
    for model in models
        losses[model][hour+1] = mean(pinball(point2quant(fs, model, window, 9, first=first, last=last)))
    end
end

println("Year $(year)")
println("Calibration window of $(window) days")
println("-"^30)
println("Model\t| Average Pinball Loss")
println("-"^30)
for model in models
    println(uppercase(string(model)), "\t|", round(mean(losses[model]), digits=3))
end
println("-"^30)
