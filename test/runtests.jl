using Test, PostForecasts

@testset "Forecasts" begin
    pred = rand(100)
    obs = rand(100)
    pf = PointForecasts(pred, obs)
    qf = QuantForecasts(pred, obs, 0.5)

    @test length(pf) == 100 && length(qf) == 100

    @test all(viewpred(qf) .≈ getpred(qf)) &&
          all(viewobs(qf) .≈ getobs(qf)) &&
          all(viewid(qf) .≈ getid(qf)) &&
          all(viewprob(qf) .≈ getprob(qf))

    @test all(viewpred(pf[begin:3:end]) .≈ viewpred(pf, 1:3:100)) &&
          all(viewpred(qf[begin:3:end]) .≈ viewpred(qf, 1:3:100)) &&
          all(viewobs(pf, 1:10) .≈ obs[1:10]) &&
          all(viewid(qf, [1, 10, 50, 100]) .≈ [1, 10, 50, 100]) &&
          all(viewprob(qf, 1) .≈ 0.5)
          

    @test getpred(pf, 100) ≈ pf[end][:pred] && 
          all(getpred(qf, 100) .≈ qf[end][:pred])
          getobs(pf, 100) ≈ pf[end][:obs] && 
          getobs(qf, 1) ≈ qf[begin][:obs] &&
          getid(pf, 1) ≈ qf[begin][:id] && 
          getid(qf, 1) ≈ qf[begin][:id] &&


    @test pf(10)[:obs] == obs[10] && all(viewobs(qf([50, 100])) .≈ @view(obs[[50, 100]]))
end

@testset "Conformalize" begin
    quantiles = sort(rand(150, 99), dims=2)
    obs = rand(150)
    qf = QuantForecasts(quantiles, obs)
    qfc = conformalize(qf, 100)
    conformalize!(qf, 100)
    qf = qf[101:end]
    @test all(viewpred(qfc) .≈ viewpred(qf)) && all(viewobs(qfc) .≈ viewobs(qf)) && all(viewid(qfc) .≈ viewid(qf))
    testvar = true
    refquantiles = zeros(99)
    for i in 1:50
        for j in 1:99
             refquantiles[j] = quantiles[100+i, j] + quantile(obs[i:100+i-1] - quantiles[i:100+i-1, j], 1 - j/100, alpha=1, beta=1)
        end
        sort!(refquantiles)
        testvar = testvar && all(viewpred(qf, i) .≈ refquantiles)
    end    
    @test testvar
end

@testset "Conformal Prediction" begin
    pred = rand(100)
    obs = rand(100)
    λ = sort(abs.(obs - pred))

    model = CP(100)
    train(model, pred, obs)
    
    @test all(λ .≈ getscores(model))

    pred_ = [pred; rand(50)]
    obs_ = [obs; rand(50)]
    pf = PointForecasts(pred_, obs_)
    qf = point2prob(pf, 100, :cp, 99, recalibration=0)

    corrections = [-quantile(λ, 0.98:-0.02:0.02, sorted=true, alpha=1, beta=1); 0; quantile(λ, 0.02:0.02:0.98, sorted=true, alpha=1, beta=1)]
    quantiles = zeros(50, 99)
    quantiles2 = zeros(50, 99)
    quantiles3 = zeros(50, 99)
    quantiles4 = zeros(50, 99)
    medians = zeros(50)
    medians2 = zeros(50)
    for i in 1:50
        predict!(model, @view(quantiles[i, :]), pred_[100+i], 0.01:0.01:0.99)
        quantiles2[i, :] = predict(model, pred_[100+i], 0.01:0.01:0.99)
        quantiles3[i, :] = predict(model, [pred_[100+i]], 0.01:0.01:0.99)
        quantiles4[i, :] = pred_[100 + i] .+ corrections
        medians[i] = predict(model, pred_[100+i], 0.5)
        medians2[i] = predict(model, [pred_[100+i]], 0.5)
    end
    @test all(quantiles .≈ viewpred(qf))
    @test all(quantiles .≈ quantiles2)
    @test all(quantiles .≈ quantiles3)
    @test all(quantiles .≈ quantiles4)
    @test all(@views(quantiles[:, 50]) .≈ medians)
    @test all(medians .≈ medians2)
end

@testset "Isotonic Distributional Regression" begin
    pred = round.(rand(100), digits = 1)
    obs = round.(rand(100), digits = 1)
    
    model = IDR(length(pred), 1)
    train(model, pred, obs)

    quantiles = zeros(9)
    quantiles2 = zeros(9)
    predict!(model, quantiles, 1.0, 0.1:0.1:0.9)
    predict!(model, quantiles2, 1.1, 0.1:0.1:0.9)
    @test all(quantiles .≈ quantiles2)
    predict!(model, quantiles, 0.0, 0.1:0.1:0.9)
    predict!(model, quantiles2, -0.1, 0.1:0.1:0.9)
    @test all(quantiles .≈ quantiles2)
    @test all(getx(model) .≈ unique(sort(pred)))
    @test all(gety(model) .≈ unique(sort(obs)))

    pred = 0.01:0.01:1.0
    obs = 0.5*pred
    
    model = IDR(length(pred), 1)
    train(model, pred, obs)

    F = getcdf(model)
    x = getx(model)
    y = gety(model)

    @test all(F[1, i] ≈ 1.0 for i in 2:100) && 
          all(F[end, i] ≈ 0.0 for i in 1:99) &&
          all(x[i] ≈ i/100 for i in 1:100) &&
          all(y[i] ≈ i/200 for i in 1:100)

    pred_ = [pred; randn(50)]
    obs_ = [obs; rand(50)]
    pf = PointForecasts(pred_, obs_)
    qf = point2prob(pf, 100, :idr, 99, recalibration=0)

    quantiles = zeros(50, 99)
    quantiles2 = zeros(50, 99)
    quantiles3 = zeros(50, 99)
    medians = zeros(50)
    medians2 = zeros(50)

    for i in 1:50
        predict!(model, @view(quantiles[i, :]), pred_[100+i], 0.01:0.01:0.99)
        quantiles2[i, :] .= predict(model, pred_[100+i], 0.01:0.01:0.99)
        quantiles3[i, :] .= predict(model, [pred_[100+i]], 0.01:0.01:0.99)
        medians[i] = predict(model, pred_[100+i], 0.01:0.01:0.99)[50]
        medians2[i] = predict(model, pred_[100+i], 0.5)
    end

    @test all(quantiles .≈ viewpred(qf))
    @test all(quantiles .≈ quantiles2)
    @test all(quantiles2 .≈ quantiles3)
    @test all(@view(quantiles[:, 50]) .≈ medians)
    @test all(medians .≈ medians2)

    train(model, pred, -obs)
    cdf = getcdf(model)
    x = getx(model)
    y = gety(model)
    @test all(cdf[j, i] ≈ cdf[1, i] for i in 1:100 for j in 2:100) && 
          all(x[i] ≈ i/100 for i in 1:100) &&
          all(y[i] ≈ -(101-i)/200 for i in 1:100)
end

@testset "Quantile Regression Averaging" begin
    pred = rand(100)
    obs = pred.*2 .+ 0.5
    W = [2. 2.; 0.5 0.5]

    model = QR(100, 1, [0.25, 0.75])
    train(model, pred, obs)
    prob = 0.5.*ones(2) # change this
    @test all(getweights(model) .≈ W)
    @test all(predict(model, -1, prob) .≈ [-1.5, -1.5])
    @test prob[1] ≈ 0.25 && prob[2] ≈ 0.75
    
    pred_ = [pred; rand(50)]
    obs_ = [obs; rand(50)]
    pf = PointForecasts(pred_, obs_)
    qf = point2prob(pf, 100, :qr, 0.5, recalibration=0)

    quantiles = zeros(50,2)
    quantiles2 = zeros(50,2)
    for i in 1:50
        predict!(model, @view(quantiles[i, :]), pred_[100+i], prob)
        quantiles2[i, :] = predict(model, pred_[100+i], prob)
    end
    @test all(quantiles .≈ viewpred(qf))
    @test all(quantiles .≈ quantiles2)

    pred = rand(100, 2)
    obs = pred*[2, 1] .+ 0.5
    W = [2.; 1.; 0.5]

    model = QR(size(pred)..., 0.5)
    train(model, pred, obs)

    @test all(getweights(model) .≈ W)
    @test predict(model, [-1, 1], 0.5) ≈ -0.5 
end

@testset "Normal Model" begin
    pred = rand(100)
    obs = pred + 0.1.*randn(100)

    model = Normal()
    train(model, pred, obs)

    @test all(getmean(model) ≈ sum(obs - pred)/100)
    @test all(getstd(model) ≈ sqrt(sum((obs - pred .- getmean(model)).^2)/(99)))

    pred_ = [pred; rand(50)]
    obs_ = [obs; rand(50)]
    pf = PointForecasts(pred_, obs_)
    qf = point2prob(pf, 100, :normal, 99, recalibration=0)

    quantiles = zeros(50, 99)
    quantiles2 = zeros(50, 99)
    for i in 1:50
        predict!(model, @view(quantiles[i, :]), pred_[100+i], 0.01:0.01:0.99)
        quantiles2[i, :] = predict(model,  pred_[100+i], 0.01:0.01:0.99)
    end
    @test all(quantiles .≈ viewpred(qf))
    @test all(quantiles .≈ quantiles2)
end

@testset "Forecast Averaging" begin    
    pred = rand(100, 3)
    sort!(pred, dims=2)
    obs = rand(100)
    pf = PointForecasts(pred, obs, 1:100)
    
    pfm = average(pf)
    pfm2 = average(decouple(pf))
    @test all(viewpred(pfm) .≈ mean(pred, dims = 2)) && all(viewpred(pfm) .≈ viewpred(pfm2)) 

    pfm = average(pf, agg=:median)
    pfm2 = average(decouple(pf), agg=:median)
    @test all(viewpred(pfm) .≈ @view(pred[:, 2])) && all(viewpred(pfm) .≈ viewpred(pfm2)) 

    quantiles1 = [-ones(100) zeros(100) ones(100)]
    quantiles2 = [-ones(100)./2 zeros(100) ones(100)./2]

    qf1 = QuantForecasts(quantiles1, obs, [0.25, 0.5, 0.75])
    qf2 = QuantForecasts(quantiles2, obs, [0.25, 0.5, 0.75])
    qfp = paverage([qf1, qf2], [0.125, 0.25, 0.5, 0.625, 0.75])
    qfpmedian = paverage([qf1, qf2], 0.5)
    qfpmedian2 = paverage([qf1, qf2], 1)
    
    quantiles = [-ones(100) -ones(100)./2 zeros(100) ones(100)./2 ones(100)]
    @test all(viewpred(qfp) .≈ quantiles) && all(viewpred(qfp, eachindex(qfp), 3) .≈ viewpred(qfpmedian)) &&  all(viewpred(qfpmedian) .≈ viewpred(qfpmedian2))

    quantiles1 += rand(100, 3)
    quantiles2 += rand(100, 3)./2
  
    qf1 = QuantForecasts(quantiles1, obs)
    qf2 = QuantForecasts(quantiles2, obs)
    qfq = qaverage([qf1, qf2])
    
    @test all(viewpred(qfq) .≈ (quantiles1 + quantiles2)./2)
end

@testset "Evaluation" begin
    obs = rand(100)
    pred = obs .- 0.1
    pred[1] -= 0.9
    pred2 = obs + [0.1*ones(50); -0.1*ones(50)]
    pred2[1] += 0.9
    pred3 = obs .+ 1.0
    pred3[1] += 9.0
    
    pf = PointForecasts(pred, obs)
    @test mae(pf) ≈ 0.109
    @test rmse(pf) ≈ sqrt(0.0199)

    pf = PointForecasts([pred pred3], obs)
    @test all(mae(pf) .≈ [0.109, 1.09])
    @test all(rmse(pf) .≈ [sqrt(0.0199), sqrt(1.99)])

    qf = QuantForecasts(pred, obs, 0.2)
    @test pinball(qf) ≈ 0.109*0.2

    qf = QuantForecasts(pred2, obs, 0.5)
    @test pinball(qf) ≈ 0.109*0.5

    qf = QuantForecasts(pred3, obs, 0.8)
    @test pinball(qf) ≈ 1.09*0.2

    qf = QuantForecasts([pred pred2 pred3], obs, [0.2, 0.5, 0.8])
    @test all(pinball(qf) .≈ 0.109.*[0.2, 0.5, 0.2*10])
    
    @test all(coverage(qf) .≈ [0.0, 0.5, 1.0])
end

@testset "Data Loading" begin
    pf = loaddata(:epex1)
    pf2 = loaddlmdata(joinpath(@__DIR__, "..", "data", "epex", "epex_hour1.csv"), colnames = true)
    
    @test all(viewpred(pf).≈ viewpred(pf2)) && all(viewobs(pf).≈ viewobs(pf2)) && all(viewid(pf).≈ viewid(pf2))

    pf = loaddata(:pangu0u10)
    pf2 = loaddlmdata(joinpath(@__DIR__, "..", "data", "pangu", "pangu_lead0.csv"), predcol = 2, obscol = 7, colnames = true)
    
    @test all(viewpred(pf).≈ viewpred(pf2)) && all(viewobs(pf).≈ viewobs(pf2)) && all(viewid(pf).≈ viewid(pf2))
    
    save(pf, joinpath(@__DIR__, "test"))
    pf2 = loadpointf(joinpath(@__DIR__, "test.pointf"))
    
    @test all(viewpred(pf).≈ viewpred(pf2)) && all(viewobs(pf).≈ viewobs(pf2)) && all(viewid(pf).≈ viewid(pf2))
    
    pred = rand(100)
    qf = QuantForecasts([pred pred.+0.1 pred.+0.2 pred.+0.3], rand(100), [0.2, 0.4, 0.6, 0.8])
    save(qf, joinpath(@__DIR__, "test"))
    qf2 = loadquantf(joinpath(@__DIR__, "test.quantf"))
    rm(joinpath(@__DIR__, "test.pointf"))
    rm(joinpath(@__DIR__, "test.quantf"))

    @test all(viewpred(qf) .≈ viewpred(qf2)) && all(viewobs(qf) .≈ viewobs(qf2)) && all(viewid(qf) .≈ viewid(qf2)) && all(viewprob(qf) .≈ viewprob(qf2))
end

@testset "Error Handling" verbose=true begin 
    @testset "Unkown Model Name" begin
        pf = PointForecasts(rand(100), rand(100))    
        testvar = false
        try 
            point2prob(pf, 10, :unknown, 99)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "provided model name not recognized"
        end
        @test testvar
    end
    @testset "Unknown Averaging Scheme" begin
        pf = PointForecasts(zeros(2, 2), zeros(2))
        testvar = false
        try 
            average(pf, agg=:unknown)
        catch e
            testvar = isa(e, ArgumentError)
        end
        try 
            average([pf, pf], agg=:unknown)
        catch e
            testvar = testvar && isa(e, ArgumentError)
        end
        @test testvar
    end
    @testset "Matching PointForecasts" begin
        pf = PointForecasts(zeros(2, 2), zeros(2))
        pf_ = PointForecasts(ones(2, 2), zeros(2)) # should match pf
        testvar = true
        try
            arematching([pf, pf_])
        catch e
            testvar = false
        end
        @test testvar

        testvar = false
        pf_ = PointForecasts(zeros(2, 2), zeros(2), [-1, 2]) # different `id`
        try
            arematching([pf, pf_])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`PointForecasts` identifiers (elements of `id` field) do not match"
        end
        @test testvar

        testvar = false
        pf_ = PointForecasts(zeros(2, 2), ones(2)) # different `obs`
        try
            arematching([pf, pf_])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`PointForecasts` observations (elements of `obs` field) do not match"
        end
        @test testvar

        testvar = false
        pf_ = PointForecasts(zeros(2), zeros(2)) # different size of the forecast pool
        try
            arematching([pf, pf_], checkpred=true)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`PointForecasts` have different sizes of forecast pools"
        end
        @test testvar

        testvar = false
        pf_ = PointForecasts(zeros(3, 2), zeros(3)) # different length of the PointForecasts
        try
            arematching([pf, pf_])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`PointForecasts` have different lengths"
        end
        @test testvar
    end

    @testset "Matching QuantForecasts" begin
        qf = QuantForecasts(zeros(2, 2), zeros(2), [0.25, 0.75])
        qf_ = QuantForecasts(ones(2, 2), zeros(2), [0.25, 0.75]) # should match qf
        testvar = true
        try
            arematching([qf, qf_])
        catch e
            println(e)
            testvar = false
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(2, 2), zeros(2), [-1, 2], [0.25, 0.75]) # different `id`
        try
            arematching([qf, qf_])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`QuantForecasts` identifiers (elements of `id` field) do not match"
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(2, 2), ones(2), [0.25, 0.75]) # different `obs`
        try
            arematching([qf, qf_])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`QuantForecasts` observations (elements of `obs` field) do not match"
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(2), zeros(2), 0.25) # different number of quantiles
        try
            arematching([qf, qf_], checkpred=true)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`QuantForecasts` have different number of quantiles"
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(2, 2), zeros(2), [0.1, 0.9]) # different quantile prob
        try
            arematching([qf, qf_], checkpred=true)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`QuantForecasts` have different quantile levels"
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(3, 2), ones(3), [0.25, 0.75]) # different length of the QuantForecasts
        try
            arematching([qf, qf_])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`QuantForecasts` have different lengths"
        end
        @test testvar
    end

    @testset "Creating PointForecasts" begin
        testvar = true
        try
            PointForecasts(zeros(2, 2), zeros(2))
        catch e
            testvar = false
        end
        @test testvar

        testvar = false
        try
            PointForecasts(zeros(2, 2), zeros(3))
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "size of `pred` is $((2, 2)) while length of `obs` is $(3)"
        end
        @test testvar

        testvar = false
        try
            PointForecasts(zeros(2, 2), zeros(2), [1, 2, 3])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "size of `pred` is $((2, 2)) while length of `id` is $(3)"
        end
        @test testvar

        testvar = false
        try
            PointForecasts(zeros(2, 2), zeros(2), [1, 1])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`id` must contain only unique elements"
        end
        @test testvar
    end

    @testset "Creating QuantForecasts" begin
        testvar = true
        try
            QuantForecasts(zeros(2, 2), zeros(2), [0.25, 0.75])
        catch e
            testvar = false
        end
        @test testvar

        testvar = false
        try
            QuantForecasts(zeros(2, 2), zeros(3), [0.25, 0.75])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "size of `pred` is $((2, 2)) while length of `obs` is $(3)"
        end
        @test testvar

        testvar = false
        try
            QuantForecasts(zeros(2, 2), zeros(2), [1, 2, 3], [0.25, 0.75])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "size of `pred` is $((2, 2)) while length of `id` is $(3)"
        end
        @test testvar

        testvar = false
        try
            QuantForecasts(zeros(2, 2), zeros(2), [0.1])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "size of `pred` is $((2, 2)) while length of `prob` is $(1)"
        end
        @test testvar

        testvar = false
        try
            QuantForecasts(zeros(2, 2), zeros(2), [0.9, 0.1])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`prob` vector has to be sorted"
        end
        @test testvar

        testvar = false
        try
            QuantForecasts(zeros(2, 2), zeros(2), [0.0, 1.0])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "elements of `prob` must belong to an open (0, 1) interval"
        end
        @test testvar
        
        testvar = false
        try
            QuantForecasts(zeros(2, 2), zeros(2), [1, 1], [0.25, 0.75])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`id` must contain only unique elements"
        end
        @test testvar

        testvar = false
        try
            QuantForecasts([0.0 0.0; 0.0 -1.0], zeros(2), [0.25, 0.75])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "quantile `pred` passed to the constructor are decreasing"
        end
        @test testvar
    end
    @testset "Loading Datasets" begin
        testvar = false
        try 
            loaddata(:unknown)
        catch e 
            testvar = isa(e, ArgumentError) && e.msg == "unknown is not a valid dataset name"
        end
        @test testvar
    end
end
