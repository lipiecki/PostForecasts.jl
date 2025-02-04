using Test, PostForecasts, Statistics

@testset "Forecasts" begin
    pred = rand(100)
    obs = rand(100)
    id = Vector(5:5:500)
    pf = PointForecasts(pred, obs, id)
    qf = QuantForecasts(pred, obs, id)

    @test length(pf) == 100 && length(qf) == 100

    @test viewpred(qf) ≈ getpred(qf) &&
        viewobs(qf) ≈ getobs(qf) &&
        viewid(qf) ≈ getid(qf) &&
        viewprob(qf) ≈ getprob(qf)

    I = 1:11:100
    @test viewpred(pf, I) ≈ @view(pred[I]) &&
        viewobs(qf, I) ≈ @view(obs[I]) &&
        viewid(pf, I) ≈ @view(id[I]) &&
        viewprob(qf, 1) ≈ [0.5]
          
    @test getpred(pf, 100) ≈ pf[end][:pred] &&
        getobs(qf, 100) ≈ qf[end][:obs] &&
        getid(pf, 1) ≈ pf[begin][:id] &&
        getprob(qf) ≈ qf[begin][:prob]

    
    @test pf(5)[:obs] ≈ obs[begin] &&
        viewobs(qf(id[I])) ≈ @view(obs[I]) &&
        viewobs(pf(id[1], id[10])) ≈ @view(obs[1:10]) &&
        viewobs(pf[I]) ≈ @view(obs[I])
    
    io = IOBuffer()
    show(io, pf)
    show(io, qf)
    @test String(take!(io)) == "PointForecasts{Float64, Int64} with a pool of 1 forecast(s) at 100 timesteps, between 5 and 500\nQuantForecasts{Float64, Int64} with a pool of 1 forecast(s) at 100 timesteps, between 5 and 500\n"
end

@testset "Normal error distribution" begin
    pred = rand(100)
    obs = pred + randn(100)

    model = Normal()
    train(model, pred, obs)

    @test all(getmean(model) ≈ sum(obs - pred)/100)
    @test all(getstd(model) ≈ sqrt(sum((obs - pred .- getmean(model)).^2)/(99)))

    pred_ = [pred; rand(50)]
    obs_ = [obs; rand(50)]
    pf = PointForecasts(pred_, obs_)
    qf = point2quant(pf, method=:normal, window=100, quantiles=99, retrain=0)

    quantiles = Matrix{Float64}(undef, 50, 99)
    quantiles2 = similar(quantiles)
    quantiles3 = similar(quantiles)
    quantiles4 = similar(quantiles)
    medians = Vector{Float64}(undef, 50)
    medians2 = similar(medians)

    for i in 1:50
        predict!(model, @view(quantiles[i, :]), pred_[100+i], 0.01:0.01:0.99)
        predict!(model, @view(quantiles2[i, :]), [pred_[100+i]], 0.01:0.01:0.99)
        quantiles3[i, :] = predict(model, pred_[100+i], 0.01:0.01:0.99)
        quantiles4[i, :] = predict(model, [pred_[100+i]], 0.01:0.01:0.99)
        medians[i] = predict(model, pred_[100+i], 0.5)
        medians2[i] = predict(model, [pred_[100+i]], 0.5)
    end

    @test quantiles ≈ viewpred(qf)
    @test quantiles ≈ quantiles2
    @test quantiles ≈ quantiles3
    @test quantiles ≈ quantiles4
    @test @views(quantiles[:, 50]) ≈ medians
    @test medians ≈ medians2

    model = Normal(zeromean=true)
    train(model, reshape(pred, 100, 1), obs) # reshape to test `train` method for matrices

    @test all(getmean(model) ≈ 0.0)
    @test all(getstd(model) ≈ sqrt(sum((obs - pred).^2)/100))

    pred_ = [pred; rand(50)]
    obs_ = [obs; rand(50)]
    pf = PointForecasts(pred_, obs_)
    qf = point2quant(pf, method=:zeronormal, window=100, quantiles=99, retrain=0)

    for i in 1:50
        predict!(model, @view(quantiles[i, :]), pred_[100+i], 0.01:0.01:0.99)
        predict!(model, @view(quantiles2[i, :]), [pred_[100+i]], 0.01:0.01:0.99)
        quantiles3[i, :] = predict(model, pred_[100+i], 0.01:0.01:0.99)
        quantiles4[i, :] = predict(model, [pred_[100+i]], 0.01:0.01:0.99)
        medians[i] = predict(model, pred_[100+i], 0.5)
        medians2[i] = predict(model, [pred_[100+i]], 0.5)
    end

    @test quantiles ≈ viewpred(qf)
    @test quantiles ≈ quantiles2
    @test quantiles ≈ quantiles3
    @test quantiles ≈ quantiles4
    @test @views(quantiles[:, 50]) ≈ medians
    @test medians ≈ medians2
end

@testset "Conformal prediction" begin
    pred = rand(100)
    obs = rand(100)
    
    # conformal prediction (absolute errors)
    λ = sort(abs.(obs - pred))
    model = CP(100)
    train(model, pred, obs)
    
    @test λ ≈ getscores(model)

    pred_ = [pred; rand(50)]
    obs_ = [obs; rand(50)]
    pf = PointForecasts(pred_, obs_)
    qf = point2quant(pf, method=:cp, window=100, quantiles=99, retrain=0)
    corrections = [-quantile(λ, 0.98:-0.02:0.02, sorted=true, alpha=1, beta=1); 0.0; quantile(λ, 0.02:0.02:0.98, sorted=true, alpha=1, beta=1)]
    
    quantiles = Matrix{Float64}(undef, 50, 99)
    quantiles2 = similar(quantiles)
    quantiles3 = similar(quantiles)
    quantiles4 = similar(quantiles)
    refquantiles = similar(quantiles)
    medians = Vector{Float64}(undef, 50)
    medians2 = similar(medians)
    
    for i in 1:50
        predict!(model, @view(quantiles[i, :]), pred_[100+i], 0.01:0.01:0.99)
        predict!(model, @view(quantiles2[i, :]), [pred_[100+i]], 0.01:0.01:0.99)
        quantiles3[i, :] = predict(model, pred_[100+i], 0.01:0.01:0.99)
        quantiles4[i, :] = predict(model, [pred_[100+i]], 0.01:0.01:0.99)
        refquantiles[i, :] = pred_[100 + i] .+ corrections
        medians[i] = predict(model, pred_[100+i], 0.5)
        medians2[i] = predict(model, [pred_[100+i]], 0.5)
    end

    @test quantiles ≈ viewpred(qf)
    @test quantiles ≈ quantiles2
    @test quantiles ≈ quantiles3
    @test quantiles ≈ quantiles4
    @test quantiles ≈ refquantiles
    @test @views(quantiles[:, 50]) ≈ medians
    @test medians ≈ medians2

    # historical simulation (non-absolute errors)
    λ = sort((obs - pred))
    model = CP(100, abs=false)
    train(model, reshape(pred, 100, 1), obs) # reshape to test `train` method for matrices

    @test λ ≈ getscores(model)

    pf = PointForecasts(pred_, obs_)
    qf = point2quant(pf, method=:hs, window=100, quantiles=99, retrain=0)
    corrections = quantile(λ, 0.01:0.01:0.99, sorted=true, alpha=1, beta=1)

    for i in 1:50
        predict!(model, @view(quantiles[i, :]), pred_[100+i], 0.01:0.01:0.99)
        predict!(model, @view(quantiles2[i, :]), [pred_[100+i]], 0.01:0.01:0.99)
        quantiles3[i, :] = predict(model, pred_[100+i], 0.01:0.01:0.99)
        quantiles4[i, :] = predict(model, [pred_[100+i]], 0.01:0.01:0.99)
        refquantiles[i, :] = pred_[100 + i] .+ corrections
        medians[i] = predict(model, pred_[100+i], 0.5)
        medians2[i] = predict(model, [pred_[100+i]], 0.5)
    end

    @test quantiles ≈ viewpred(qf)
    @test quantiles ≈ quantiles2
    @test quantiles ≈ quantiles3
    @test quantiles ≈ quantiles4
    @test quantiles ≈ refquantiles
    @test @views(quantiles[:, 50]) ≈ medians
    @test medians ≈ medians2
end

@testset "Isotonic distributional regression" begin
    model = IDR(4, 2)
    train(model, [1.5 0.5; 2.0 3.0; 3.5 0.5; 4.5 3.5], [1.0, 2.5, 2.0, 4.0])

    F = getcdf(model, 1)
    @test size(F, 1) == 4 &&
        F[1, :] ≈ [1.0, 1.0, 1.0, 1.0] &&
        F[2, :] ≈ [0.0, 0.5, 1.0, 1.0] &&
        F[3, :] ≈ [0.0, 0.5, 1.0, 1.0] &&
        F[4, :] ≈ [0.0, 0.0, 0.0, 1.0]

    F = getcdf(model, 2)
    @test size(F, 1) == 3 &&
        F[1, :] ≈ [0.5, 1.0, 1.0, 1.0] &&
        F[2, :] ≈ [0.0, 0.0, 1.0, 1.0] &&
        F[3, :] ≈ [0.0, 0.0, 0.0, 1.0]

    pred = round.(rand(100), digits = 1)
    obs = round.(rand(100), digits = 1)

    model = IDR(length(pred), 1)
    train(model, pred, obs)

    @test getx(model) ≈ unique(sort(pred))
    @test gety(model) ≈ unique(sort(obs))

    @test predict(model, maximum(pred), 0.01:0.01:0.99) ≈ predict(model, 1.1, 0.01:0.01:0.99)
    @test predict(model, minimum(pred), 0.01:0.01:0.99) ≈ predict(model, -0.1, 0.01:0.01:0.99)

    pred_ = [pred; randn(50)]
    obs_ = [obs; rand(50)]
    pf = PointForecasts(pred_, obs_)
    qf = point2quant(pf, method=:idr, window=100, quantiles=99, retrain=0)

    quantiles = Matrix{Float64}(undef, 50, 99)
    quantiles2 = similar(quantiles)
    quantiles3 = similar(quantiles)
    quantiles4 = similar(quantiles)
    medians = Vector{Float64}(undef, 50)
    medians2 = similar(medians)

    for i in 1:50
        predict!(model, @view(quantiles[i, :]), pred_[100+i], 0.01:0.01:0.99)
        predict!(model, @view(quantiles2[i, :]), [pred_[100+i]], 0.01:0.01:0.99)
        quantiles3[i, :] .= predict(model, pred_[100+i], 0.01:0.01:0.99)
        quantiles4[i, :] .= predict(model, [pred_[100+i]], 0.01:0.01:0.99)
        medians[i] = predict(model, pred_[100+i], 0.01:0.01:0.99)[50]
        medians2[i] = predict(model, pred_[100+i], 0.5)
    end

    @test quantiles ≈ viewpred(qf)
    @test quantiles ≈ quantiles2
    @test quantiles ≈ quantiles3
    @test quantiles ≈ quantiles4
    @test @views(quantiles[:, 50]) ≈ medians
    @test medians ≈ medians2
end

@testset "Quantile regression" begin
    pred = rand(100)
    obs = pred.*2 .+ 0.5
    W = [2. 2.; 0.5 0.5]

    prob = [0.25, 0.75]
    model = QR(100, 1, prob)
    train(model, pred, obs)
    
    @test getweights(model) ≈ W
    @test getquantprob(model) == [0.25, 0.75]
    
    @test predict(model, -1, prob) ≈ [-1.5, -1.5]
    @test predict(model, [-1], prob) ≈ [-1.5, -1.5]
    @test predict(model, -1) ≈ [-1.5, -1.5]
    @test predict(model, [-1]) ≈ [-1.5, -1.5]
    @test predict(model, -1, 0.25) ≈ -1.5
    @test predict(model, [-1], 0.25) ≈ -1.5

    pred_ = [pred; rand(50)]
    obs_ = [obs; rand(50)]
    pf = PointForecasts(pred_, obs_)
    qf = point2quant(pf, method=:qr, window=100, quantiles=[0.25, 0.75], retrain=0)

    quantiles = Matrix{Float64}(undef, 50, 2)
    quantiles2 = similar(quantiles)
    quantiles3 = similar(quantiles)

    for i in 1:50
        predict!(model, @view(quantiles[i, :]), pred_[100+i], prob)
        predict!(model, @view(quantiles2[i, :]), pred_[100+i])
        predict!(model, @view(quantiles3[i, :]), [pred_[100+i]], prob)
    end

    @test quantiles ≈ viewpred(qf)
    @test quantiles ≈ quantiles2
    @test quantiles ≈ quantiles3

    pred = rand(100, 2)
    obs = pred*[2, 1] .+ 0.5
    W = [2.; 1.; 0.5]

    model = QR(size(pred)..., 0.5)
    train(model, pred, obs)

    pred_ = [pred; rand(50, 2)]
    obs_ = [obs; rand(50)]
    pf = PointForecasts(pred_, obs_)
    qf = point2quant(pf, method=:qr, window=100, quantiles=0.5, retrain=0)

    @test getweights(model) ≈ W
    @test getquantprob(model) == [0.5]

    median = Matrix{Float64}(undef, 50, 1)

    for i in 1:50
        predict!(model, @view(median[i, :]), pred_[100+i, :])
    end

    @test median ≈ viewpred(qf)
end

@testset "Conformalization" begin
    pred = sort(rand(150, 99), dims=2)
    obs = rand(150)
    qf = QuantForecasts(pred, obs)
    qfc = conformalize(qf, window=100)
    conformalize!(qf, window=100)
    qf = qf[101:end]
    
    @test all(viewpred(qfc) .≈ viewpred(qf)) && all(viewobs(qfc) .≈ viewobs(qf)) && all(viewid(qfc) .≈ viewid(qf))
    
    refquantiles = Matrix{Float64}(undef, 50, 99)
    for i in 1:50
        for j in 1:99
             refquantiles[i, j] = pred[100+i, j] + quantile(obs[i:100+i-1] - pred[i:100+i-1, j], 1 - j/100, alpha=1, beta=1)
        end
    end    
    
    sort!(refquantiles, dims=2)
    @test viewpred(qf) ≈ refquantiles
end

@testset "Forecast averaging" begin    
    pred = rand(100, 3)
    sort!(pred, dims=2)
    obs = rand(100)
    pf = PointForecasts(pred, obs, Vector(1:100))
    
    pfm = average(pf)
    pfm2 = average(decouple(pf))
    @test viewpred(pfm) ≈ mean(pred, dims = 2) && viewpred(pfm) ≈ viewpred(pfm2) 

    pfm = average(pf, agg=:median)
    pfm2 = average(decouple(pf), agg=:median)
    @test viewpred(pfm) ≈ @view(pred[:, 2]) && viewpred(pfm) ≈ viewpred(pfm2) 

    qf1 = QuantForecasts( [-ones(100) zeros(100) ones(100)], obs, [0.25, 0.5, 0.75])
    qf2 = QuantForecasts([-ones(100)./2 zeros(100) ones(100)./2], obs, [0.25, 0.5, 0.75])
    qfp = paverage([qf1, qf2], quantiles=[0.125, 0.25, 0.5, 0.625, 0.75])
    qfq = qaverage([qf1, qf2])
    qfpmedian = paverage([qf1, qf2], quantiles=0.5)
    qfpmedian2 = paverage([qf1, qf2], quantiles=1) # integer 1 means that one quantile (median) will be calculated
    
    quantiles = [-ones(100) -ones(100)./2 zeros(100) ones(100)./2 ones(100)]
    @test viewpred(qfp) ≈ quantiles
    @test viewpred(qfp, eachindex(qfp), 3) ≈ viewpred(qfpmedian)
    @test viewpred(qfpmedian) ≈ viewpred(qfpmedian2)
    @test viewpred(qfq) ≈ [-ones(100).*0.75 zeros(100) ones(100).*0.75]
end

@testset "Evaluation" begin
    obs = rand(100)
    pred = obs .- 0.1
    pred[1] -= 0.9
    pred2 = obs + [0.1*ones(50); -0.1*ones(50)]
    pred2[1] += 0.9
    pred3 = obs .+ 1.0
    pred3[1] += 9.0
    
    pf = PointForecasts([pred pred3], obs)
    @test mae(pf) ≈ [0.109, 1.09]
    @test rmse(pf) ≈ [sqrt(0.0199), sqrt(1.99)]

    qf = QuantForecasts([pred pred2 pred3], obs, [0.2, 0.5, 0.8])
    @test pinball(qf) ≈ 0.109.*[0.2, 0.5, 0.2*10]
    @test crps(qf) ≈ 2*mean(pinball(qf))
    @test coverage(qf) ≈ [0.0, 0.5, 1.0]
end

@testset "Shapley values" begin
    obs = rand(100)
    pred = copy(obs)
    err = randn(100)
    pred2 = obs + err
    loss = mean(abs.(err))
    pf = PointForecasts(pred, obs)
    pf2 = PointForecasts(pred2, obs)
    shapvals = shapley([pf, pf2], average, (x -> -mae(x)[1]))
    shapvals2 = shapley([PointForecasts(pred, obs), PointForecasts(pred2, obs)], average, (x -> -mae(x)[1]), 0.0)
    @test shapvals[1] ≈ loss/4 && shapvals[2] ≈ -loss/4
    @test (shapvals2[1] ≈ shapvals[1] - mae(pf)[1]/2) && (shapvals2[2] ≈ shapvals[2] - mae(pf2)[1]/2)
end

@testset "Data loading" begin
    pf = loaddata(:epex1)
    pf2 = loaddlmdata(joinpath(@__DIR__, "..", "data", "epex", "epex_hour1.csv"), colnames = true)
    
    @test viewpred(pf) ≈ viewpred(pf2) && viewobs(pf) ≈ viewobs(pf2) && viewid(pf) ≈ viewid(pf2)

    pf = loaddata(:pangu0u10)
    pf2 = loaddlmdata(joinpath(@__DIR__, "..", "data", "pangu", "pangu_lead0.csv"), predcol = 2, obscol = 7, colnames = true)
    
    @test viewpred(pf) ≈ viewpred(pf2) && viewobs(pf) ≈ viewobs(pf2) && viewid(pf) ≈ viewid(pf2)
    
    saveforecasts(pf, joinpath(@__DIR__, "test"))
    pf2 = loadforecasts(joinpath(@__DIR__, "test.pointf"))
    
    @test viewpred(pf) ≈ viewpred(pf2) && viewobs(pf) ≈ viewobs(pf2) && viewid(pf) ≈ viewid(pf2)

    pf2 = loadpointf(joinpath(@__DIR__, "test"))
    rm(joinpath(@__DIR__, "test.pointf"))
    
    @test viewpred(pf) ≈ viewpred(pf2) && viewobs(pf) ≈ viewobs(pf2) && viewid(pf) ≈ viewid(pf2)
    
    pred = rand(100)
    qf = QuantForecasts([pred pred.+0.1 pred.+0.2 pred.+0.3], rand(100), [0.2, 0.4, 0.6, 0.8])
    saveforecasts(qf, joinpath(@__DIR__, "test"))
    qf2 = loadforecasts(joinpath(@__DIR__, "test.quantf"))

    @test viewpred(qf) ≈ viewpred(qf2) && viewobs(qf) ≈ viewobs(qf2) && viewid(qf) ≈ viewid(qf2) && viewprob(qf) ≈ viewprob(qf2)
    
    qf2 = loadquantf(joinpath(@__DIR__, "test"))
    rm(joinpath(@__DIR__, "test.quantf"))

    @test viewpred(qf) ≈ viewpred(qf2) && viewobs(qf) ≈ viewobs(qf2) && viewid(qf) ≈ viewid(qf2) && viewprob(qf) ≈ viewprob(qf2)
end

@testset "Error handling" verbose=true begin 
    @testset "Unknown arguments" begin
        pf = PointForecasts(rand(100), rand(100))    
        testvar = false
        try 
            point2quant(pf, method=:unknown, window=10, quantiles=99)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "provided model name not recognized"
        end
        @test testvar

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

        testvar = false
        try 
            loaddata(:unknown)
        catch e 
            testvar = isa(e, ArgumentError) && e.msg == "unknown is not a valid dataset name"
        end
        @test testvar

        testvar = false
        try 
            loadforecasts("filename")
        catch e 
            testvar = isa(e, ArgumentError) && e.msg == "`.pointf` or `.quantf` extension required"
        end
        @test testvar
    end

    @testset "Matching PointForecasts" begin
        pf = PointForecasts(zeros(2, 2), zeros(2))
        pf_ = PointForecasts(ones(2, 2), zeros(2)) # should match pf
        testvar = true
        try
            checkmatch([pf, pf_])
        catch e
            testvar = false
        end
        @test testvar

        testvar = false
        pf_ = PointForecasts(zeros(2, 2), zeros(2), [-1, 2]) # different `id`
        try
            checkmatch([pf, pf_])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`PointForecasts` identifiers (elements of `id` field) do not match"
        end
        @test testvar

        testvar = false
        pf_ = PointForecasts(zeros(2, 2), ones(2)) # different `obs`
        try
            checkmatch([pf, pf_])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`PointForecasts` observations (elements of `obs` field) do not match"
        end
        @test testvar

        testvar = false
        pf_ = PointForecasts(zeros(2), zeros(2)) # different size of the forecast pool
        try
            checkmatch([pf, pf_], checkpred=true)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`PointForecasts` have different sizes of forecast pools"
        end
        @test testvar

        testvar = false
        pf_ = PointForecasts(zeros(3, 2), zeros(3)) # different length of the PointForecasts
        try
            checkmatch([pf, pf_])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`PointForecasts` have different lengths"
        end
        @test testvar
    end

    @testset "Matching QuantForecasts" begin
        qf = QuantForecasts(zeros(2, 2), zeros(2))
        qf_ = QuantForecasts(ones(2, 2), zeros(2), [1.0/3, 2.0/3]) # should match qf
        testvar = true
        try
            checkmatch([qf, qf_])
        catch e
            println(e)
            testvar = false
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(2, 2), zeros(2), [0, 1]) # different `id`
        try
            checkmatch([qf, qf_])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`QuantForecasts` identifiers (elements of `id` field) do not match"
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(2, 2), ones(2)) # different `obs`
        try
            checkmatch([qf, qf_])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`QuantForecasts` observations (elements of `obs` field) do not match"
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(2), zeros(2), 0.25) # different number of quantiles
        try
            checkmatch([qf, qf_], checkpred=true)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`QuantForecasts` have different number of quantiles"
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(2, 2), zeros(2), [0.1, 0.9]) # different quantile prob
        try
            checkmatch([qf, qf_], checkpred=true)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`QuantForecasts` have different quantile levels"
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(3, 2), ones(3), [1.0/3, 2.0/3]) # different length of the QuantForecasts
        try
            checkmatch([qf, qf_])
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
end
