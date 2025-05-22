@testset "QR" begin
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
             refquantiles[i, j] = pred[100+i, j] + quantile(obs[i:100+i-1] - pred[i:100+i-1, j], j/100, alpha=1, beta=1)
        end
    end    
    
    sort!(refquantiles, dims=2)
    @test viewpred(qf) ≈ refquantiles
end
