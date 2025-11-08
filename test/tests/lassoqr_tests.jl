@testset "LassoQR" begin
    
    setlambda([0])
    @test getlambda() ≈ [0]
    
    pred = rand(100, 2)
    obs = pred[:, 1].*0.8 + pred[:, 2].*0.2
    pred .= pred .+ randn(size(pred))./10
    
    prob = [0.1, 0.9]

    qr = QR(100, 2, prob)
    lassoqr = LassoQR(100, 2, prob)

    @test getquantprob(lassoqr) ≈ prob
    
    train(qr, pred, obs)
    train(lassoqr, pred, obs)
    
    W = getweights(lassoqr)

    input = rand(2)
    @test predict(qr, input, prob) ≈ predict(lassoqr, input, prob)
    @test predict(qr, input) ≈ predict(lassoqr, input)
    @test predict(qr, input, 0.1) ≈ predict(lassoqr, input, 0.1)

    pred_ = [pred; rand(50, 2)]
    obs_ = [obs; rand(50)]
    
    pf = PointForecasts(pred_, obs_)
    qf = point2quant(pf, method=:lassoqr, window=100, quantiles=prob, retrain=0)

    quantiles = Matrix{Float64}(undef, 50, 2)
    for i in 1:50
        predict!(lassoqr, @view(quantiles[i, :]), pred_[100+i, :])
    end

    @test quantiles ≈ viewpred(qf)

    setlambda([10])
    lassoqr = LassoQR(100, 2, prob)
    train(lassoqr, pred, obs)
    lassoW = getweights(lassoqr)
    
    @test @views all(abs.(lassoW[1:end-1, :]) .< abs.(W[1:end-1, :]))
end
