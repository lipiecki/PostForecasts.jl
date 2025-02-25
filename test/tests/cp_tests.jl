@testset "CP" begin
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
