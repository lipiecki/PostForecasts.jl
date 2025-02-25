@testset "Normal" begin
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
