@testset "IDR" begin
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
