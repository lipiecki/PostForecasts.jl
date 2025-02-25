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
