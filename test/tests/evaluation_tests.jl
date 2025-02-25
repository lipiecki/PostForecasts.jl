@testset "Evaluation" begin
    obs = randn(100)
    pred = obs .- 0.1
    pred[1] -= 0.9
    pred2 = obs + [0.1*ones(50); -0.1*ones(50)]
    pred2[1] += 0.9
    pred3 = obs .+ 1.0
    pred3[1] += 9.0

    pf = PointForecasts([0.5 1.5; 1.0 3.0; 2.0 6.0], [1.0, 2.0, 4.0])
    @test mape(pf) ≈ [50.0, 50.0]
    @test smape(pf) ≈ [200/3, 40.0]

    pf = PointForecasts([pred pred3], obs)
    @test mae(pf) ≈ [0.109, 1.09]
    @test mse(pf) ≈ [0.0199, 1.99]

    qf = QuantForecasts([pred pred2 pred3], obs, [0.25, 0.5, 0.75])
    @test pinball(qf) ≈ 0.109.*[0.25, 0.5, 0.25*10]
    @test crps(qf) ≈ 2*mean(pinball(qf))
    @test coverage(qf) ≈ [0.0, 0.5, 1.0]
end
