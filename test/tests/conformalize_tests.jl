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
