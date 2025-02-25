@testset "Forecast averaging" begin    
    pred = rand(100, 3)
    sort!(pred, dims=2)
    obs = rand(100)
    pf = PointForecasts(pred, obs, Vector(1:100))
    
    pfm = average(pf)
    pfm2 = average(decouple(pf))
    @test viewpred(pfm) ≈ mean(pred, dims = 2) && viewpred(pfm) ≈ viewpred(pfm2) 
    @test viewpred(pfm2) ≈ viewpred(average(decouple(pf)...))

    pfm = average(pf, agg=:median)
    pfm2 = average(decouple(pf), agg=:median)
    @test viewpred(pfm) ≈ @view(pred[:, 2]) && viewpred(pfm) ≈ viewpred(pfm2)
    @test viewpred(pfm2) ≈ viewpred(average(decouple(pf)..., agg=:median))

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
    @test viewpred(qfq) ≈ viewpred(qaverage(qf1, qf2))
    @test viewpred(qfpmedian) ≈ viewpred(paverage(qf1, qf2, quantiles=0.5))
    @test viewpred(paverage(qf1, qf2, quantiles=0.8)) ≈ viewpred(paverage(qf1, qf2, quantiles=0.75))
end
