@testset "Loading and saving" begin
    pf = loaddata("epex1")
    pf2 = loaddlm(joinpath(@__DIR__, "..", "..", "data", "epex", "epex_hour1.csv"), idcol=1, obscol=2, colnames=true)
    
    @test viewpred(pf) ≈ viewpred(pf2) && viewobs(pf) ≈ viewobs(pf2) && viewid(pf) ≈ viewid(pf2)

    pf = loaddata("pangu0u10")
    pf2 = loaddlm(joinpath(@__DIR__, "..", "..", "data", "pangu", "pangu_lead0.csv"), idcol=1, predcol = 2, obscol = 7, colnames=true)
    
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
