@testset "Forecasts" begin
    pred = sort(rand(100, 2), dims=2)
    obs = rand(100)
    id = Vector(5:5:500)
    pf = PointForecasts(pred, obs, id)
    qf = QuantForecasts(pred, obs, id)

    @test length(pf) == 100 && length(qf) == 100

    @test viewpred(qf) ≈ getpred(qf) &&
        viewobs(qf) ≈ getobs(qf) &&
        viewid(qf) ≈ getid(qf) &&
        viewprob(qf) ≈ getprob(qf)

    I = 1:11:100
    @test viewpred(pf, I) ≈ @view(pred[I, :]) &&
        viewobs(qf, I) ≈ @view(obs[I]) &&
        viewid(pf, I) ≈ @view(id[I]) &&
        viewprob(qf, 1) ≈ [1/3]
          
    @test getpred(pf, 100) ≈ pf[end][:pred] &&
        getobs(qf, 100) ≈ qf[end][:obs] &&
        getid(pf, 1) ≈ pf[begin][:id] &&
        getprob(qf) ≈ qf[begin][:prob]

    
    @test pf(5)[:obs] ≈ obs[begin] &&
        viewobs(qf(id[I])) ≈ @view(obs[I]) &&
        viewobs(pf(id[1], id[10])) ≈ @view(obs[1:10]) &&
        viewobs(pf[I]) ≈ @view(obs[I])

    @test couple(decouple(pf)) ≈ pf && couple(decouple(qf)) ≈ qf
    
    io = IOBuffer()
    show(io, pf)
    show(io, qf)
    @test String(take!(io)) == "PointForecasts{Float64, Int64} with a pool of 2 forecast(s) at 100 timesteps, between 5 and 500\nQuantForecasts{Float64, Int64} with a pool of 2 forecast(s) at 100 timesteps, between 5 and 500\n"
end
