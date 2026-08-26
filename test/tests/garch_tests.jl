@testset "GARCH" begin
    n = 10_000
    residuals = zeros(n)
    alpha = 0.5
    beta = 0.2
    variance = 2
    omega = variance * (1 - alpha - beta)
    residuals[1] = 1
    
    for i in 2:length(residuals)
        variance = alpha * variance + beta * residuals[i-1]^2 + omega
        residuals[i] = sqrt(variance)
    end

    model = GARCH(n)
    train(model, zeros(n), residuals)
    
    @test sum(abs.(getparams(model) .- (alpha, beta, omega))) < 0.001
    
    new_variance = alpha * variance + beta * residuals[end]^2 + omega
    @test sum(abs.(predict(model, 0, [0.25, 0.5, 0.75]) .- [-0.6745*sqrt(new_variance), 0.0, 0.6745*sqrt(new_variance)])) < 0.001

    advance!(model)
    new_variance = (alpha+beta)*variance + omega
    @test sum(abs.(predict(model, 0, [0.25, 0.5, 0.75]) .- [-0.6745*sqrt(new_variance), 0.0, 0.6745*sqrt(new_variance)])) < 0.001

    model = GARCH(n, filter=true)
    train(model, zeros(n), residuals)
    new_variance = alpha * variance + beta * residuals[end]^2 + omega
    @test all(predict(model, 0, [0.25, 0.5, 0.75]) .≈ sqrt(new_variance))

    model = GARCH(n, filter=true, abs=true)
    train(model, zeros(n), residuals)
    new_variance = alpha * variance + beta * residuals[end]^2 + omega
    @test all(predict(model, 0, [0.25, 0.5, 0.75]) .≈ [-sqrt(new_variance), 0.0, sqrt(new_variance)])
end
