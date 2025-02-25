@testset "Error handling" verbose=true begin 
    @testset "Unknown arguments" begin
        pf = PointForecasts(rand(100), rand(100))    
        testvar = false
        try 
            point2quant(pf, method=:unknown, window=10, quantiles=99)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "provided model name not recognized"
        end
        @test testvar

        pf = PointForecasts(zeros(2, 2), zeros(2))
        testvar = false
        try 
            average(pf, agg=:unknown)
        catch e
            testvar = isa(e, ArgumentError)
        end
        try 
            average([pf, pf], agg=:unknown)
        catch e
            testvar = testvar && isa(e, ArgumentError)
        end
        @test testvar

        testvar = false
        try 
            loaddata(:unknown)
        catch e 
            testvar = isa(e, ArgumentError) && e.msg == "unknown is not a valid dataset name"
        end
        @test testvar

        testvar = false
        try 
            loadforecasts("filename")
        catch e 
            testvar = isa(e, ArgumentError) && e.msg == "`.pointf` or `.quantf` extension required"
        end
        @test testvar
    end

    @testset "Matching PointForecasts" begin
        pf = PointForecasts(zeros(2, 2), zeros(2))
        pf_ = PointForecasts(ones(2, 2), zeros(2)) # should match pf
        testvar = true
        try
            checkmatch(pf, pf_)
        catch e
            testvar = false
        end
        @test testvar

        testvar = false
        pf_ = PointForecasts(zeros(2, 2), zeros(2), [-1, 2]) # different `id`
        try
            checkmatch(pf, pf_)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "Forecasts objects have incompatible `id`entifiers"
        end
        @test testvar

        testvar = false
        pf_ = PointForecasts(zeros(2, 2), ones(2)) # different `obs`
        try
            checkmatch(pf, pf_)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "Forecasts objects have incompatible `obs`ervations"
        end
        @test testvar

        testvar = false
        pf_ = PointForecasts(zeros(2), zeros(2)) # different size of the forecast pool
        try
            checkmatch(pf, pf_, checkpred=true)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "Forecasts objects have different sizes of forecast pools"
        end
        @test testvar

        testvar = false
        pf_ = PointForecasts(zeros(3, 2), zeros(3)) # different length of the PointForecasts
        try
            checkmatch(pf, pf_)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "Forecasts objects have different lengths"
        end
        @test testvar
    end

    @testset "Matching QuantForecasts" begin
        qf = QuantForecasts(zeros(2, 2), zeros(2))
        qf_ = QuantForecasts(ones(2, 2), zeros(2), [1.0/3, 2.0/3]) # should match qf
        testvar = true
        try
            checkmatch(qf, qf_)
        catch e
            println(e)
            testvar = false
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(2, 2), zeros(2), [0, 1]) # different `id`
        try
            checkmatch(qf, qf_)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "Forecasts objects have incompatible `id`entifiers"
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(2, 2), ones(2)) # different `obs`
        try
            checkmatch(qf, qf_)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "Forecasts objects have incompatible `obs`ervations"
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(2), zeros(2), 0.25) # different number of quantiles
        try
            checkmatch(qf, qf_, checkpred=true)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "Forecasts objects have different quantile levels"
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(2, 2), zeros(2), [0.1, 0.9]) # different quantile prob
        try
            checkmatch(qf, qf_, checkpred=true)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "Forecasts objects have different quantile levels"
        end
        @test testvar

        testvar = false
        qf_ = QuantForecasts(zeros(3, 2), ones(3), [1.0/3, 2.0/3]) # different length of the QuantForecasts
        try
            checkmatch(qf, qf_)
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "Forecasts objects have different lengths"
        end
        @test testvar
    end

    @testset "Creating PointForecasts" begin
        testvar = true
        try
            PointForecasts(zeros(2, 2), zeros(2))
        catch e
            testvar = false
        end
        @test testvar

        testvar = false
        try
            PointForecasts(zeros(2, 2), zeros(3))
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "size of `pred` is $((2, 2)) while length of `obs` is $(3)"
        end
        @test testvar

        testvar = false
        try
            PointForecasts(zeros(2, 2), zeros(2), [1, 2, 3])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "size of `pred` is $((2, 2)) while length of `id` is $(3)"
        end
        @test testvar

        testvar = false
        try
            PointForecasts(zeros(2, 2), zeros(2), [1, 1])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`id` must contain only unique elements"
        end
        @test testvar
    end

    @testset "Creating QuantForecasts" begin
        testvar = true
        try
            QuantForecasts(zeros(2, 2), zeros(2), [0.25, 0.75])
        catch e
            testvar = false
        end
        @test testvar

        testvar = false
        try
            QuantForecasts(zeros(2, 2), zeros(3), [0.25, 0.75])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "size of `pred` is $((2, 2)) while length of `obs` is $(3)"
        end
        @test testvar

        testvar = false
        try
            QuantForecasts(zeros(2, 2), zeros(2), [1, 2, 3], [0.25, 0.75])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "size of `pred` is $((2, 2)) while length of `id` is $(3)"
        end
        @test testvar

        testvar = false
        try
            QuantForecasts(zeros(2, 2), zeros(2), [0.1])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "size of `pred` is $((2, 2)) while length of `prob` is $(1)"
        end
        @test testvar

        testvar = false
        try
            QuantForecasts(zeros(2, 2), zeros(2), [0.9, 0.1])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`prob` vector has to be sorted"
        end
        @test testvar

        testvar = false
        try
            QuantForecasts(zeros(2, 2), zeros(2), [0.0, 1.0])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "elements of `prob` must belong to an open (0, 1) interval"
        end
        @test testvar
        
        testvar = false
        try
            QuantForecasts(zeros(2, 2), zeros(2), [1, 1], [0.25, 0.75])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "`id` must contain only unique elements"
        end
        @test testvar

        testvar = false
        try
            QuantForecasts([0.0 0.0; 0.0 -1.0], zeros(2), [0.25, 0.75])
        catch e
            testvar = isa(e, ArgumentError) && e.msg == "quantile `pred` passed to the constructor are decreasing"
        end
        @test testvar
    end
end
