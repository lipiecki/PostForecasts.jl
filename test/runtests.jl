using Test, PostForecasts, Statistics

include(joinpath("tests", "forecasts_tests.jl"))
include(joinpath("tests", "normal_tests.jl"))
include(joinpath("tests", "cp_tests.jl"))
include(joinpath("tests", "idr_tests.jl"))
include(joinpath("tests", "qr_tests.jl"))
include(joinpath("tests", "averaging_tests.jl"))
include(joinpath("tests", "evaluation_tests.jl"))
include(joinpath("tests", "shapley_tests.jl"))
include(joinpath("tests", "loadsave_tests.jl"))
include(joinpath("tests", "error_tests.jl"))
