using Documenter, PostForecasts

makedocs(
    sitename="PostForecasts.jl",
    format = Documenter.HTML(
        repolink = "https://github.com/lipiecki/PostForecasts.jl",
        inventory_version = ""),
    pages = [
        "Home" => "index.md",
        "Structures" => "forecasts.md",
        "Postprocessing" => "postprocess.md",
        "Models" => "models.md",
        "Loading and saving forecasts" => "loadsave.md",
        "Averaging forecasts" => "averaging.md",
        "Evaluation metrics" => "evaluation.md",
        "Shapley values and ensemble contributions" => "shapley.md",
        "Utilities" => "utils.md",
        "Datasets" => "datasets.md",
        "Examples" => "examples.md"])
deploydocs(
    repo = "github.com/lipiecki/PostForecasts.jl.git",
    branch = "gh-pages")
