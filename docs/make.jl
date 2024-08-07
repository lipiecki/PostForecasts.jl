using Documenter, PostForecasts

makedocs(
    sitename="PostForecasts.jl",
    
    format = Documenter.HTML(
    repolink = "https://github.com/lipiecki/PostForecasts.jl",
    inventory_version = ""),
    pages = [
        "Home" => "index.md",
        "Series Structures" => "forecasts.md",
        "Loading and Saving Forecasts" => "loadnsave.md",
        "Datasets" => "data.md",
        "Models" => "models.md",
        "Averaging Forecasts" => "averaging.md",
        "Evaluation Metrics" => "evaluation.md",
        "Examples" => "examples.md"
        ]
)

   deploydocs(
    repo = "github.com/lipiecki/PostForecasts.jl.git",
    branch = "docs",
)