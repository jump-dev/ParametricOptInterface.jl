using Documenter, ParametricOptInterface

makedocs(
    modules = [ParametricOptInterface],
    doctest = false,
    clean = true,
    # See https://github.com/JuliaDocs/Documenter.jl/issues/868
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        mathengine = Documenter.MathJax(),
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    sitename = "ParametricOptInterface.jl",
    authors = "Tomás Gutierrez, and contributors",
    pages = [
        "Home" => "index.md",
        "manual.md",
        "example.md",
        # "reference.md"
    ],
)

deploydocs(repo = "github.com/jump-dev/ParametricOptInterface.jl.git")
