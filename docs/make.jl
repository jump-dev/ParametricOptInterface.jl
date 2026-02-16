# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import Documenter
import ParametricOptInterface

Documenter.makedocs(;
    modules = [ParametricOptInterface],
    clean = true,
    # See https://github.com/JuliaDocs/Documenter.jl/issues/868
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        mathengine = Documenter.MathJax(),
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    sitename = "ParametricOptInterface.jl",
    authors = "Tomás Gutierrez, and contributors",
    pages = ["Home" => "index.md", "reference.md"],
    checkdocs = :none,
)

Documenter.deploydocs(;
    repo = "github.com/jump-dev/ParametricOptInterface.jl.git",
    push_preview = true,
)
