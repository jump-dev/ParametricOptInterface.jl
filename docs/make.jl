# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import Documenter
import ParametricOptInterface

# ==============================================================================
#  Modify the release notes
# ==============================================================================

function fix_release_line(
    line::String,
    url::String = "https://github.com/jump-dev/ParametricOptInterface.jl",
)
    # (#XXXX) -> ([#XXXX](url/issue/XXXX))
    while (m = match(r"\(\#([0-9]+)\)", line)) !== nothing
        id = m.captures[1]
        line = replace(line, m.match => "([#$id]($url/issues/$id))")
    end
    # ## Version X.Y.Z -> [Version X.Y.Z](url/releases/tag/vX.Y.Z)
    while (m = match(r"\#\# Version ([0-9]+.[0-9]+.[0-9]+)", line)) !== nothing
        tag = m.captures[1]
        line = replace(
            line,
            m.match => "## [Version $tag]($url/releases/tag/v$tag)",
        )
    end
    # ## vX.Y.Z -> [vX.Y.Z](url/releases/tag/vX.Y.Z)
    while (m = match(r"\#\# (v[0-9]+.[0-9]+.[0-9]+)", line)) !== nothing
        tag = m.captures[1]
        line = replace(line, m.match => "## [$tag]($url/releases/tag/$tag)")
    end
    return line
end

function _fix_release_lines(changelog, release_notes, args...)
    open(release_notes, "w") do io
        for line in readlines(changelog; keep = true)
            write(io, fix_release_line(line, args...))
        end
    end
    return
end

_fix_release_lines(
    joinpath(@__DIR__, "src", "changelog.md"),
    joinpath(@__DIR__, "src", "release_notes.md"),
)

function _add_edit_url(filename, url)
    contents = read(filename, String)
    open(filename, "w") do io
        write(io, "```@meta\nEditURL = \"$url\"\n```\n\n")
        write(io, contents)
        return
    end
    return
end

_add_edit_url(joinpath(@__DIR__, "src", "release_notes.md"), "changelog.md")

# ==============================================================================
#  Build and deploy the documentation
# ==============================================================================

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
    pages = [
        "Home" => "index.md",
        "background.md",
        "reference.md",
        "release_notes.md",
    ],
    checkdocs = :none,
)

Documenter.deploydocs(;
    repo = "github.com/jump-dev/ParametricOptInterface.jl.git",
    push_preview = true,
)
