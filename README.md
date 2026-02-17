# ParametricOptInterface.jl

[![Build Status](https://github.com/jump-dev/ParametricOptInterface.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/jump-dev/ParametricOptInterface.jl/actions?query=workflow%3ACI)
[![Coverage](https://codecov.io/gh/jump-dev/ParametricOptInterface.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jump-dev/ParametricOptInterface.jl)

[ParametricOptInterface.jl](https://github.com/jump-dev/ParametricOptInterface.jl)
is a package for managing parameters in JuMP and MathOptInterface.

## Getting help

If you need help, please ask a question on the [JuMP community forum](https://jump.dev/forum).

If you have a reproducible example of a bug, please [open a GitHub issue](https://github.com/jump-dev/HiGHS.jl/issues/new).

## License

`ParametricOptInterface.jl` is licensed under the
[MIT License](https://github.com/jump-dev/ParametricOptInterface.jl/blob/master/LICENSE.md).

## Installation

Install ParametricOptInterface using `Pkg.add`:

```julia
import Pkg
Pkg.add("ParametricOptInterface")
```

## Documentation

See the [documentation for ParametricOptInterface.jl](https://jump.dev/ParametricOptInterface.jl),
as well as [tutorials that use ParametricOptInterface](https://jump.dev/JuMP.jl/stable/tutorials/overview/#tutorial_ParametricOptInterface)
in the JuMP documentation.

## Use with JuMP

Use ParametricOptInterface with JuMP by following this brief example:

```julia
julia> using JuMP, HiGHS

julia> import ParametricOptInterface as POI

julia> model = Model(() -> POI.Optimizer(HiGHS.Optimizer));

julia> set_silent(model)

julia> @variable(model, x)
x

julia> @variable(model, p in Parameter(1))
p

julia> @constraint(model, x + p >= 3)
x + p ≥ 3

julia> @objective(model, Min, 2x)
2 x

julia> optimize!(model)

julia> value(x)
2.0

julia> set_parameter_value(p, 2.0)

julia> optimize!(model)

julia> value(x)
1.0
```

## GSOC2020

ParametricOptInterface began as a [NumFOCUS sponsored Google Summer of Code (2020) project](https://summerofcode.withgoogle.com/archive/2020/projects/4959861055422464).
