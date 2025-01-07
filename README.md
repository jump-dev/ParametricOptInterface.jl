# ParametricOptInterface.jl

[![stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://jump.dev/ParametricOptInterface.jl/stable)
[![development docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://jump.dev/ParametricOptInterface.jl/dev)
[![Build Status](https://github.com/jump-dev/ParametricOptInterface.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/jump-dev/ParametricOptInterface.jl/actions?query=workflow%3ACI)
[![Coverage](https://codecov.io/gh/jump-dev/ParametricOptInterface.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jump-dev/ParametricOptInterface.jl)

[ParametricOptInterface.jl](https://github.com/jump-dev/ParametricOptInterface.jl)
is a package that adds parameters to models in JuMP and MathOptInterface.

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

The [documentation for ParametricOptInterface.jl](https://jump.dev/ParametricOptInterface.jl/stable/)
includes a detailed description of the theory behind the package, along with
examples, tutorials, and an API reference.

## Use with JuMP

Use ParametricOptInterface with JuMP by following this brief example:

```julia
using JuMP, HiGHS
import ParametricOptInterface as POI
model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
@variable(model, x)
@variable(model, p in Parameter(1.0))
@constraint(model, cons, x + p >= 3)
@objective(model, Min, 2x)
optimize!(model)
@show value(x)
set_parameter_value(p, 2.0)
optimize!(model)
@show value(x)
```

## GSOC2020

ParametricOptInterface began as a [NumFOCUS sponsored Google Summer of Code (2020) project](https://summerofcode.withgoogle.com/archive/2020/projects/4959861055422464).
