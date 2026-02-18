# ParametricOptInterface.jl

[ParametricOptInterface.jl](https://github.com/jump-dev/ParametricOptInterface.jl)
is a package for managing parameters in JuMP and MathOptInterface.

## License

`ParametricOptInterface.jl` is licensed under the
[MIT License](https://github.com/jump-dev/ParametricOptInterface.jl/blob/master/LICENSE.md).

## Installation

Install ParametricOptInterface using `Pkg.add`:

```julia
import Pkg
Pkg.add("ParametricOptInterface")
```

## Use with JuMP

Use ParametricOptInterface with JuMP by following this brief example:

```@repl
using JuMP, HiGHS
import ParametricOptInterface as POI
model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
@variable(model, x)
@variable(model, p in Parameter(1))
@constraint(model, p * x + p >= 3)
@objective(model, Min, 2x + p)
optimize!(model)
value(x)
set_parameter_value(p, 2.0)
optimize!(model)
value(x)
```

## The dual of a parameter

In some applications you may need the dual of a parameter. The dual can be
computed only if the parameter appears additively in the problem. Query the dual
associated with the parameter as follows:
```@repl
using JuMP, HiGHS
import ParametricOptInterface as POI
model = Model(() -> POI.Optimizer(HiGHS.Optimizer()));
set_silent(model)
@variable(model, x);
@variable(model, p in Parameter(1));
@constraint(model, x + p >= 3);
@objective(model, Min, 2x);
optimize!(model)
dual(VariableInSetRef(p))
```

Note how the dual is the same as the `reduced_cost` of an equivalent fixed
variable:
```@repl
using JuMP, HiGHS
model = Model(HiGHS.Optimizer);
set_silent(model)
@variable(model, x);
@variable(model, p == 1);
@constraint(model, x + p >= 3);
@objective(model, Min, 2x);
optimize!(model)
reduced_cost(p)
```

Note that, even though ParametricOptInterface supports querying the dual of a
parameter, this is not true in general. For example, Ipopt does not support
querying the dual of a parameter. Moreover, the dual is available only if all of
the parameters appear additively in the model; the dual is not available if
there are multiplicative parameters.

## Variable bounds

An ambiguity arises when the user writes a model like:
```@repl
using JuMP
model = Model();
@variable(model, x)
@variable(model, p in Parameter(2))
@constraint(model, x >= p)
```
Did they mean an affine constraint like `1.0 * x + 0.0 in GreaterThan(2.0)`, or
did they mean a variable bound like `x in GreaterThan(2.0)`?

By default, ParametricOptInterface does not attempt to simplify affine
constraints involving parameters to variable bounds, but this behavior can be
controlled using the [`ConstraintsInterpretation`](@ref) attribute.

For example, the default is:
```@repl
using JuMP, HiGHS
import ParametricOptInterface as POI
model = Model(() -> POI.Optimizer(HiGHS.Optimizer()));
set_attribute(model, POI.ConstraintsInterpretation(), POI.ONLY_CONSTRAINTS)
@variable(model, x);
@variable(model, p in Parameter(1));
@constraint(model, x >= p)
optimize!(model)
```
but by setting the [`ConstraintsInterpretation`](@ref) attribute to
`POI.BOUNDS_AND_CONSTRAINTS`, we can solve a problem with one decision variable
and zero constraint rows:
```@repl
using JuMP, HiGHS
import ParametricOptInterface as POI
model = Model(() -> POI.Optimizer(HiGHS.Optimizer()));
set_attribute(model, POI.ConstraintsInterpretation(), POI.BOUNDS_AND_CONSTRAINTS)
@variable(model, x);
@variable(model, p in Parameter(1));
@constraint(model, x >= p)
optimize!(model)
```

## Parameters multiplying quadratic terms

POI supports parameters that multiply quadratic variable terms in objectives
**only**. This creates cubic polynomial expressions of the form `c * p * x * y`
where `c` is a number, `p` is a parameter, and `x` and `y` are variables. After
parameter substitution, the objective is quadratic instead of cubic.

Note that the maximum degree is 3 (cubic), at least one factor in each cubic
term must be a parameter, and pure cubic variable terms (for example,
`x * y * z` with no parameters) are not supported.

```@repl
using JuMP, HiGHS
import ParametricOptInterface as POI
model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
set_silent(model)
@variable(model, 0 <= x <= 10)
@variable(model, p in Parameter(2))
@objective(model, Min, p * x^2 - 3x)
optimize!(model)
value(x)  # x = 3 / 2p = 0.75
set_parameter_value(p, 3)
optimize!(model)
value(x)  # x = 3 / 2p = 0.5
```
