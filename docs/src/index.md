# ParametricOptInterface.jl

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

## Use with JuMP

Use ParametricOptInterface with JuMP by following this brief example:

```@repl
using JuMP, HiGHS
import ParametricOptInterface as POI
model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
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

## How and why ParametricOptInterface is needed

JuMP and MathOptInterface have support for _parameters_. Parameters are decision
variables that belong to the `Parameter` set. The `Parameter` set is
conceptually similar to the `EqualTo` set, except that solvers may treat a
decision variable constrained to the `Parameter` set as a constant, and they
need not add it as a decision variable to the model.

In JuMP, a parameter can be added using the following syntax:
```@repl
using JuMP
model = Model();
@variable(model, p in Parameter(2))
parameter_value(p)
set_parameter_value(p, 3.0)
parameter_value(p)
```

In MathOptInterface, a parameter can be added using the following syntax:
```@repl
import MathOptInterface as MOI
model = MOI.Utilities.Model{Float64}();
p, p_con = MOI.add_constrained_variable(model, MOI.Parameter(2.0))
MOI.get(model, MOI.ConstraintSet(), p_con)
new_set = MOI.Parameter(3.0)
MOI.set(model, MOI.ConstraintSet(), p_con, new_set)
MOI.get(model, MOI.ConstraintSet(), p_con)
```

### Some solvers have native support for parameters

Some solvers have native support for parameters. example is Ipopt. To
demonstrate, look at the following example. Even though there are two
`@variable` calls, the log of Ipopt shows that it solved a problem with only one
decision variable:
```@repl
using JuMP, Ipopt
model = Model(Ipopt.Optimizer)
@variable(model, x)
@variable(model, p in Parameter(1))
@constraint(model, x + p >= 3)
@objective(model, Min, 2x)
optimize!(model)
```
Internally, Ipopt replaced the parameter `p` with the constant `1.0`, and solved
the problem:
```@repl
using JuMP, Ipopt
model = Model(Ipopt.Optimizer)
@variable(model, x)
@constraint(model, x + 1 >= 3)
@objective(model, Min, 2x)
optimize!(model)
```

Parameters are most useful when you want to solve a sequence of problems in
which some of the data changes between iterations:
```@repl
using JuMP, Ipopt
model = Model(Ipopt.Optimizer)
set_silent(model)
@variable(model, x)
@variable(model, p in Parameter(1))
@constraint(model, x + p >= 3)
@objective(model, Min, 2x)
solution = Dict{Int,Float64}()
for p_value in 0:5
    set_parameter_value(p, p_value)
    optimize!(model)
    assert_is_solved_and_feasible(model)
    solution[p_value] = value(x)
end
solution
```

### Some solvers do not have native support for parameters

Even though solvers like Ipopt support parameters, many solvers do not. One
example is HiGHS. Despite the fact that HiGHS doesn't support parameters, you
can still build and solve a model with parameters:
```@repl index_highs
using JuMP, HiGHS
model = Model(HiGHS.Optimizer)
@variable(model, x)
@variable(model, p in Parameter(1))
@constraint(model, x + p >= 3)
@objective(model, Min, 2x)
optimize!(model)
```
This works because, behind the scenes, the bridges in MathOptInterface rewrote
`p in Parameter(1)` to `p in MOI.EqualTo(1.0)`:
```@repl index_highs
print_active_bridges(model)
```
Thus, HiGHS solved the problem:
```@repl
using JuMP, HiGHS
model = Model(HiGHS.Optimizer)
@variable(model, x)
@variable(model, p == 1)
@constraint(model, x + p >= 3)
@objective(model, Min, 2x)
optimize!(model)
```

The downside to the bridge approach is that it adds a new decision variable with
fixed bounds for every parameter in the problem. Moreover, the bridge approach
does not handle `parameter * variable` terms, because the resulting problem is a
quadratic constraint:

```jldoctest
julia> using JuMP, HiGHS

julia> model = Model(HiGHS.Optimizer);

julia> @variable(model, x);

julia> @variable(model, p in Parameter(1));

julia> @constraint(model, p * x >= 3)
ERROR: Constraints of type MathOptInterface.ScalarQuadraticFunction{Float64}-in-MathOptInterface.GreaterThan{Float64} are not supported by the solver.

If you expected the solver to support your problem, you may have an error in your formulation. Otherwise, consider using a different solver.

The list of available solvers, along with the problem types they support, is available at https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers.
Stacktrace:
[...]
```

### ParametricOptInterface

ParametricOptInterface provides [`Optimizer`](@ref), which is a meta-optimizer
that wraps another optimizer. Instead of adding fixed variables to the model,
POI substitutes out the parameters with their value before passing the
constraint or objective to the inner optimizer. When the parameter value is
changed, POI efficiently modifies the inner optimizer to reflect the new
parameter values.

```@repl
using JuMP, HiGHS
import ParametricOptInterface as POI
model = Model(() -> POI.Optimizer(HiGHS.Optimizer()));
@variable(model, x);
@variable(model, p in Parameter(1));
@constraint(model, x + p >= 3);
@objective(model, Min, 2x);
optimize!(model)
```
Note how HiGHS now solves a problem with one decision variable.

Because POI replaces parameters with their constant value, POI supports
`parameter * variable` terms:
```@repl
using JuMP, HiGHS
import ParametricOptInterface as POI
model = Model(() -> POI.Optimizer(HiGHS.Optimizer()));
@variable(model, x);
@variable(model, p in Parameter(1));
@constraint(model, p * x >= 3)
@objective(model, Min, 2x)
optimize!(model)
```

## When to use ParametricOptInterface

You should use ParametricOptInterface when:

 * you are using a solver that does not have native support for parameters
 * you are solving a single problem for multiple values of the parameters.

For problems with a small number of parameters, and in which the parameters
appear additively in the constraints and the objective, it may be more efficient
to use the bridge approach. In general, you should try with and without POI and
choose the approach which works best for your model.

## The dual of a parameter

In some applications you may need the dual of a parameter. The dual can be
computed only if the parameter appears additively in the problem. Query the dual
assocaited with the parameter using [`ParameterDual`](@ref):
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
get_attribute(p, POI.ParameterDual())
```

Note how the dual is the same as the `reduced_cost` of an equivalent fixed variable:
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
where `c` is a number, `p` is a parameter and `x`, `y` are variables. After
parameter substitution, these become standard quadratic terms that solvers can
handle.


### Attention

- Maximum degree is 3 (cubic)
- At least one factor in each cubic term must be a parameter
- Pure cubic variable terms (e.g., `x * y * z` with no parameters) are
  **not supported**

### Example using JuMP

```@repl
using JuMP, HiGHS
import ParametricOptInterface as POI

# Create model with POI optimizer
model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
set_silent(model)

# Define variables and parameter
@variable(model, 0 <= x <= 10)
@variable(model, p in MOI.Parameter(2.0))

# Set cubic objective: p * x * y
@objective(model, Min, p * x ^ 2 - 3x)

# Solve
optimize!(model)

@show value(x) # == 3 / (2 * p)

# Update parameter and re-solve
# Parameters are automatically updated when optimize! is called
set_parameter_value(p, 3.0)
optimize!(model)

@show value(x) # == 3 / (2 * p)
```
