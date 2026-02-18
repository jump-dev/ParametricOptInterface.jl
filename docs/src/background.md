# How and why ParametricOptInterface is needed

JuMP and MathOptInterface have support for _parameters_. Parameters are decision
variables that belong to the `Parameter` set. The `Parameter` set is
conceptually similar to the `EqualTo` set, except that solvers may treat a
decision variable constrained to the `Parameter` set as a constant, and they
do not need to add it as a decision variable to the model.

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

## Some solvers have native support for parameters

Some solvers have native support for parameters. One example is Ipopt. To
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

## Why parameters are useful

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

## Some solvers do not have native support for parameters

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

## ParametricOptInterface

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

To summarize, you should use ParametricOptInterface when:

 * you are using a solver that does not have native support for parameters
 * you are solving a single problem for multiple values of the parameters.

For problems with a small number of parameters, and in which the parameters
appear additively in the constraints and the objective, it may be more efficient
to use the bridge approach. In general, you should try with and without POI and
choose the approach which works best for your model.
