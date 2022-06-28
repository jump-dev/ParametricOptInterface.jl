# Basic Examples

## MOI example - step by step usage

Let's write a step-by-step example of `POI` usage at the MOI level.

First, we declare a [`ParametricOptInterface.Optimizer`](@ref) on top of a `MOI` optimizer. In the example, we consider `HiGHS` as the underlying solver:

```@example moi1
using HiGHS
using MathOptInterface
using ParametricOptInterface

const MOI = MathOptInterface
const POI = ParametricOptInterface

optimizer = POI.Optimizer(HiGHS.Optimizer())
```

We declare the variable `x` as in a typical `MOI` model, and we add a non-negativity constraint:

```@example moi1
x = MOI.add_variables(optimizer, 2)
for x_i in x
    MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
end
```

Now, let's consider 3 [`ParametricOptInterface.Parameter`](@ref). Two of them, `y`, `z`, will be placed in the constraints and one, `w`, in the objective function. We'll start all three of them with a value equal to `0`:

```@example moi1
w, cw = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
z, cz = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
```

Let's add the constraints. Notice that we treat parameters and variables in the same way when building the functions that will be placed in some set to create a constraint (`Function-in-Set`):

```@example moi1
cons1 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 1.0, 3.0], [x[1], x[2], y]), 0.0)
ci1 = MOI.add_constraint(optimizer, cons1, MOI.LessThan(4.0))
cons2 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0, 0.5], [x[1], x[2], z]), 0.0)
ci2 = MOI.add_constraint(optimizer, cons2, MOI.LessThan(4.0))
```

Finally, we declare and add the objective function, with its respective sense:

```@example moi1
obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([4.0, 3.0, 2.0], [x[1], x[2], w]), 0.0)
MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
```

Now we can optimize the model and assess its termination and primal status:

```@example moi1
MOI.optimize!(optimizer)
MOI.get(optimizer, MOI.TerminationStatus())
MOI.get(optimizer, MOI.PrimalStatus())
```

Given the optimized solution, we check that its value is, as expected, equal to `28/3`, and the solution vector `x` is `[4/3, 4/3]`:

```@example moi1
isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 28/3, atol = 1e-4)
isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4/3, atol = 1e-4)
isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 4/3, atol = 1e-4)
```

We can also retrieve the dual values associated to each parameter, **as they are all additive**:

```@example moi1
MOI.get(optimizer, MOI.ConstraintDual(), cy)
MOI.get(optimizer, MOI.ConstraintDual(), cz)
MOI.get(optimizer, MOI.ConstraintDual(), cw)
```

Notice the direct relationship in this case between the parameters' duals and the associated constraints' duals.
The  `y` parameter, for example, only appears in the `cons1`. If we compare their duals, we can check that the dual of `y` is equal to its coefficient in `cons1` multiplied by the constraint's dual itself, as expected:

```@example moi1
isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cy), 3*MOI.get(optimizer, MOI.ConstraintDual(), ci1), atol = 1e-4)
```

The same is valid for the remaining parameters. In case a parameter appears in more than one constraint, or both some constraints and in the objective function, its dual will be equal to the linear combination of the functions' duals multiplied by the respective coefficients.

So far, we only added some parameters that had no influence at first in solving the model. Let's change the values associated to each parameter to assess its implications.
First, we set the value of parameters `y` and `z` to `1.0`. Notice that we are changing the feasible set of the decision variables:

```@example moi1
MOI.set(optimizer, POI.ParameterValue(), y, 1.0)
MOI.set(optimizer, POI.ParameterValue(), z, 1.0)
```

However, if we check the optimized model now, there will be no changes in the objective function value or the in the optimized decision variables:

```@example moi1
isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 28/3, atol = 1e-4)
isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4/3, atol = 1e-4)
isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 4/3, atol = 1e-4)
```

Although we changed the parameter values, we didn't optimize the model yet. Thus, **to apply the parameters' changes, the model must be optimized again**:

```@example moi1
MOI.optimize!(optimizer)
```

The `MOI.optimize!()` function handles the necessary updates, properly fowarding the new outer model (`POI` model) additions to the inner model (`MOI` model) which will be handled by the solver. Now we can assess the updated optimized information:

```@example moi1
isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 3.0, atol = 1e-4)
MOI.get.(optimizer, MOI.VariablePrimal(), x) == [0.0, 1.0]
```

If we update the parameter `w`, associated to the objective function, we are simply adding a constant to it. Notice how the new objective function is precisely equal to the previous one plus the new value of `w`. In addition, as we didn't update the feasible set, the optimized decision variables remain the same.

```@example moi1
MOI.set(optimizer, POI.ParameterValue(), w, 2.0)
# Once again, the model must be optimized to incorporate the changes
MOI.optimize!(optimizer)
# Only the objective function value changes
isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 7.0, atol = 1e-4)
MOI.get.(optimizer, MOI.VariablePrimal(), x) == [0.0, 1.0]
```

## MOI Example - Parameters multiplying Quadratic terms

Let's start with a simple quadratic problem

```@example moi2
using Ipopt
using MathOptInterface
using ParametricOptInterface

const MOI = MathOptInterface
const POI = ParametricOptInterface

optimizer = POI.Optimizer(Ipopt.Optimizer())

x = MOI.add_variable(optimizer)
y = MOI.add_variable(optimizer)
MOI.add_constraint(optimizer, x, MOI.GreaterThan(0.0))
MOI.add_constraint(optimizer, y, MOI.GreaterThan(0.0))

cons1 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 1.0], [x, y]), 0.0)
ci1 = MOI.add_constraint(optimizer, cons1, MOI.LessThan(4.0))
cons2 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0], [x, y]), 0.0)
ci2 = MOI.add_constraint(optimizer, cons2, MOI.LessThan(4.0))

MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
obj_func = MOI.ScalarQuadraticFunction(
    [MOI.ScalarQuadraticTerm(1.0, x, x)
    MOI.ScalarQuadraticTerm(1.0, y, y)],
    MathOptInterface.ScalarAffineTerm{Float64}[],
    0.0,
)
MOI.set(
    optimizer,
    MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
    obj_func,
)
```

To multiply a parameter in a quadratic term, the user will 
need to use the `POI.QuadraticObjectiveCoef` model attribute.

```@example moi2
p = first(MOI.add_constrained_variable.(optimizer, POI.Parameter(1.0)))
MOI.set(optimizer, POI.QuadraticObjectiveCoef(), (x,y), p)
```

This function will add the term `p*xy` to the objective function.
It's also possible to multiply a scalar affine function to the quadratic term.

```@example moi2
MOI.set(optimizer, POI.QuadraticObjectiveCoef(), (x,y), 2p+3)
```

This will set the term `(2p+3)*xy` to the objective function (it overwrites the last set).
Then, just optimize the model.

```@example moi2
MOI.optimize!(model)
isapprox(MOI.get(model, MOI.ObjectiveValue()), 32/3, atol=1e-4)
isapprox(MOI.get(model, MOI.VariablePrimal(), x), 4/3, atol=1e-4)
isapprox(MOI.get(model, MOI.VariablePrimal(), y), 4/3, atol=1e-4)
```

To change the parameter just set `POI.ParameterValue` and optimize again.

```@example moi2
MOI.set(model, POI.ParameterValue(), p, 2.0)
MOI.optimize!(model)
isapprox(MOI.get(model, MOI.ObjectiveValue()), 128/9, atol=1e-4)
isapprox(MOI.get(model, MOI.VariablePrimal(), x), 4/3, atol=1e-4)
isapprox(MOI.get(model, MOI.VariablePrimal(), y), 4/3, atol=1e-4)
```

## JuMP Example - step by step usage

Let's write a step-by-step example of `POI` usage at the JuMP level.

First, we declare a `Model` on top of a `Optimizer` of an underlying solver. In the example, we consider `HiGHS` as the underlying solver:

```@example jump1
using HiGHS
using JuMP

using ParametricOptInterface
const POI = ParametricOptInterface

model = Model(() -> ParametricOptInterface.Optimizer(HiGHS.Optimizer()))
```

We declare the variable `x` as in a typical `JuMP` model:

```@example jump1
@variable(model, x[i = 1:2] >= 0)
```

Now, let's consider 3 [`ParametricOptInterface.Parameter`](@ref). Two of them, `y`, `z`, will be placed in the constraints and one, `w`, in the objective function. We'll start all three of them with a value equal to `0`:

```@example jump1
@variable(model, y in ParametricOptInterface.Parameter(0))
@variable(model, z in ParametricOptInterface.Parameter(0))
@variable(model, w in ParametricOptInterface.Parameter(0))
```

Let's add the constraints. Notice that we treat parameters the same way we treat variables when writing the model:

```@example jump1
@constraint(model, c1, 2x[1] + x[2] + 3y <= 4)
@constraint(model, c2, x[1] + 2x[2] + 0.5z <= 4)
```

Finally, we declare and add the objective function, with its respective sense:

```@example jump1
@objective(model, Max, 4x[1] + 3x[2] + 2w)
```

We can optimize the model and assess its termination and primal status:

```@example jump1
optimize!(model)
termination_status(model)
primal_status(model)
```

Given the optimized solution, we check that its value is, as expected, equal to `28/3`, and the solution vector `x` is `[4/3, 4/3]`:

```@example jump1
isapprox(objective_value(model), 28/3)
isapprox(value.(x), [4/3, 4/3])
```

We can also retrieve the dual values associated to each parameter, **as they are all additive**:

```@example jump1
MOI.get(model, POI.ParameterDual(), y)
MOI.get(model, POI.ParameterDual(), z)
MOI.get(model, POI.ParameterDual(), w)
```

Notice the direct relationship in this case between the parameters' duals and the associated constraints' duals. The `y` parameter, for example, only appears in the `c1`. If we compare their duals, we can check that the dual of `y` is equal to its coefficient in `c1` multiplied by the constraint's dual itself, as expected:

```@example jump1
dual_of_y = MOI.get(model, POI.ParameterDual(), y)
isapprox(dual_of_y, 3 * dual(c1))
```

The same is valid for the remaining parameters. In case a parameter appears in more than one constraint, or both some constraints and in the objective function, its dual will be equal to the linear combination of the functions' duals multiplied by the respective coefficients.

So far, we only added some parameters that had no influence at first in solving the model. Let's change the values associated to each parameter to assess its implications. First, we set the value of parameters `y` and `z` to `1.0`. Notice that we are changing the feasible set of the decision variables:

```@example jump1
MOI.set(model, POI.ParameterValue(), y, 1)
MOI.set(model, POI.ParameterValue(), z, 1)
# We can also query the value in the parameters
MOI.get(model, POI.ParameterValue(), y)
MOI.get(model, POI.ParameterValue(), z)
```

To apply the parameters' changes, the model must be optimized again:

```@example jump1
optimize!(model)
```

The `optimize!` function handles the necessary updates, properly fowarding the new outer model (`POI` model) additions to the inner model (`MOI` model) which will be handled by the solver. Now we can assess the updated optimized information:

```@example jump1
isapprox(objective_value(model), 3)
isapprox(value.(x), [0, 1])
```

If we update the parameter `w`, associated to the objective function, we are simply adding a constant to it. Notice how the new objective function is precisely equal to the previous one plus the new value of `w`. In addition, as we didn't update the feasible set, the optimized decision variables remain the same.

```@example jump1
MOI.set(model, POI.ParameterValue(), w, 2)
# Once again, the model must be optimized to incorporate the changes
optimize!(model)
# Only the objective function value changes
isapprox(objective_value(model), 7)
isapprox(value.(x), [0, 1])
```

## JuMP Example - Declaring vectors of parameters

Many times it is useful to declare a vector of parameters just like we declare a vector of variables, the JuMP syntax for variables works with parameters too:


```@example jump2
using HiGHS
using JuMP
using ParametricOptInterface
const POI = ParametricOptInterface

model = Model(() -> ParametricOptInterface.Optimizer(HiGHS.Optimizer()))
@variable(model, x[i = 1:3] >= 0)
@variable(model, p1[i = 1:3] in ParametricOptInterface.Parameter(0))
@variable(model, p2[i = 1:3] in ParametricOptInterface.Parameter.([1, 10, 45]))
@variable(model, p3[i = 1:3] in ParametricOptInterface.Parameter.(ones(3)))
```

## JuMP Example - Dealing with parametric expressions as variable bounds

A very common pattern that appears when using ParametricOptInterface is to add variable and later add some expression with parameters that represent the variable bound. The following code illustrates the pattern:

```@example jump3
using HiGHS
using JuMP
using ParametricOptInterface
const POI = ParametricOptInterface

model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
@variable(model, x)
@variable(model, p in POI.Parameter(0.0))
@constraint(model, x >= p)
```

Since parameters are treated like variables JuMP lowers this to MathOptInterface as `x - p >= 0` which is not a variable bound but a linear constraint.This means that the current representation of this problem at the solver level is:

```math
\begin{align}
    & \min_{x} & 0
    \\
    & \;\;\text{s.t.} & x & \in \mathbb{R} \\
    &   & x - p & \geq 0
\end{align}
```

This behaviour might be undesirable because it creates extra rows in your problem. Users can set the [`ParametricOptInterface.ConstraintsInterpretation`](@ref) to control how the linear constraints should be interpreted. The pattern advised for users seeking the most performance out of ParametricOptInterface should use the followig pattern:

```@example jump3
using HiGHS
using JuMP
using ParametricOptInterface
const POI = ParametricOptInterface

model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
@variable(model, x)
@variable(model, p in POI.Parameter(0.0))

# Indicate that all the new constraints will be valid variable bounds
MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)
@constraint(model, x >= p)
# The name of this constraint was different to inform users that this is a
# variable bound.

# Indicate that all the new constraints will not be variable bounds
MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_CONSTRAINTS)
# @constraint(model, ...)
```

This way the mathematical representation of the problem will be:

```math
\begin{align}
    & \min_{x} & 0
    \\
    & \;\;\text{s.t.} & x & \geq p
\end{align}
```

which might lead to faster solves.

Users that just want everything to work can use the default value `POI.ONLY_CONSTRAINTS` or try to use `POI.BOUNDS_AND_CONSTRAINTS` and leave it to ParametricOptInterface to interpret the constraints as bounds when applicable and linear constraints otherwise.

## JuMP Example - Parameters multiplying Quadratic terms

Let's get the same MOI example

```@example jump4
using Ipopt
using JuMP
using ParametricOptInterface
const POI = ParametricOptInterface

optimizer = POI.Optimizer(Ipopt.Optimizer())
model = direct_model(optimizer)

@variable(model, x >= 0)
@variable(model, y >= 0)
@variable(model, p in POI.Parameter(1.0))
@constraint(model, 2x + y <= 4)
@constraint(model, x + 2y <= 4)
@objective(model, Max, (x^2 + y^2)/2)
```

We use the same MOI function to add the parameter multiplied to the quadratic term.

```@example jump4
MOI.set(backend(model), POI.QuadraticObjectiveCoef(), (index(x),index(y)), 2index(p)+3)
```

If the user print the `model`, the term `(2p+3)*xy` won't show. 
It's possible to retrieve the parametric function multiplying the term `xy` with `MOI.get`.

```@example jump4
MOI.get(backend(model), POI.QuadraticObjectiveCoef(), (index(x),index(y)))
```

Then, just optimize the model

```@example jump4
optimize!(model)
isapprox(objective_value(model), 32/3, atol=1e-4)
isapprox(value(x), 4/3, atol=1e-4)
isapprox(value(y), 4/3, atol=1e-4)
```

To change the parameter just set `POI.ParameterValue` and optimize again.

```@example jump4
MOI.set(model, POI.ParameterValue(), p, 2.0)
optimize!(model)
isapprox(objective_value(model), 128/9, atol=1e-4)
isapprox(value(x), 4/3, atol=1e-4)
isapprox(value(y), 4/3, atol=1e-4)
```
## JuMP Example - Non Linear Programming (NLP)

POI currently works with NLPs when users wish to add the parameters to the non-NL constraints or objective. This means that POI works with models like this one:

```julia
@variable(model, x)
@variable(model, y)
@variable(model, z in POI.Parameter(10))
@constraint(model, x + y >= z)
@NLobjective(model, Min, x^2 + y^2)
```

but does not work with models that have parameters on the NL expressions like this one:

```julia
@variable(model, x)
@variable(model, y)
@variable(model, z in POI.Parameter(10))
@constraint(model, x + y >= z)
@NLobjective(model, Min, x^2 + y^2 + z) # There is a parameter here
```

If users with to add parameters in NL expressions we strongly recommend them to read [this section on the JuMP documentation]((https://jump.dev/JuMP.jl/stable/manual/nlp/#Create-a-nonlinear-parameter))

Although POI works with NLPs there are some important information for users to keep in mind. All come from the fact that POI relies on the MOI interface for problem modifications and these are not common on NLP solvers, most solvers only allow users to modify variable bounds using their official APIs. This means that if users wish to make modifications on some constraint that is not a variable bound we are not allowed to call `MOI.modify` because the function is not supported in the MOI solver interface. The work-around to this is defining a [`POI.Optimizer`](@ref) on a caching optimizer:

```julia
ipopt = Ipopt.Optimizer()
MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
cached =
    () -> MOI.Bridges.full_bridge_optimizer(
        MOIU.CachingOptimizer(
            MOIU.UniversalFallback(MOIU.Model{Float64}()),
            ipopt,
        ),
        Float64,
    )
POI_cached_optimizer() = POI.Optimizer(cached())
model = Model(() -> POI_cached_optimizer())
@variable(model, x)
@variable(model, y)
@variable(model, z in POI.Parameter(10))
@constraint(model, x + y >= z)
@NLobjective(model, Min, x^2 + y^2)
```

This works but keep in mind that the model has an additional layer of between the solver and the [`POI.Optimizer`](@ref). This will make most operations slower than with the version without the caching optimizer. Keep in mind that since the official APIs of most solvers don't allow for modifications on linear constraints there should have no big difference between making a modification using POI or re-building the model from scratch.

If users wish to make modifications on variable bounds the POI interface will help you save time between solves. In this case you should use the [`ParametricOptInterface.ConstraintsInterpretation`](@ref) as we do in this example:

```julia
model = Model(() -> POI.Optimizer(Ipopt.Optimizer()))
@variable(model, x)
@variable(model, z in POI.Parameter(10))
MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)
@constraint(model, x >= z)
@NLobjective(model, Min, x^2)
```

This use case should help users diminsh the time of making model modifications and re-solve the model. To increase the performance users that are familiar with [JuMP direct mode](https://jump.dev/JuMP.jl/stable/manual/models/#Direct-mode) can also use it.
