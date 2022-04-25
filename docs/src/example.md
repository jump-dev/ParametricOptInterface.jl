# Examples

## MOI example

Lets write a setep-by-step example of `POI` usage at the MOI level.

First, we declare a `Optimizer` on top of a `MOI` optimizer. In the example, we consider `GLPK` as the underlying solver:

```julia
using GLPK
using MathOptInterface
using ParametricOptInterface

const MOI = MathOptInterface
const POI = ParametricOptInterface

optimizer = POI.Optimizer(GLPK.Optimizer())
```

Then, we declare the constants that will be used in this model, for ease of reference:

```julia
c = [4.0, 3.0]
A1 = [2.0, 1.0, 3.0]
A2 = [1.0, 2.0, 0.5]
b1 = 4.0
b2 = 4.0
```

We declare the variable `x` as in a typical `MOI` model, and we add a non-negativity constraint:

```julia
x = MOI.add_variables(optimizer, length(c))

for x_i in x
            MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
        end
```

Now, let's consider 3 parameters. Two of them, `y`, `z`, will be placed in the constraints and one, `w`, in the objective function. We'll start all three of them with a value equal to `0`:

```julia
w, cw = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
z, cz = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
```

Now, let's add the constraints. Notice that we treat parameters and variables in the same way when building the functions that will be placed in some set to create a constraint (`Function-in-Set`):

```julia
cons1 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A1, [x[1], x[2], y]), 0.0)
ci1 = MOI.add_constraint(optimizer, cons1, MOI.LessThan(b1))
```

```julia
cons2 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A2, [x[1], x[2], z]), 0.0)
ci2 = MOI.add_constraint(optimizer, cons2, MOI.LessThan(b2))
```

Finally, we declare and add the objective function, with its respective sense:

```julia
obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([c[1], c[2], 2.0], [x[1], x[2], w]), 0.0)
MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
```

Now we can optimize the model and assess its termination and primal status:

```julia
MOI.optimize!(optimizer)
MOI.get(optimizer, MOI.TerminationStatus())
MOI.get(optimizer, MOI.PrimalStatus())
```

Given the optimized solution, we check that its value is, as expected, equal to `28/3`, and the solution vector `x` is `[4/3, 4/3]`:

```julia
isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 28/3, atol = 1e-4)
isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4/3, atol = 1e-4)
isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 4/3, atol = 1e-4)
```

We can also retrieve the dual values associated to each parameter, **as they are all additive**:

```julia
MOI.get(optimizer, MOI.ConstraintDual(), cy)
MOI.get(optimizer, MOI.ConstraintDual(), cz)
MOI.get(optimizer, MOI.ConstraintDual(), cw)
```

Notice the direct relationship in this case between the parameters' duals and the associated constraints' duals.
The  `y` parameter, for example, only appears in the `cons1`. If we compare their duals, we can check that the dual of `y` is equal to its coefficient in `cons1` multiplied by the constraint's dual itself, as expected:

```julia
isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cy), 3*MOI.get(optimizer, MOI.ConstraintDual(), ci1), atol = 1e-4)
```

The same is valid for the remaining parameters. In case a parameter appears in more than one constraint, or both some constraints and in the objective function, its dual will be equal to the linear combination of the functions' duals multiplied by the respective coefficientes.

So far, we only added some parameters that had no influence at first in solving the model. Let's change the values associated to each parameter to assess its implications.
First, we set the value of parameters `y` and `z` to `1.0`. Notice that we are changing the feasible set of the decision variables:

```julia
MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(1.0))
MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(1.0))
```

However, if we check the optimized model now, there will be no changes in the objective function value or the in the optimized decision variables:

```julia
isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 28/3, atol = 1e-4)
isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4/3, atol = 1e-4)
isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 4/3, atol = 1e-4)
```

Although we changed the parameter values, we didn't optimize the model yet. Thus, **to apply the parameters' changes, the model must be optimized again**:

```julia
MOI.optimize!(optimizer)
```

The `MOI.optimize!()` function handles the necessary updates, properly fowarding the new outer model (`POI` model) additions to the inner model (`MOI` model) which will be handled by the solver. Now we can assess the updated optimized information:

```julia
isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 3.0, atol = 1e-4)
MOI.get.(optimizer, MOI.VariablePrimal(), x) == [0.0, 1.0]
```

If we update the parameter `w`, associated to the objective function, we are simply adding a constant to it. Notice how the new objective function is precisely equal to the previous one plus the new value of `w`. In addition, as we didn't update the feasible set, the optimized decision variables remain the same.

```julia
MOI.set(optimizer, MOI.ConstraintSet(), cw, POI.Parameter(2.0))
# Once again, the model must be optimized to incorporate the changes
MOI.optimize!(optimizer)
# Only the objective function value changes
isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 7.0, atol = 1e-4)
MOI.get.(optimizer, MOI.VariablePrimal(), x) == [0.0, 1.0]
```

## JuMP Example

Lets write a setep-by-step example of `POI` usage at the JuMP level.

First, we declare a `Optimizer` on top of a `MOI` optimizer. In the example, we consider `GLPK` as the underlying solver:

```julia
using GLPK
using JuMP
using ParametricOptInterface

const POI = ParametricOptInterface

model = Model(() -> POI.Optimizer(GLPK.Optimizer()))

@variable(model, x[i = 1:2] >= 0)
@variable(model, y in POI.Parameter(0))
@variable(model, z in POI.Parameter(0))
@variable(model, w in POI.Parameter(0))

@constraint(model, c1, 2x[1] + x[2] + 3y <= 4)
@constraint(model, c2, x[1] + 2x[2] + 0.5z <= 4)

@objective(model, Max, 4x[1] + 3x[2] + 2w)

optimize!(model)

MOI.get(model, POI.ParameterDual(), MOI.VariableIndex())

value.(x)

objective_value(model)

MOI.set(model, POI.ParameterValue(), y, 1)
MOI.set(model, POI.ParameterValue(), z, 1)

optimize!(model)

value.(x)

MOI.set(model, POI.ParameterValue(), w, 2)

optimize!(model)

objective_value(model)
```



