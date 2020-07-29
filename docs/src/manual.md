# Manual

```@contents
```


## Supported constraints

This is a list of supported `MOI` constraint functions that can handle parameters. If you try to add a parameter to 
a function that is not listed here, it will return an unsupported error.

|  MOI Function | 
|:-------|
|    `ScalarAffineFunction`    |
|    `ScalarQuadraticFunction`    |


## Supported objective functions

|  MOI Function | 
|:-------|
|    `ScalarAffineFunction`    |
|    `ScalarQuadraticFunction`    |

## Declare a ParametricOptimizer

in order to use parameters, the user needs to declare a `ParametricOptimizer` on top of a `MOI` optimizer, such as `GLPK.Optimizer()`.

```@docs
ParametricOptimizer{T, OT <: MOI.ModelLike} <: MOI.AbstractOptimizer
```
```julia

using ParametricOptInterface, MathOptInterface, GLPK

# Rename ParametricOptInterface and MathOptInterface to simplify the code
const POI = ParametricOptInterface
const MOI = MathOptInterface

# Define a ParametricOptimizer on top of the MOI optimizer
optimizer = POI.ParametricOptimizer(GLPK.Optimizer())

```

## Parameters

A `Parameter` is a variable with a fixed value that can be changed by the user.

```docs
Parameter
```

## Adding a new parameter to a model

```julia
y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
```

## Changing the parameter value

To change a given parameter's value, access its `ConstraintIndex` and set it to the new value using the `Parameter` structure.

```julia
MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
```