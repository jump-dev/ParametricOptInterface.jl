# Manual

## Why use parameters?

A typical optimization model built using `MathOptInterface.jl` (`MOI`for short) has two main components:
1. Variables
2. Constants

Using these basic elements, one can create functions and sets that, together, form the desired optimization model. The goal of `POI` is the implementation of a third type, parameters, which
* are declared similar to a variable, and inherits some functionalities (e.g. dual calculation)
* acts like a constant, in the sense that it has a fixed value that will remain the same unless explicitely changed by the user

A main concern is to efficiently implement this new type, as one typical usage is to change its value to analyze the model behavior, without the need to build a new one from scratch.

## How it works

The main idea applied in POI is that the interaction between the solver, e.g. `GLPK`, and the optimization model will be handled by `MOI` as usual. Because of that, `POI` is a higher level wrapper around `MOI`, responsible for receiving variables, constants and parameters, and forwarding to the lower level model only variables and constants.

As `POI` receives parameters, it must analyze and decide how they should be handled on the lower level optimization model (the `MOI` model).

## Usage

In this manual we describe how to interact with the optimization model at the MOI level. In the [Examples](@ref) section you can find some tutorials with the JuMP usage.

### Supported constraints

This is a list of supported `MOI` constraint functions that can handle parameters. If you try to add a parameter to
a function that is not listed here, it will return an unsupported error.

|  MOI Function |
|:-------|
|    `ScalarAffineFunction`    |
|    `ScalarQuadraticFunction`    |
|    `VectorAffineFunction`    |


### Supported objective functions

|  MOI Function |
|:-------|
|    `ScalarAffineFunction`    |
|    `ScalarQuadraticFunction`    |

### Declare a Optimizer

In order to use parameters, the user needs to declare a [`ParametricOptInterface.Optimizer`](@ref) on top of a `MOI` optimizer, such as `HiGHS.Optimizer()`.

```julia
using ParametricOptInterface, MathOptInterface, HiGHS
# Rename ParametricOptInterface and MathOptInterface to simplify the code
const POI = ParametricOptInterface
const MOI = MathOptInterface
# Define a Optimizer on top of the MOI optimizer
optimizer = POI.Optimizer(HiGHS.Optimizer())
```

### Parameters

A [`ParametricOptInterface.Parameter`](@ref) is a variable with a fixed value that can be changed by the user.

### Adding a new parameter to a model

To add a parameter to a model, we must use the `MOI.add_constrained_variable()` function, passing as its arguments the model and a [`ParametricOptInterface.Parameter`](@ref) with its given value:

```julia
y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
```

### Changing the parameter value

To change a given parameter's value, access its `VariableIndex` and set it to the new value using the [`ParametricOptInterface.Parameter`](@ref) structure.

```julia
MOI.set(optimizer, POI.ParameterValue(), y, POI.Parameter(2.0))
```

### Retrieving the dual of a parameter

Given an optimized model, one can calculate the dual associated to a parameter, **as long as it is an additive term in the constraints or objective**.
One can do so by getting the `MOI.ConstraintDual` attribute of the parameter's `MOI.ConstraintIndex`:

```julia
MOI.get(optimizer, POI.ParameterDual(), y)
```
