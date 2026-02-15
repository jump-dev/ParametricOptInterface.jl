```@meta
CurrentModule = ParametricOptInterface
```

# Manual

## Why use parameters?

A typical optimization model built using `MathOptInterface.jl` (`MOI`for short)
has two main components:

1. Variables
2. Constants

Using these basic elements, one can create functions and sets that, together,
form the desired optimization model. The goal of `POI` is the implementation of
a third type, parameters, which:

* are declared similar to a variable, and inherits some functionalities (for
  example, dual calculation)
* acts like a constant, in the sense that it has a fixed value that will remain
  the same unless explicitly changed by the user

A main concern is to efficiently implement this new type, as one typical usage
is to change its value to analyze the model behavior, without the need to build
a new one from scratch.

## How it works

The main idea applied in POI is that the interaction between the solver, for
example `HiGHS`, and the optimization model will be handled by `MOI` as usual.

Because of that, `POI` is a higher level wrapper around `MOI`, responsible for
receiving variables, constants and parameters, and forwarding to the lower level
model only variables and constants.

As `POI` receives parameters, it must analyze and decide how they should be
handled on the lower level optimization model (the `MOI` model).

## Usage

In this manual we describe how to interact with the optimization model at the
MOI level. In the _Examples_ section you can find some tutorials with the
JuMP usage.

### Supported constraints

This is a list of supported `MOI` constraint functions that can handle
parameters. If you try to add a parameter to a function that is not listed here,
it will return an unsupported error.

| MOI Function              |
| :------------------------ |
| `ScalarAffineFunction`    |
| `ScalarQuadraticFunction` |
| `VectorAffineFunction`    |


### Supported objective functions

| MOI Function                                       |
| :------------------------------------------------- |
| `ScalarAffineFunction`                             |
| `ScalarQuadraticFunction`                          |
| `ScalarNonlinearFunction` (cubic polynomials only) |

### Declare a Optimizer

In order to use parameters, the user needs to declare an
[`Optimizer`](@ref) on top of a `MOI` optimizer, such as `HiGHS.Optimizer()`.

```@repl manual
import ParametricOptInterface as POI
import HiGHS
optimizer = POI.Optimizer(HiGHS.Optimizer())
```

### Parameters

A `MOI.Parameter` is a set used to define a variable with a fixed value that
can be changed by the user. It is analogous to `MOI.EqualTo`, but can be used
by special methods like the ones in this package to remove the fixed variable
from the optimization problem. This permits the usage of multiplicative
parameters in linear models and might speedup solves since the number of
variables is reduced.

### Adding a new parameter to a model

To add a parameter to a model, we must use the `MOI.add_constrained_variable`
function, passing as its arguments the model and a `MOI.Parameter` with its
given value:

```@repl manual
y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
```

### Changing the parameter value

To change a given parameter's value, access its `VariableIndex` and set it to
the new value using the `MOI.Parameter` structure.

```@repl manual
MOI.set(optimizer, POI.ParameterValue(), y, MOI.Parameter(2.0))
```

### Retrieving the dual of a parameter

Given an optimized model, one can compute the dual associated to a parameter,
**as long as it is an additive term in the constraints or objective**.

One can do so by getting the `MOI.ConstraintDual` attribute of the parameter's
`MOI.ConstraintIndex`:
```julia
julia> MOI.get(optimizer, POI.ParameterDual(), y)
```

### Parameters multiplying quadratic terms

POI supports parameters that multiply quadratic variable terms in objectives
**only**. This creates cubic polynomial expressions of the form `c * p * x * y`
where `c` is a number, `p` is a parameter and `x`, `y` are variables. After
parameter substitution, these become standard quadratic terms that solvers can
handle.


#### Attention

- Maximum degree is 3 (cubic)
- At least one factor in each cubic term must be a parameter
- Pure cubic variable terms (e.g., `x * y * z` with no parameters) are
  **not supported**

#### Example using JuMP

```julia
using JuMP, Ipopt
import ParametricOptInterface as POI

# Create model with POI optimizer
model = Model(() -> POI.Optimizer(Ipopt.Optimizer()))
set_silent(model)

# Define variables and parameter
@variable(model, 0 <= x <= 10)
@variable(model, p in MOI.Parameter(2.0))

# Set cubic objective: p * x * y
@objective(model, Min, p * x ^ 2 - 3x)

# p (x - 3) * (x - 2)

p x^2 - c x

2px - c

x = c / 2p

# Solve
optimize!(model)

@show value(x) # == 3 / (2 * p)

# Update parameter and re-solve
# Parameters are automatically updated when optimize! is called
set_parameter_value(p, 3.0)
optimize!(model)

@show value(x) # == 3 / (2 * p)
```
