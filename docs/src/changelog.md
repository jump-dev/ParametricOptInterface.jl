```@meta
CurrentModule = ParametricOptInterface
```

# Release notes

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version 0.15.0 (February 19, 2026)

This breaking release removes a number of ParametricOptInterface-specific
features in favor of the officially supported MathOptInterface functions for
dealing with parameters.

### Breaking

- Removed `QuadraticObjectiveCoef` (#213)

  This breaking change removed `QuadraticObjectiveCoef` and all related
  functions.

  To represent a parameter multiplied by a quadratic term, you must now use
  `MOI.ScalarNonlinearFunction` to define a cubic polynomial:
  ```julia
  using JuMP, HiGHS
  import ParametricOptInterface as POI
  model = Model(() -> POI.Optimizer(HiGHS.Ootimizer))
  @variable(model, x)
  @variable(model, p in Parameter(1))
  @objective(model, Min, p * x^2)
  ```

- Removed `ParameterValue` and `ParameterDual` (#219)

  This breaking change removed `ParameterValue` and `ParameterDual` and all
  related functions.

  In JuMP, follow these replacements:
  ```julia
  using JuMP
  model = Model()
  @variable(model, x)
  @variable(model, p in Parameter(1))
  # Replace MOI.get(model, POI.ParameterValue(), p) with
  p_value = parameter_value(p)
  # Replace MOI.set(model, POI.ParameterValue(), p, 2.0) with
  set_parameter_value(p, 2.0)
  # Replace MOI.get(model, POI.ParameterDual(), p) with
  p_dual = dual(ParameterRef(p))
  ```

  In MathOptInterface, follow these replacements:
  ```julia
  import MathOptInterface as MOI
  model = MOI.Utilities.Model{Float64}()
  p, cp in MOI.add_constrained_variable(model, MOI.Parameter(1.0))
  # Replace MOI.get(model, POI.ParameterValue(), p) with
  p_value = MOI.get(model, MOI.ConstraintSet(), cp).value
  # Replace MOI.set(model, POI.ParameterValue(), p, 2.0) with
  MOI.set(model, MOI.ConstraintSet(), cp, MOI.Parameter(2.0))
  # Replace MOI.get(model, POI.ParameterDual(), p) with
  p_dual = MOI.get(model, MOI.ConstraintDual(), cp)
  ```

### Other

- Various changes to improve code style (#214)
- Make all of the test optimizers silent (#215)
- Update tests for SCS.jl@2 (#216)
- Make the explanation a separate page (#217)
- Increase coverage (#218)
- Run the benchmarks in CI to ensure they stay updated (#220)
