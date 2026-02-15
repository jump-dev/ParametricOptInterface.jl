# Progressive Hedging

Progressive Hedging (PH) is a popular decomposition algorithm for stochastic programming. It decomposes a stochastic problem into scenario subproblems that are solved iteratively, with penalty terms driving solutions toward consensus. POI is well-suited for PH because the penalty parameters and target values can be updated efficiently without rebuilding the model.

## Background

In progressive hedging, each scenario subproblem includes a quadratic penalty term:

```
minimize: f_s(x) + (ρ/2) * ||x - x̄||² + w' * x
```

where:
- `f_s(x)` is the original scenario objective
- `ρ` is the penalty parameter
- `x̄` is the current consensus (average) solution
- `w` is the dual price (Lagrangian multiplier)

The penalty term `(ρ/2) * (x - x̄)²` expands to `(ρ/2) * x² - ρ * x̄ * x + (ρ/2) * x̄²`. Using POI parameters for `ρ` and `x̄` allows efficient updates between PH iterations.

## Simple Example: Two-Stage Stochastic Program

Consider a simple production planning problem with two scenarios:

```julia
using JuMP, HiGHS
import ParametricOptInterface as POI

# Problem data
scenarios = [
    (demand = 100, probability = 0.4),
    (demand = 150, probability = 0.6),
]
production_cost = 10
penalty_cost = 25  # unmet demand penalty

# PH parameters
ρ = 1.0  # penalty parameter
max_iterations = 20
tolerance = 1e-4

# Build scenario subproblems with POI
function build_subproblem(scenario, ρ_init, x_bar_init, w_init)
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    # First-stage variable (production quantity to decide before demand is known)
    @variable(model, x >= 0)

    # Second-stage variable (unmet demand)
    @variable(model, y >= 0)

    # Parameters for PH updates
    @variable(model, ρ_param in MOI.Parameter(ρ_init))
    @variable(model, x_bar in MOI.Parameter(x_bar_init))
    @variable(model, w in MOI.Parameter(w_init))

    # Demand satisfaction constraint
    @constraint(model, x + y >= scenario.demand)

    # Objective: original cost + PH penalty terms
    @objective(model, Min,
        production_cost * x + penalty_cost * y +  # original objective
        w * x +                                   # dual price term
        0.5 * ρ_param * (x - x_bar)^2             # quadratic penalty
    )

    return model, x, ρ_param, x_bar, w
end

# Initialize subproblems
subproblems = []
x_vars = []
ρ_params = []
x_bar_params = []
w_params = []

for s in scenarios
    model, x, ρ_p, x_bar_p, w_p = build_subproblem(s, ρ, 0.0, 0.0)
    push!(subproblems, model)
    push!(x_vars, x)
    push!(ρ_params, ρ_p)
    push!(x_bar_params, x_bar_p)
    push!(w_params, w_p)
end

# Progressive Hedging iterations
x_bar = 0.0
w_values = zeros(length(scenarios))

for iter in 1:max_iterations
    # Solve all subproblems
    x_values = Float64[]
    for (i, model) in enumerate(subproblems)
        optimize!(model)
        push!(x_values, value(x_vars[i]))
    end

    # Compute new consensus (probability-weighted average)
    x_bar_new = sum(scenarios[i].probability * x_values[i] for i in eachindex(scenarios))

    # Check convergence
    max_deviation = maximum(abs.(x_values .- x_bar_new))
    println("Iteration $iter: x̄ = $(round(x_bar_new, digits=2)), max deviation = $(round(max_deviation, digits=4))")

    if max_deviation < tolerance
        println("Converged!")
        break
    end

    # Update dual prices
    for i in eachindex(scenarios)
        w_values[i] += ρ * (x_values[i] - x_bar_new)
    end

    # Update parameters for next iteration (this is where POI shines!)
    # Parameters are automatically updated when optimize! is called
    for i in eachindex(scenarios)
        set_parameter_value(x_bar_params[i], x_bar_new)
        set_parameter_value(w_params[i], w_values[i])
    end

    x_bar = x_bar_new
end

println("\nFinal consensus solution: x̄ = $(round(x_bar, digits=2))")
```

## Why POI for Progressive Hedging?

1. **Efficient updates**: Parameters `x̄` and `w` change every iteration. Without POI, you would need to rebuild the model or modify constraints manually.

2. **Quadratic penalties**: The term `ρ * x̄ * x` is a parameter times a variable, which POI handles natively. This creates the cross-term needed for the quadratic penalty.

3. **Warm starting**: Since the model structure is preserved, solvers can warm-start from the previous solution, significantly speeding up convergence.

4. **Clean separation**: The scenario-specific data stays fixed while PH-specific parameters are clearly identified and updated.

## Advanced: Adaptive Penalty Parameter

You can also make the penalty parameter `ρ` adaptive:

```julia
# Increase penalty if not converging fast enough
if iter > 5 && max_deviation > prev_deviation * 0.95
    ρ *= 1.5
    for i in eachindex(scenarios)
        set_parameter_value(ρ_params[i], ρ)
    end
end
```

This showcases POI's flexibility: both the consensus target and the penalty strength can be parameters that evolve during the algorithm.
