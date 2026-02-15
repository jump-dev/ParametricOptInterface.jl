# Markowitz Efficient Frontier

In this example, we solve the classical portfolio problem where we introduce the
weight parameter $\gamma$ and maximize $\gamma \text{ risk} - \text{expected return}$.

By updating the values of $\gamma$ we trace the efficient frontier.

Given the prices changes with mean $\mu$ and covariance $\Sigma$, we can
construct the classical portfolio problem:

$$\begin{array}{ll}
     \text{maximize}   & \gamma* x^T \mu - x^T \Sigma x \\
     \text{subject to} & \| x \|_1 = 1 \\
     & x \succeq 0
\end{array}$$

The problem data was gotten from the example [portfolio optimization](https://jump.dev/Convex.jl/dev/examples/portfolio_optimization/portfolio_optimization2/)

```@repl markowitz
using JuMP
import Ipopt
import ParametricOptInterface as POI
import Plots
μ = [11.5; 9.5; 6] / 100
Σ = [166 34 58; 34 64 4; 58 4 100] / 100^2
```

We first build the model with $\gamma$ as parameter in POI
```@repl markowitz
function first_model(μ, Σ)
    portfolio = Model() do
        inner = MOI.instantiate(Ipopt.Optimizer; with_cache_type = Float64)
        return POI.Optimizer(inner)
    end
    set_silent(portfolio)
    N = length(μ)
    @variable(portfolio, x[1:N] >= 0)
    @variable(portfolio, γ in Parameter(0.0))
    @objective(portfolio, Max, γ * μ' * x - x' * Σ * x)
    @constraint(portfolio, sum(x) == 1)
    optimize!(portfolio)
    return portfolio
end
```

Then, we update the $\gamma$ value in the model

```@repl markowitz
function update_model!(portfolio, γ_value)
    γ = portfolio[:γ]
    set_parameter_value(γ, γ_value)
    optimize!(portfolio)
    return portfolio
end
```

Collecting all the return and risk resuls for each $\gamma$

```@repl markowitz
function add_to_dict(portfolios_values, portfolio, μ, Σ)
    γ = portfolio[:γ]
    γ_value = value(γ)
    x = portfolio[:x]
    x_value = value.(x)
    portfolio_return = μ' * x_value
    portfolio_deviation = x_value' * Σ * x_value
    portfolios_values[γ_value] = (portfolio_return, portfolio_deviation)
    return
end
```

Run the portfolio optimization for different values of $\gamma$

```@repl markowitz
portfolio = first_model(μ, Σ)
portfolios_values = Dict()
# Create a reference to the model to change it later
portfolio_ref = [portfolio]
add_to_dict(portfolios_values, portfolio, μ, Σ)
for γ_value in 0.02:0.02:1.0
    portfolio_ref[] = update_model!(portfolio_ref[], γ_value)
    add_to_dict(portfolios_values,portfolio_ref[], μ, Σ)
end
```

Plot the efficient frontier

```@repl markowitz
sort!(portfolios_values, by = first)
portfolios_values_matrix =
    hcat([[v[1], v[2]] for v in values(portfolios_values)]...)'
Plots.plot(
    portfolios_values_matrix[:,2],
    portfolios_values_matrix[:,1];
    legend = false,
    xlabel = "Standard Deviation",
    ylabel = "Return",
    title = "Efficient Frontier",
)
```
