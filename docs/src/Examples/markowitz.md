# Markowitz Efficient Frontier

In this example, we solve the classical portfolio problem where we introduce the
weight parameter $\gamma$ and maximize $\gamma \text{ risk} - \text{expected return}$. By updating the values of $\gamma$ we trace the efficient frontier.

Given the prices changes with mean $\mu$ and covariance $\Sigma$, we can construct the classical portfolio problem:

$$\begin{array}{ll}
     \text{minimize}   & \gamma* x^T \mu - x^T \Sigma x \\
     \text{subject to} & \| x \|_1 = 1 \\
     & x \succeq 0
\end{array}$$

The problem data was gotten from the example [portfolio optimization](https://jump.dev/Convex.jl/dev/examples/portfolio_optimization/portfolio_optimization2/)

```julia
using ParametricOptInterface, MathOptInterface, JuMP, Ipopt
using LinearAlgebra, Plots

const POI = ParametricOptInterface
const MOI = MathOptInterface

# generate problem data
μ = [11.5; 9.5; 6] / 100          #expected returns
Σ = [
    166 34 58              #covariance matrix
    34 64 4
    58 4 100
] / 100^2

```

We first build the model with $\gamma$ as parameter in POI

```julia
function first_model(μ,Σ)
    cached = MOI.Bridges.full_bridge_optimizer(
        MOIU.CachingOptimizer(
            MOIU.UniversalFallback(MOIU.Model{Float64}()),
            Ipopt.Optimizer(),
        ),
        Float64,
    )
    optimizer = POI.Optimizer(cached)
    portfolio = direct_model(optimizer)
    set_silent(portfolio)
    
    N = length(μ)
    @variable(portfolio, x[1:N] >= 0)
    @variable(portfolio, γ in POI.Parameter(0.0))

    @objective(portfolio, Max, γ*dot(μ,x) - x' * Σ * x)
    @constraint(portfolio, sum(x) == 1)
    optimize!(portfolio)

    return portfolio
end
```

Then, we update the $\gamma$ value in the model

```julia
function update_model!(portfolio,γ_value)
    γ = portfolio[:γ]
    MOI.set(portfolio, POI.ParameterValue(), γ, γ_value)
    optimize!(portfolio)
    return portfolio
end
```

Collecting all the return and risk resuls for each $\gamma$

```julia
function add_to_dict(portfolios_values,portfolio,μ,Σ)
    γ = portfolio[:γ]
    γ_value = value(γ)
    x = portfolio[:x]
    x_value = value.(x)
    portfolio_return = dot(μ,x_value)
    portfolio_deviation = x_value' * Σ * x_value
    portfolios_values[γ_value] = (portfolio_return,portfolio_deviation)
end
```

Run the portfolio optimization for different values of $\gamma$

```julia
portfolio = first_model(μ,Σ)
portfolios_values = Dict()
add_to_dict(portfolios_values,portfolio,μ,Σ)

for γ_value in 0.02:0.02:1.0
    global portfolio = update_model!(portfolio,γ_value)
    add_to_dict(portfolios_values,portfolio,μ,Σ)
end
```

Plot the efficient frontier

```julia
portfolios_values = sort(portfolios_values,by=x->x[1])
portfolios_values_matrix = hcat([[v[1],v[2]] for v in values(portfolios_values)]...)'
plot(portfolios_values_matrix[:,2],portfolios_values_matrix[:,1],legend=false,
xlabel="Standard Deviation", ylabel = "Return", title = "Efficient Frontier")
```