# Benders Quantile Regression

We will apply Norm-1 regression to the [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression) problem.
Linear regression is a statistical tool to obtain the relation between one **dependent variable** and other **explanatory variables**.
In other words, given a set of $n$ explanatory variables $X = \{ X_1, \dots, X_n \}$
we would like to obtain the best possible estimate for $Y$.
In order to accomplish such a task we make the hypothesis that $Y$
is approximately linear function of $X$:

$$Y = \sum_{j =1}^n \beta_j X_j + \varepsilon$$

where $\varepsilon$ is some random error.

The estimation of the $\beta$ values relies on observations of the variables:
$\{y^i, x_1^i, \dots, x_n^i\}_i$.

In this example we will solve a problem where the explanatory variables are sinusoids of differents frequencies.
First, we define the number of explanatory variables and observations

```julia
using ParametricOptInterface,MathOptInterface,JuMP,GLPK
using TimerOutputs,LinearAlgebra,Random

const POI = ParametricOptInterface
const MOI = MathOptInterface
const OPTIMIZER = GLPK.Optimizer;

const N_Candidates = 200
const N_Observations = 2000
const N_Nodes = 200

const Observations = 1:N_Observations
const Candidates = 1:N_Candidates
const Nodes = 1:N_Nodes;
```


Initialize a random number generator to keep results deterministic
```julia
rng = Random.MersenneTwister(123);
```

Building regressors (explanatory) sinusoids

```julia
const X = zeros(N_Candidates, N_Observations)
const time = [obs / N_Observations * 1 for obs in Observations]
for obs in Observations, cand in Candidates
    t = time[obs]
    f = cand
    X[cand, obs] = sin(2 * pi * f * t)
end
```

Define coefficients

```julia
β = zeros(N_Candidates)
for i in Candidates
    if rand(rng) <= (1 - i / N_Candidates)^2 && i <= 100
        β[i] = 4 * rand(rng) / i
    end
end
```

Create noisy observations

```julia
const y = X' * β .+ 0.1 * randn(rng, N_Observations)
```

### Benders Decomposition

Benders decomposition is used to solve large optimization problems with some special characteristics.
LP's can be solved with classical linear optimization methods
such as the Simplex method or Interior point methods provided by
solvers like GLPK.
However, these methods do not scale linearly with the problem size.
In the Benders decomposition framework we break the problem in two pieces:
A outer and a inner problem.
Of course some variables will belong to both problems, this is where the
cleverness of Benders kicks in:
The outer problem is solved and passes the shared variables to the inner.
The inner problem is solved with the shared variables FIXED to the values
given by the outer problem. The solution of the inner problem can be used
to generate a constraint to the outer problem to describe the linear
approximation of the cost function of the shared variables.
In many cases, like stochastic programming, the inner problems have a
interesting structure and might be broken in smaller problem to be solved
in parallel.

We will descibe the decomposition similarly to what is done in:
Introduction to Linear Optimization, Bertsimas & Tsitsiklis (Chapter 6.5):
Where the problem in question has the form

$$\begin{align}
   & \min_{x, y_k}     &&  c^T x && + f_1^T y_1 && + \dots && + f_n^T y_n  &&  \notag \\
   & \text{subject to} &&  Ax    &&             &&         &&              && = b \notag \\
   &                   &&  B_1 x && + D_1 y_1   &&         &&              && = d_1 \notag \\
   &                   &&  \dots &&             &&  \dots  &&              &&       \notag \\
   &                   &&  B_n x &&             &&         && + D_n y_n    && = d_n \notag \\
   &                   &&   x,   &&     y_1,    &&         &&       y_n    && \geq 0 \notag \\
\end{align}$$

### Inner Problem

Given a solution for the $x$ variables we can define the inner problem as

$$\begin{align}
 z_k(x) \ = \ & \min_{y_k}        &&  f_k^T y_k &&                  \notag \\
              & \text{subject to} &&  D_k y_k   &&  = d_k - B_k x \notag \\
              &                   &&  y_k       && \geq 0          \notag \\
\end{align}$$

The $z_k(x)$ function represents the cost of the subproblem given a
solution for $x$. This function is a convex function because $x$
affects only the right hand side of the problem (this is a standard
results in LP theory).

For the special case of the Norm-1 reggression the problem is written as:

$$\begin{align}
z_k(\beta) \ = \ & \min_{\varepsilon^{up}, \varepsilon^{dw}}         &&  \sum_{i \in ObsSet(k)} {\varepsilon^{up}}_i + {\varepsilon^{dw}}_i               && \notag \\
                 & \text{subject to}     &&  {\varepsilon^{up}}_i \geq + y_i - \sum_{j \in Candidates} \beta_j x_{i,j} && \forall i \in ObsSet(k) \notag \\
                 &                       &&  {\varepsilon^{dw}}_i \geq - y_i + \sum_{j \in Candidates} \beta_j x_{i,j} && \forall i \in ObsSet(k) \notag \\
                 &                       &&  {\varepsilon^{up}}_i, {\varepsilon^{dw}}_i \geq 0                             && \forall i \in ObsSet(k) \notag \\
\end{align}$$

The collection $ObsSet(k)$ is a sub-set of the `N_Observations`.
Any partition of the `N_Observations` collection is valid.
In this example we will partition with the function:

```julia
function ObsSet(K)
    obs_per_block = div(N_Observations, N_Nodes)
    return (1+(K-1)*obs_per_block):(K*obs_per_block)
end
```

Which can be written in POI as follows:

```julia
function inner_model(K)

    # initialize the POI model
    inner = direct_model(POI.Optimizer(OPTIMIZER()))

    # Define local optimization variables for norm-1 error
    @variables(inner, begin
        ɛ_up[ObsSet(K)] >= 0
        ɛ_dw[ObsSet(K)] >= 0
    end)

    # create the regression coefficient representation
    # Create parameters
    β = [@variable(inner, set = POI.Parameter(0)) for i in 1:N_Candidates]
    for (i, βi) in enumerate(β)
        set_name(βi, "β[$i]")
    end

    # create local constraints
    # Note that *parameter* algebra is implemented just like variables
    # algebra. We can multiply parameters by constants, add parameters,
    # sum parameters and variables and so on.
    @constraints(
        inner,
        begin
            ɛ_up_ctr[i in ObsSet(K)],
            ɛ_up[i] >= +sum(X[j, i] * β[j] for j in Candidates) - y[i]
            ɛ_dw_ctr[i in ObsSet(K)],
            ɛ_dw[i] >= -sum(X[j, i] * β[j] for j in Candidates) + y[i]
        end
    )

    # create local objective function
    @objective(inner, Min, sum(ɛ_up[i] + ɛ_dw[i] for i in ObsSet(K)))

    # return the correct group of parameters
    return (inner, β)
end
```

### Outer Problem

Now that all pieces of the original problem can be representad by
the convex $z_k(x)$ functions we can recast the problem in the the equivalent form:

$$\begin{align}
  & \min_{x}          &&  c^T x + z_1(x) + \dots + z_n(x) && \notag \\
  & \text{subject to} &&  Ax = b                          && \notag \\
  &                   &&  x \geq 0                        && \notag \\
\end{align}$$

However we cannot pass a problem in this form to a linear programming
solver (it could be passed to other kinds of solvers).

Another standart result of optimization theory is that a convex function
can be represented by its supporting hyper-planes:

$$\begin{align}
  z_k(x) \ = \ & \min_{z, x}       &&  z && \notag \\
               & \text{subject to} &&  z \geq \pi_k(\hat{x}) (x - \hat{x}) + z_k(\hat{x}), \ \forall \hat{x} \in dom(z_k) && \notag \\
\end{align}$$

Then we can re-write (again) the outer problem as

$$\begin{align}
  & \min_{x, z_k}     &&  c^T x + z_1 + \dots + z_n \notag \\
  & \text{subject to} &&  z_i \geq \pi_i(\hat{x}) (x - \hat{x}) + z_i(\hat{x}), \ \forall \hat{x} \in dom(z_i), i \in \{1, \dots, n\} \notag \\
  &                   &&  Ax = b \notag \\
  &                   &&  x \geq 0 \notag \\
\end{align}$$

Which is a linear program! However, it has infinitely many constraints !!

We can relax the infinite constraints and write:

$$\begin{align}
  & \min_{x, z_k}     &&  c^T x + z_1 + \dots + z_n \notag \\
  & \text{subject to} &&  Ax = b \notag \\
  &                   &&  x \geq 0 \notag \\
\end{align}$$

But now its only an underestimated problem.
In the case of our problem it can be written as:

$$\begin{align}
  & \min_{\varepsilon, \beta} &&  \sum_{i \in Nodes} \varepsilon_i \notag \\
  & \text{subject to} &&  \varepsilon_i \geq 0 \notag \\
\end{align}$$

This model can be written in JuMP:

```julia
function outer_model()
    outer = Model(OPTIMIZER)
    @variables(outer, begin
        ɛ[Nodes] >= 0
        β[1:N_Candidates]
    end)
    @objective(outer, Min, sum(ɛ[i] for i in Nodes))
    sol = zeros(N_Candidates)
    return (outer, ɛ, β, sol)
end
```

The method to solve the outer problem and query its solution is given here:

```julia
function outer_solve(outer_model)
    model = outer_model[1]
    β = outer_model[3]
    optimize!(model)
    return (value.(β), objective_value(model))
end
```

### Supporting Hyperplanes

With these building blocks in hand, we can start building the algorithm.
So far we know how to:
- Solve the relaxed outer problem
- Obtain the solution for the $\hat{x}$ (or $\beta$ in our case)


Now we can:
- Fix the values of $\hat{x}$ in the inner problems
- Solve the inner problems
- query the solution of the inner problems to obtain the supporting hyperplane

the value of $z_k(\hat{x})$, which is the objective value of the inner problem

and the derivative $\pi_k(\hat{x}) = \frac{d z_k(x)}{d x} \Big|_{x = \hat{x}}$
The derivative is the dual variable associated to the variable $\hat{x}$,
which results by applying the chain rule on the constraints duals.
These new steps are executed by the function:

```julia
function inner_solve(model, outer_solution)
    β0 = outer_solution[1]
    inner = model[1]

    # The first step is to fix the values given by the outer problem
    @timeit "fix" begin
        β = model[2]
        MOI.set.(inner, POI.ParameterValue(), β, β0)
    end

    # here the inner problem is solved
    @timeit "opt" optimize!(inner)

    # query dual variables, which are sensitivities
    # They represent the subgradient (almost a derivative)
    # of the objective function for infinitesimal variations
    # of the constants in the linear constraints
    # POI: we can query dual values of *parameters*
    π = MOI.get.(inner, POI.ParameterDual(), β)

    # π2 = shadow_price.(β_fix)
    obj = objective_value(inner)
    rhs = obj - dot(π, β0)
    return (rhs, π, obj)
end
```

Now that we have cutting plane in hand we can add them to the outer problem

```julia
function outer_add_cut(outer_model, cut_info, node)
    outer = outer_model[1]
    ɛ = outer_model[2]
    β = outer_model[3]

    rhs = cut_info[1]
    π = cut_info[2]

    @constraint(outer, ɛ[node] >= sum(π[j] * β[j] for j in Candidates) + rhs)
end
```

### Algorithm wrap up

The complete algorithm is

- Solve the relaxed master problem
- Obtain the solution for the $\hat{x}$ (or $\beta$ in our case)
- Fix the values of $\hat{x}$ in the slave problems
- Solve the slave problem
- query the solution of the slave problem to obtain the supporting hyperplane
- add hyperplane to master problem
- repeat

Now we grab all the pieces that we built and we write the benders
algorithm by calling the above function in a proper order.

The macros `@timeit` are use to time each step of the algorithm.

```julia
function decomposed_model(;print_timer_outputs::Bool = true)
    reset_timer!() # reset timer fo comparision
    time_init = @elapsed @timeit "Init" begin
        # Create the outer problem with no cuts
        @timeit "outer" outer = outer_model()

        # initialize solution for the regression coefficients in zero
        @timeit "Sol" solution = (zeros(N_Candidates), Inf)
        best_sol = deepcopy(solution)

        # Create the inner problems
        @timeit "inners" inners =
            [inner_model(i) for i in Candidates]

        # Save initial version of the inner problems and create
        # the first set of cuts
        @timeit "Cuts" cuts =
            [inner_solve(inners[i], solution) for i in Candidates]
    end

    UB = +Inf
    LB = -Inf

    # println("Initialize Iterative step")
    time_loop = @elapsed @timeit "Loop" for k in 1:80

        # Add cuts generated from each inner problem to the outer problem
        @timeit "add cuts" for i in Candidates
            outer_add_cut(outer, cuts[i], i)
        end

        # Solve the outer problem with the new set of cuts
        # Obtain new solution candidate for the regression coefficients
        @timeit "solve outer" solution = outer_solve( outer)

        # Pass the new candidate solution to each of the inner problems
        # Solve the inner problems and obtain cutting planes
        @timeit "solve nodes" for i in Candidates
            cuts[i] = inner_solve( inners[i], solution)
        end

        LB = solution[2]
        new_UB = sum(cuts[i][3] for i in Candidates)
        if new_UB <= UB
            best_sol = deepcopy(solution)
        end
        UB = min(UB, new_UB)

        if abs(UB - LB) / (abs(UB) + abs(LB)) < 0.05
            break
        end
    end

    print_timer_outputs && print_timer()

    return best_sol[1]
end
```

Run benders decomposition with POI

```julia
β2 = decomposed_model(; print_timer_outputs = false);
GC.gc()
β2 = decomposed_model();
```