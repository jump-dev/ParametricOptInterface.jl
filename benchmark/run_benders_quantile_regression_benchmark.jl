# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using ParametricOptInterface
using MathOptInterface

const POI = ParametricOptInterface
const MOI = MathOptInterface

using ParameterJuMP
using JuMP

using GLPK
const OPTIMIZER = GLPK.Optimizer
using TimerOutputs

using LinearAlgebra
using Random

const N_Candidates = 200
const N_Observations = 2000
const N_Nodes = 200

const Observations = 1:N_Observations
const Candidates = 1:N_Candidates
const Nodes = 1:N_Nodes;

#' Initialize a random number generator to keep results deterministic

rng = Random.MersenneTwister(123);

#' Building regressors (explanatory) sinusoids

const X = zeros(N_Candidates, N_Observations)
const time = [obs / N_Observations * 1 for obs in Observations]
for obs in Observations, cand in Candidates
    t = time[obs]
    f = cand
    X[cand, obs] = sin(2 * pi * f * t)
end

#' Define coefficients

β = zeros(N_Candidates)
for i in Candidates
    if rand(rng) <= (1 - i / N_Candidates)^2 && i <= 100
        β[i] = 4 * rand(rng) / i
    end
end
# println("First coefs: $(β[1:min(10, N_Candidates)])")

const y = X' * β .+ 0.1 * randn(rng, N_Observations)

function full_model_regression()
    time_build = @elapsed begin # measure time to create a model

        # initialize a optimization model
        full_model = direct_model(OPTIMIZER)
        MOI.set(full_model, MOI.Silent(), true)

        # create optimization variables of the problem
        @variables(full_model, begin
            ɛ_up[Observations] >= 0
            ɛ_dw[Observations] >= 0
            β[1:N_Candidates]
            # 0 <= β[Candidates] <= 8
        end)

        # define constraints of the model
        @constraints(
            full_model,
            begin
                ɛ_up_ctr[i in Observations],
                ɛ_up[i] >= +sum(X[j, i] * β[j] for j in Candidates) - y[i]
                ɛ_dw_ctr[i in Observations],
                ɛ_dw[i] >= -sum(X[j, i] * β[j] for j in Candidates) + y[i]
            end
        )

        # construct the objective function to be minimized
        @objective(
            full_model,
            Min,
            sum(ɛ_up[i] + ɛ_dw[i] for i in Observations)
        )
    end

    # solve the problem
    time_solve = @elapsed optimize!(full_model)

    println(
        "First coefficients in solution: $(value.(β)[1:min(10, N_Candidates)])",
    )
    println("Objective value: $(objective_value(full_model))")
    println("Time in solve: $time_solve")
    println("Time in build: $time_build")

    return nothing
end

function ObsSet(K)
    obs_per_block = div(N_Observations, N_Nodes)
    return (1+(K-1)*obs_per_block):(K*obs_per_block)
end

function slave_model(PARAM, K)

    # initialize the JuMP model
    slave = if PARAM == 0
        # special constructor exported by ParameterJuMP
        # to add the functionality to the model
        Model(OPTIMIZER)
    elseif PARAM == 1
        # POI constructor
        direct_model(POI.Optimizer(OPTIMIZER()))
        # Model(() -> POI.ParametricOptimizer(OPTIMIZER()))
    else
        # regular JuMP constructor
        direct_model(OPTIMIZER())
    end
    MOI.set(slave, MOI.Silent(), true)

    # Define local optimization variables for norm-1 error
    @variables(slave, begin
        ɛ_up[ObsSet(K)] >= 0
        ɛ_dw[ObsSet(K)] >= 0
    end)

    # create the regression coefficient representation
    if PARAM == 0
        # here is the main constructor of the Parameter JuMP packages
        # It will create model *parameters* instead of variables
        # Variables are added to the optimization model, while parameters
        # are not. Parameters are merged with LP problem constants and do not
        # increase the model dimensions.
        @variable(slave, β[i=1:N_Candidates] == 0, Param())
    elseif PARAM == 1
        # Create parameters
        @variable(slave, β[i=1:N_Candidates] in MOI.Parameter.(0.0))
    else
        # Create fixed variables
        @variables(slave, begin
            β[Candidates]
            β_fixed[1:N_Candidates] == 0
        end)
        @constraint(slave, β_fix[i in Candidates], β[i] == β_fixed[i])
    end

    # create local constraints
    # Note that *parameter* algebra is implemented just like variables
    # algebra. We can multiply parameters by constants, add parameters,
    # sum parameters and variables and so on.
    @constraints(
        slave,
        begin
            ɛ_up_ctr[i in ObsSet(K)],
            ɛ_up[i] >= +sum(X[j, i] * β[j] for j in Candidates) - y[i]
            ɛ_dw_ctr[i in ObsSet(K)],
            ɛ_dw[i] >= -sum(X[j, i] * β[j] for j in Candidates) + y[i]
        end
    )

    # create local objective function
    @objective(slave, Min, sum(ɛ_up[i] + ɛ_dw[i] for i in ObsSet(K)))

    # return the correct group of parameters
    return (slave, β)
end

function master_model(PARAM)
    master = Model(OPTIMIZER)
    @variables(master, begin
        ɛ[Nodes] >= 0
        β[1:N_Candidates]
    end)
    @objective(master, Min, sum(ɛ[i] for i in Nodes))
    sol = zeros(N_Candidates)
    return (master, ɛ, β, sol)
end

function master_solve(PARAM, master_model)
    model = master_model[1]
    β = master_model[3]
    optimize!(model)
    return (value.(β), objective_value(model))
end

function slave_solve(PARAM, model, master_solution)
    β0 = master_solution[1]
    slave = model[1]

    # The first step is to fix the values given by the master problem
    @timeit "fix" if PARAM == 0
        # ParameterJuMP: *parameters* can be set to new values and the optimization
        # model will be automatically updated
        β = model[2]
        set_value.(β, β0)
    elseif PARAM == 1
        # POI: assigning new values to *parameters* and the optimization
        # model will be automatically updated
        β = model[2]
        MOI.set.(slave, POI.ParameterValue(), β, β0)
    else
        # JuMP: it is also possible to fix variables to new values
        β_fixed = slave[:β_fixed]
        fix.(β_fixed, β0)
    end

    # here the slave problem is solved
    @timeit "opt" optimize!(slave)

    # query dual variables, which are sensitivities
    # They represent the subgradient (almost a derivative)
    # of the objective function for infinitesimal variations
    # of the constants in the linear constraints
    @timeit "dual" if PARAM == 0
        # ParameterJuMP: we can query dual values of *parameters*
        π = dual.(β)
    elseif PARAM == 1
        # POI: we can query dual values of *parameters*
        π = MOI.get.(slave, POI.ParameterDual(), β)
    else
        # or, in pure JuMP, we query the duals form
        # constraints that fix the values of our regression
        # coefficients
        π = dual.(slave[:β_fix])
    end

    # π2 = shadow_price.(β_fix)
    # @show sum(π .- π2)
    obj = objective_value(slave)
    rhs = obj - dot(π, β0)
    return (rhs, π, obj)
end

function master_add_cut(PARAM, master_model, cut_info, node)
    master = master_model[1]
    ɛ = master_model[2]
    β = master_model[3]

    rhs = cut_info[1]
    π = cut_info[2]

    @constraint(master, ɛ[node] >= sum(π[j] * β[j] for j in Candidates) + rhs)
end
function decomposed_model(PARAM; print_timer_outputs::Bool = true)
    reset_timer!() # reset timer fo comparision
    time_init = @elapsed @timeit "Init" begin
        # println("Initialize decomposed model")

        # Create the master problem with no cuts
        # println("Build master problem")
        @timeit "Master" master = master_model(PARAM)

        # initialize solution for the regression coefficients in zero
        # println("Build initial solution")
        @timeit "Sol" solution = (zeros(N_Candidates), Inf)
        best_sol = deepcopy(solution)

        # Create the slave problems
        # println("Build slave problems")
        @timeit "Slaves" slaves =
            [slave_model(PARAM, i) for i in Candidates]

        # Save initial version of the slave problems and create
        # the first set of cuts
        # println("Build initial cuts")
        @timeit "Cuts" cuts =
            [slave_solve(PARAM, slaves[i], solution) for i in Candidates]
    end

    UB = +Inf
    LB = -Inf

    # println("Initialize Iterative step")
    time_loop = @elapsed @timeit "Loop" for k in 1:80

        # Add cuts generated from each slave problem to the master problem
        @timeit "add cuts" for i in Candidates
            master_add_cut(PARAM, master, cuts[i], i)
        end

        # Solve the master problem with the new set of cuts
        # Obtain new solution candidate for the regression coefficients
        @timeit "solve master" solution = master_solve(PARAM, master)

        # Pass the new candidate solution to each of the slave problems
        # Solve the slave problems and obtain cutting planes
        # @show solution[2]
        @timeit "solve nodes" for i in Candidates
            cuts[i] = slave_solve(PARAM, slaves[i], solution)
        end

        LB = solution[2]
        new_UB = sum(cuts[i][3] for i in Candidates)
        if new_UB <= UB
            best_sol = deepcopy(solution)
        end
        UB = min(UB, new_UB)
        # println("Iter = $k, LB = $LB, UB = $UB")

        if abs(UB - LB) / (abs(UB) + abs(LB)) < 0.05
            # println("Converged!")
            break
        end
    end
    # println(
    #     "First coefficients in solution: $(solution[1][1:min(10, N_Candidates)])",
    # )
    # println("Objective value: $(solution[2])")
    # println("Time in loop: $time_loop")
    # println("Time in init: $time_init")

    print_timer_outputs && print_timer()

    return best_sol[1]
end

println("ParameterJuMP")
GC.gc()
β1 = decomposed_model(0; print_timer_outputs = false);
GC.gc()
β1 = decomposed_model(0);

println("POI, direct mode")
GC.gc()
β2 = decomposed_model(1; print_timer_outputs = false);
GC.gc()
β2 = decomposed_model(1);

println("Pure JuMP, direct mode")
GC.gc()
β3 = decomposed_model(2; print_timer_outputs = false);
GC.gc()
β3 = decomposed_model(2);
