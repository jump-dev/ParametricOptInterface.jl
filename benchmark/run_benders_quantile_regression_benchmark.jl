# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using JuMP
using TimerOutputs

import HiGHS
import MathOptInterface as MOI
import ParameterJuMP
import ParametricOptInterface as POI
import Random

const N_Candidates = 200
const N_Observations = 2000
const N_Nodes = 200
const Observations = 1:N_Observations
const Candidates = 1:N_Candidates
const Nodes = 1:N_Nodes
const rng = Random.MersenneTwister(123)
const X = zeros(N_Candidates, N_Observations)
const time = [obs / N_Observations * 1 for obs in Observations]
for obs in Observations, cand in Candidates
    t = time[obs]
    f = cand
    X[cand, obs] = sin(2 * pi * f * t)
end
β = zeros(N_Candidates)
for i in Candidates
    if rand(rng) <= (1 - i / N_Candidates)^2 && i <= 100
        β[i] = 4 * rand(rng) / i
    end
end
const y = X' * β .+ 0.1 * randn(rng, N_Observations)

function full_model_regression()
    time_build = @elapsed begin
        full_model = direct_model(HiGHS.Optimizer)
        set_silent(full_model)
        @variables(full_model, begin
            ɛ_up[Observations] >= 0
            ɛ_dw[Observations] >= 0
            β[1:N_Candidates]
        end)
        @constraints(
            full_model,
            begin
                ɛ_up_ctr[i in Observations],
                ɛ_up[i] >= +sum(X[j, i] * β[j] for j in Candidates) - y[i]
                ɛ_dw_ctr[i in Observations],
                ɛ_dw[i] >= -sum(X[j, i] * β[j] for j in Candidates) + y[i]
            end
        )
        @objective(
            full_model,
            Min,
            sum(ɛ_up[i] + ɛ_dw[i] for i in Observations)
        )
    end
    time_solve = @elapsed optimize!(full_model)
    println(
        "First coefficients in solution: $(value.(β)[1:min(10, N_Candidates)])",
    )
    println("Objective value: $(objective_value(full_model))")
    println("Time in solve: $time_solve")
    println("Time in build: $time_build")
    return
end

function ObsSet(K)
    obs_per_block = div(N_Observations, N_Nodes)
    return (1+(K-1)*obs_per_block):(K*obs_per_block)
end

function slave_model(PARAM, K)
    slave = if PARAM == 0
        Model(HiGHS.Optimizer)
    elseif PARAM == 1
        direct_model(POI.Optimizer(HiGHS.Optimizer))
    else
        direct_model(HiGHS.Optimizer())
    end
    set_silent(slave)
    @variables(slave, begin
        ɛ_up[ObsSet(K)] >= 0
        ɛ_dw[ObsSet(K)] >= 0
    end)
    if PARAM == 0
        @variable(slave, β[1:N_Candidates] == 0, ParameterJuMP.Param())
    elseif PARAM == 1
        @variable(slave, β[1:N_Candidates] in Parameter(0))
    else
        @variables(slave, begin
            β[Candidates]
            β_fixed[1:N_Candidates] == 0
        end)
        @constraint(slave, β_fix[i in Candidates], β[i] == β_fixed[i])
    end
    @constraints(
        slave,
        begin
            ɛ_up_ctr[i in ObsSet(K)],
            ɛ_up[i] >= +sum(X[j, i] * β[j] for j in Candidates) - y[i]
            ɛ_dw_ctr[i in ObsSet(K)],
            ɛ_dw[i] >= -sum(X[j, i] * β[j] for j in Candidates) + y[i]
        end
    )
    @objective(slave, Min, sum(ɛ_up[i] + ɛ_dw[i] for i in ObsSet(K)))
    return (slave, β)
end

function master_model(PARAM)
    master = Model(HiGHS.Optimizer)
    set_silent(master)
    @variables(master, begin
        ɛ[Nodes] >= 0
        β[1:N_Candidates]
    end)
    @objective(master, Min, sum(ɛ))
    return (master, ɛ, β, zeros(N_Candidates))
end

function master_solve(PARAM, master_model)
    model, _, β, _ = master_model
    optimize!(model)
    return (value.(β), objective_value(model))
end

function slave_solve(PARAM, model, master_solution)
    β0, _ = master_solution
    slave, β = model
    @timeit "fix" if PARAM == 0
        set_value.(β, β0)
    elseif PARAM == 1
        set_parameter_value.(β, β0)
    else
        fix.(slave[:β_fixed], β0)
    end
    @timeit "opt" optimize!(slave)
    @timeit "dual" π = if PARAM == 0
        dual.(β)
    elseif PARAM == 1
        dual.(ParameterRef.(β))
    else
        dual.(slave[:β_fix])
    end
    obj = objective_value(slave)
    rhs = obj - π' * β0
    return (rhs, π, obj)
end

function master_add_cut(PARAM, master_model, cut_info, node)
    master, ɛ, β, _ = master_model
    rhs, π, _ = cut_info
    @constraint(master, ɛ[node] >= sum(π[j] * β[j] for j in Candidates) + rhs)
    return
end

function decomposed_model(PARAM; print_timer_outputs::Bool = true)
    reset_timer!()
    time_init = @elapsed @timeit "Init" begin
        @timeit "Master" master = master_model(PARAM)
        @timeit "Sol" solution = (zeros(N_Candidates), Inf)
        best_sol = deepcopy(solution)
        @timeit "Slaves" slaves =
            [slave_model(PARAM, i) for i in Candidates]
        @timeit "Cuts" cuts =
            [slave_solve(PARAM, slaves[i], solution) for i in Candidates]
    end
    LB, UB = -Inf, +Inf
    time_loop = @elapsed @timeit "Loop" for k in 1:80
        @timeit "add cuts" for i in Candidates
            master_add_cut(PARAM, master, cuts[i], i)
        end
        @timeit "solve master" solution = master_solve(PARAM, master)
        @timeit "solve nodes" for i in Candidates
            cuts[i] = slave_solve(PARAM, slaves[i], solution)
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
    if print_timer_outputs
        print_timer()
    end
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

nothing;
