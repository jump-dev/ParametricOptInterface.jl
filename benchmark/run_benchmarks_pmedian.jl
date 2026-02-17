# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import HiGHS
import MathOptInterface as MOI
import ParametricOptInterface as POI
import Random
import TimerOutputs

struct PMedianData
    num_facilities::Int
    num_customers::Int
    num_locations::Int
    customer_locations::Vector{Float64}
end

# This is the LP relaxation.
function generate_poi_problem(model, data::PMedianData, add_parameters::Bool)
    NL = data.num_locations
    NC = data.num_customers

    ###
    ### 0 <= facility_variables <= 1
    ###

    facility_variables = MOI.add_variables(model, NL)

    for v in facility_variables
        MOI.add_constraint(model, v, MOI.Interval(0.0, 1.0))
    end

    ###
    ### assignment_variables >= 0
    ###

    assignment_variables = reshape(MOI.add_variables(model, NC * NL), NC, NL)
    for v in assignment_variables
        MOI.add_constraint(model, v, MOI.GreaterThan(0.0))
        # "Less than 1.0" constraint is redundant.
    end

    ###
    ### Objective function
    ###

    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            [
                MOI.ScalarAffineTerm(
                    data.customer_locations[i] - j,
                    assignment_variables[i, j],
                ) for i in 1:NC for j in 1:NL
            ],
            0.0,
        ),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    ###
    ### assignment_variables[i, j] <= facility_variables[j]
    ###

    for i in 1:NC, j in 1:NL
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [
                    MOI.ScalarAffineTerm(1.0, assignment_variables[i, j]),
                    MOI.ScalarAffineTerm(-1.0, facility_variables[j]),
                ],
                0.0,
            ),
            MOI.LessThan(0.0),
        )
    end

    ###
    ### sum_j assignment_variables[i, j] = 1
    ###

    for i in 1:NC
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [
                    MOI.ScalarAffineTerm(1.0, assignment_variables[i, j]) for
                    j in 1:NL
                ],
                0.0,
            ),
            MOI.EqualTo(1.0),
        )
    end

    ###
    ### sum_j facility_variables[j] == num_facilities
    ###

    if add_parameters
        d, cd = MOI.add_constrained_variable(
            model,
            MOI.Parameter{Float64}(data.num_facilities),
        )
    end

    if add_parameters
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.(1.0, vcat(facility_variables, d)),
                0.0,
            ),
            MOI.EqualTo{Float64}(0.0),
        )
    else
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.(1.0, facility_variables),
                0.0,
            ),
            MOI.EqualTo{Float64}(data.num_facilities),
        )
    end

    return assignment_variables, facility_variables
end

function solve_moi(
    data::PMedianData,
    optimizer;
    params,
    add_parameters::Bool = false,
)
    model = MOI.instantiate(optimizer)
    for (param, value) in params
        MOI.set(model, param, value)
    end
    TimerOutputs.@timeit(
        "generate",
        generate_poi_problem(model, data, add_parameters),
    )
    TimerOutputs.@timeit "solve" MOI.optimize!(model)
    return MOI.get(model, MOI.TerminationStatus())
end

function solve_moi_loop(
    optimizer,
    name,
    data::PMedianData;
    max_iters = Inf,
    time_limit_sec = Inf,
    loops,
    kwargs...,
)
    params = Any[(MOI.Silent(), true)]
    if isfinite(time_limit_sec)
        push!(params, (MOI.TimeLimitSec(), time_limit_sec))
    end
    if isfinite(max_iters)
        push!(
            params,
            (MOI.RawOptimizerAttribute("simplex_iteration_limit"), max_iters),
        )
    end
    TimerOutputs.@timeit "$name" for _ in 1:loops
        solve_moi(data, optimizer; params, kwargs...)
    end
    return
end

function run_benchmark(;
    num_facilities,
    num_customers,
    num_locations,
    kwargs...,
)
    Random.seed!(10)
    TimerOutputs.reset_timer!()
    data = PMedianData(
        num_facilities,
        num_customers,
        num_locations,
        rand(num_customers) .* num_locations,
    )
    GC.gc()
    solve_moi_loop(HiGHS.Optimizer, "HiGHS MOI NO PARAMS", data; kwargs...)
    GC.gc()
    solve_moi_loop("HiGHS POI NO PARAMS", data; kwargs...) do
        return POI.Optimizer(HiGHS.Optimizer)
    end
    GC.gc()
    solve_moi_loop("HiGHS POI", data; add_parameters = true, kwargs...) do
        return POI.Optimizer(HiGHS.Optimizer)
    end
    GC.gc()
    TimerOutputs.print_timer()
    return
end

run_benchmark(;
    num_facilities = 100,
    num_customers = 100,
    num_locations = 100,
    time_limit_sec = 0.0001,
    max_iters = 1,
    loops = 1,
)

run_benchmark(;
    num_facilities = 100,
    num_customers = 100,
    num_locations = 100,
    time_limit_sec = 0.0001,
    max_iters = 1,
    loops = 100,
)
