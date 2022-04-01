push!(LOAD_PATH, "./src")
using ParametricOptInterface
using MathOptInterface
using GLPK
import Random
#using SparseArrays
#using TimerOutputs
using BenchmarkTools

const MOI = MathOptInterface
const POI = ParametricOptInterface

struct PMedianData
    num_facilities::Int
    num_customers::Int
    num_locations::Int
    customer_locations::Vector{Float64}
end

# This is the LP relaxation.
function generate_poi_problem(model, data::PMedianData; add_parameters::Bool=true)
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
                    assignment_variables[i, j]
                )
                for i in 1:NC for j in 1:NL
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
                    MOI.ScalarAffineTerm(-1.0, facility_variables[j])
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
                    MOI.ScalarAffineTerm(1.0, assignment_variables[i, j])
                    for j in 1:NL
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
        d, cd = MOI.add_constrained_variable(model, POI.Parameter(data.num_facilities))
    end

    if add_parameters
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.(1.0, vcat(facility_variables,d)),
                0.0,
            ),
            MOI.EqualTo{Float64}(0),
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

function solve_moi(data::PMedianData, optimizer; vector_version, params)
    cache = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        optimizer()
    )
    model = MOI.Bridges.full_bridge_optimizer(cache, Float64)
    # resetting optimizer is necessary to use copy_to to transfer all the data
    # from caching optimizer to GLPK in a single batch. If that is not called
    # constraints are passed one by one during model generation phase.
    MOI.Utilities.reset_optimizer(cache)
    for (param, value) in params
        MOI.set(model, param, value)
    end
    if vector_version
        generate_poi_problem_vector(model, data, add_parameters = false)
    else
        generate_poi_problem(model, data, add_parameters = false)
    end
    MOI.optimize!(model)
    return MOI.get(model, MOI.ObjectiveValue())
end

function solve_poi(data::PMedianData, optimizer; vector_version, params)

    model = MOI.Bridges.full_bridge_optimizer(optimizer(), Float64)
    # resetting optimizer is necessary to use copy_to to transfer all the data
    # from caching optimizer to GLPK in a single batch. If that is not called
    # constraints are passed one by one during model generation phase.
    #MOI.Utilities.reset_optimizer(cache)
    for (param, value) in params
        MOI.set(model, param, value)
    end
    if vector_version
        generate_poi_problem_vector(model, data)
    else
        generate_poi_problem(model, data)
    end
    MOI.optimize!(model)
    return MOI.get(model, MOI.ObjectiveValue())
end

function solve_poi_no_params(data::PMedianData, optimizer; vector_version, params)

    model = MOI.Bridges.full_bridge_optimizer(optimizer(), Float64)
    # resetting optimizer is necessary to use copy_to to transfer all the data
    # from caching optimizer to GLPK in a single batch. If that is not called
    # constraints are passed one by one during model generation phase.
    #MOI.Utilities.reset_optimizer(cache)
    for (param, value) in params
        MOI.set(model, param, value)
    end
    if vector_version
        generate_poi_problem_vector(model, data, add_parameters = false)
    else
        generate_poi_problem(model, data, add_parameters = false)
    end
    MOI.optimize!(model)
    return MOI.get(model, MOI.ObjectiveValue())
end

function solve_glpk_moi(data::PMedianData; vector_version, time_limit_sec=Inf)
    params = []
    if isfinite(time_limit_sec)
        push!(params, (MOI.TimeLimitSec(), time_limit_sec))
    end
    s_type = vector_version ? "vector" : "scalar"
    solve_moi(data, GLPK.Optimizer; vector_version=vector_version, params=params)
end

function solve_poi_no_params_glpk(data::PMedianData; vector_version, time_limit_sec=Inf)
    params = []
    if isfinite(time_limit_sec)
        push!(params, (MOI.TimeLimitSec(), time_limit_sec))
    end
    s_type = vector_version ? "vector" : "scalar"

    solve_poi_no_params(data,() -> POI.Optimizer(GLPK.Optimizer()); vector_version=vector_version, params=params)
end

function solve_poi_glpk(data::PMedianData; vector_version, time_limit_sec=Inf)
    params = []
    if isfinite(time_limit_sec)
        push!(params, (MOI.TimeLimitSec(), time_limit_sec))
    end
    s_type = vector_version ? "vector" : "scalar"

    solve_poi(data,() -> POI.Optimizer(GLPK.Optimizer()); vector_version=vector_version, params=params)
end

function run_benchmark(;
    num_facilities, num_customers, num_locations, time_limit_sec, max_iters
)
    Random.seed!(10)
    data = PMedianData(num_facilities, num_customers, num_locations, rand(num_customers) .* num_locations)
    @info "Solve MOI"
    b1 = @benchmark solve_glpk_moi($data, vector_version=false, time_limit_sec=$time_limit_sec) samples=100 
    display(b1)
    println()
    @info "Solve POI no params"
    b2 = @benchmark solve_poi_no_params_glpk($data, vector_version=false, time_limit_sec=$time_limit_sec) samples=100
    display(b2)
    println()
    @info "Solve POI"
    b3 = @benchmark solve_poi_glpk($data, vector_version=false, time_limit_sec=$time_limit_sec) samples=100
    display(b3)
    m1 = median(b1)
    m2 = median(b2)
    m3 = median(b3)
    @info "Comparison MOI-POI"
    j1 = judge(m1,m2)
    display(j1)

    @info "Comparison MOI-POI with parameters"
    j2 = judge(m1,m3)
    display(j2)
    println()
end

run_benchmark(
            num_facilities = 100,
            num_customers = 100,
            num_locations = 100,
            time_limit_sec = 0.0001,
            max_iters = 100,
        )