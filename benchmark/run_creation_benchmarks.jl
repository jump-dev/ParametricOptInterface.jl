import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using ParametricOptInterface
using MathOptInterface
using GLPK
import Random
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
function generate_problem(
    optimizer, 
    data::PMedianData; 
    add_parameters_rhs::Bool= false, 
    add_parameters_lhs::Bool = false, 
    add_parameters_objective::Bool = false)

    model = optimizer()

    NL = data.num_locations
    NC = data.num_customers

    facility_variables = MOI.add_variables(model, NL)
    assignment_variables = reshape(MOI.add_variables(model, NC * NL), NC, NL)

    if add_parameters_rhs || add_parameters_lhs || add_parameters_objective
        d = Vector{MOI.VariableIndex}(undef, NL)
        for i in 1:NL
            d[i], c_param = MOI.add_constrained_variable(model, POI.Parameter(1.0))
        end
    end
   
    ###
    ### Objective function
    ###
    if add_parameters_objective
        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarQuadraticFunction(
                [
                    MOI.ScalarQuadraticTerm(
                        1.0,
                        d[j],
                        assignment_variables[i, j]
                    )
                    for i in 1:NC for j in 1:NL
                ],
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
    else
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
    end
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)


    for v in assignment_variables
        MOI.add_constraint(model, v, MOI.GreaterThan(0.0))
        # "Less than 1.0" constraint is redundant.
    end
    for v in facility_variables
        MOI.add_constraint(model, v, MOI.Interval(0.0, 1.0))
    end
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
    

    if !add_parameters_lhs && add_parameters_rhs
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.(1.0, d),
                0.0,
            ),
            MOI.EqualTo{Float64}(0),
        )
    elseif add_parameters_lhs && !add_parameters_rhs
        MOI.add_constraint(
            model,
            MOI.ScalarQuadraticFunction(
                MOI.ScalarQuadraticTerm.(1.0, d, facility_variables),
                MOI.ScalarAffineTerm.(1.0, facility_variables),
                0.0,
            ),
            MOI.EqualTo{Float64}(0),
        )
    elseif add_parameters_lhs && add_parameters_rhs
        MOI.add_constraint(
            model,
            MOI.ScalarQuadraticFunction(
                MOI.ScalarQuadraticTerm.(1.0, d, facility_variables),
                MOI.ScalarAffineTerm.(1.0, d),
                0.0,
            ),
            MOI.EqualTo{Float64}(0),
        )
    elseif !add_parameters_lhs && !add_parameters_rhs
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.(1.0, facility_variables),
                0.0,
            ),
            MOI.EqualTo{Float64}(0),
        )
    end

    return assignment_variables, facility_variables
end

function POI_OPTIMIZER()
    return POI.Optimizer(GLPK.Optimizer())
end

function MOI_OPTIMIZER()
    return GLPK.Optimizer()
end

function run_benchmark(;
    num_facilities, num_customers, num_locations,
)
    Random.seed!(10)
    data = PMedianData(num_facilities, num_customers, num_locations, rand(num_customers) .* num_locations)
    println("MOI no params")
    @btime generate_problem(
                $(Ref(MOI_OPTIMIZER))[], $(Ref(data))[]; 
                add_parameters_rhs = $(Ref(false))[],
                add_parameters_lhs = $(Ref(false))[],
                add_parameters_objective = $(Ref(false))[]
            )
    println("POI no params")
    @btime generate_problem(
                $(Ref(POI_OPTIMIZER))[], $(Ref(data))[]; 
                add_parameters_rhs = $(Ref(false))[],
                add_parameters_lhs = $(Ref(false))[],
                add_parameters_objective = $(Ref(false))[]
            )
    println("POI params on the rhs")
    @btime generate_problem(
                $(Ref(POI_OPTIMIZER))[], $(Ref(data))[]; 
                add_parameters_rhs = $(Ref(true))[],
                add_parameters_lhs = $(Ref(false))[],
                add_parameters_objective = $(Ref(false))[]
            )
    println("POI params on the objective")
    @btime generate_problem(
                $(Ref(POI_OPTIMIZER))[], $(Ref(data))[]; 
                add_parameters_rhs = $(Ref(false))[],
                add_parameters_lhs = $(Ref(false))[],
                add_parameters_objective = $(Ref(true))[]
            )
    println("POI params on the lhs")
    @btime generate_problem(
                $(Ref(POI_OPTIMIZER))[], $(Ref(data))[]; 
                add_parameters_rhs = $(Ref(false))[],
                add_parameters_lhs = $(Ref(true))[],
                add_parameters_objective = $(Ref(false))[]
            )
    println("POI params on the rhs and lhs")
    @btime generate_problem(
                $(Ref(POI_OPTIMIZER))[], $(Ref(data))[]; 
                add_parameters_rhs = $(Ref(true))[],
                add_parameters_lhs = $(Ref(true))[],
                add_parameters_objective = $(Ref(false))[]
            )
    println("POI params on the rhs, lhs and objective")
    @btime generate_problem(
                $(Ref(POI_OPTIMIZER))[], $(Ref(data))[]; 
                add_parameters_rhs = $(Ref(true))[],
                add_parameters_lhs = $(Ref(true))[],
                add_parameters_objective = $(Ref(true))[]
            )
    return nothing
end

run_benchmark(
    num_facilities = 500,
    num_customers = 500,
    num_locations = 1000
)