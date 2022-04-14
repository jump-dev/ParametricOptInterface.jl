using MathOptInterface
using ParametricOptInterface
using BenchmarkTools
const MOI = MathOptInterface
const POI = ParametricOptInterface
import Pkg

function moi_add_variables(N::Int)
    model = MOI.Utilities.Model{Float64}()
    MOI.add_variables(model, N)
    return nothing
end

function poi_add_variables(N::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    MOI.add_variables(model, N)
    return nothing
end

function poi_add_parameters(N::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    MOI.add_constrained_variable.(model, POI.Parameter.(ones(N)));
    return nothing
end

function poi_add_parameters_and_variables(N::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    MOI.add_variables(model, N/2)
    MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2))))
    return nothing
end

function poi_add_parameters_and_variables_alternating(N::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    for i in 1:Int(N/2)
        MOI.add_variable(model)
        MOI.add_constrained_variable(model, POI.Parameter(1))
    end
    return nothing
end

function moi_add_saf_ctr(N::Int, M::Int)
    model = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(model, N)
    for _ in 1:M
        MOI.add_constraint(
                model,
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.(1.0, x),
                    0.0,
                ),
                MOI.GreaterThan(1.0),
            )
    end
    return nothing
end

function poi_add_saf_ctr(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N)
    for _ in 1:M
        MOI.add_constraint(
                model,
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.(1.0, x),
                    0.0,
                ),
                MOI.GreaterThan(1.0),
            )
    end
    return nothing
end

function poi_add_saf_variables_and_parameters_ctr(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N/2)
    y = first.(MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2)))))
    for _ in 1:M
        MOI.add_constraint(
                model,
                MOI.ScalarAffineFunction(
                    [MOI.ScalarAffineTerm.(1.0, x); MOI.ScalarAffineTerm.(1.0, y)],
                    0.0,
                ),
                MOI.GreaterThan(1.0),
            )
    end
    return nothing
end

function poi_add_saf_variables_and_parameters_ctr_parameter_update(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N/2)
    y = first.(MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2)))))
    for _ in 1:M
        MOI.add_constraint(
                model,
                MOI.ScalarAffineFunction(
                    [MOI.ScalarAffineTerm.(1.0, x); MOI.ScalarAffineTerm.(1.0, y)],
                    0.0,
                ),
                MOI.GreaterThan(1.0),
            )
    end
    MOI.set.(model, POI.ParameterValue(), y, 0.5)
    POI.update_parameters!(model)
    return nothing
end

function moi_add_sqf_variables_ctr(N::Int, M::Int)
    model = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(model, N)
    for _ in 1:M
        MOI.add_constraint(
                model,
                MOI.ScalarQuadraticFunction(
                    MOI.ScalarQuadraticTerm.(1.0, x, x),
                    MOI.ScalarAffineTerm{Float64}[],
                    0.0,
                ),
                MOI.GreaterThan(1.0),
            )
    end
    return nothing
end

function poi_add_sqf_variables_ctr(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N)
    for _ in 1:M
        MOI.add_constraint(
                model,
                MOI.ScalarQuadraticFunction(
                    MOI.ScalarQuadraticTerm.(1.0, x, x),
                    MOI.ScalarAffineTerm{Float64}[],
                    0.0,
                ),
                MOI.GreaterThan(1.0),
            )
    end
    return nothing
end

function poi_add_sqf_variables_parameters_ctr(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N/2)
    y = first.(MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2)))))
    for _ in 1:M
        MOI.add_constraint(
                model,
                MOI.ScalarQuadraticFunction(
                    MOI.ScalarQuadraticTerm.(1.0, x, y),
                    MOI.ScalarAffineTerm{Float64}[],
                    0.0,
                ),
                MOI.GreaterThan(1.0),
            )
    end
    return nothing
end

function poi_add_sqf_variables_parameters_ctr_parameter_update(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N/2)
    y = first.(MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2)))))
    for _ in 1:M
        MOI.add_constraint(
                model,
                MOI.ScalarQuadraticFunction(
                    MOI.ScalarQuadraticTerm.(1.0, x, y),
                    MOI.ScalarAffineTerm{Float64}[],
                    0.0,
                ),
                MOI.GreaterThan(1.0),
            )
    end
    MOI.set.(model, POI.ParameterValue(), y, 0.5)
    POI.update_parameters!(model)
    return nothing
end

function poi_add_sqf_parameters_parameters_ctr(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N/2)
    y = first.(MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2)))))
    for _ in 1:M
        MOI.add_constraint(
                model,
                MOI.ScalarQuadraticFunction(
                    MOI.ScalarQuadraticTerm.(1.0, y, y),
                    MOI.ScalarAffineTerm{Float64}[],
                    0.0,
                ),
                MOI.GreaterThan(1.0),
            )
    end
    return nothing
end

function poi_add_sqf_parameters_parameters_ctr_parameter_update(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N/2)
    y = first.(MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2)))))
    for _ in 1:M
        MOI.add_constraint(
                model,
                MOI.ScalarQuadraticFunction(
                    MOI.ScalarQuadraticTerm.(1.0, y, y),
                    MOI.ScalarAffineTerm{Float64}[],
                    0.0,
                ),
                MOI.GreaterThan(1.0),
            )
    end
    MOI.set.(model, POI.ParameterValue(), y, 0.5)
    POI.update_parameters!(model)
    return nothing
end

function moi_add_saf_obj(N::Int, M::Int)
    model = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(model, N)
    for _ in 1:M
        MOI.set(
            model,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.(1.0, x),
                    0.0,
                )
            )
    end
    return nothing
end

function poi_add_saf_obj(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N)
    for _ in 1:M
        MOI.set(
                model,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.(1.0, x),
                    0.0,
                )
            )
    end
    return nothing
end

function poi_add_saf_variables_and_parameters_obj(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N/2)
    y = first.(MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2)))))
    for _ in 1:M
        MOI.set(
                model,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(
                    [MOI.ScalarAffineTerm.(1.0, x); MOI.ScalarAffineTerm.(1.0, y)],
                    0.0,
                )
            )
    end
    return nothing
end

function poi_add_saf_variables_and_parameters_obj_parameter_update(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N/2)
    y = first.(MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2)))))
    for _ in 1:M
        MOI.set(
                model,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(
                    [MOI.ScalarAffineTerm.(1.0, x); MOI.ScalarAffineTerm.(1.0, y)],
                    0.0,
                )
            )
    end
    for _ in 1:M
        MOI.set.(model, POI.ParameterValue(), y, 0.5)
        POI.update_parameters!(model)
    end
    return nothing
end

function moi_add_sqf_variables_obj(N::Int, M::Int)
    model = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(model, N)
    for _ in 1:M
        MOI.set(
                model,
                MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
                MOI.ScalarQuadraticFunction(
                    MOI.ScalarQuadraticTerm.(1.0, x, x),
                    MOI.ScalarAffineTerm{Float64}[],
                    0.0,
                )
            )
    end
    return nothing
end

function poi_add_sqf_variables_obj(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N)
    for _ in 1:M
        MOI.set(
                model,
                MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
                MOI.ScalarQuadraticFunction(
                    MOI.ScalarQuadraticTerm.(1.0, x, x),
                    MOI.ScalarAffineTerm{Float64}[],
                    0.0,
                )
            )
    end
    return nothing
end

function poi_add_sqf_variables_parameters_obj(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N/2)
    y = first.(MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2)))))
    for _ in 1:M
        MOI.set(
                model,
                MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
                MOI.ScalarQuadraticFunction(
                    MOI.ScalarQuadraticTerm.(1.0, x, y),
                    MOI.ScalarAffineTerm{Float64}[],
                    0.0,
                )
            )
    end
    return nothing
end

function poi_add_sqf_variables_parameters_obj_parameter_update(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N/2)
    y = first.(MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2)))))
    for _ in 1:M
        MOI.set(
                model,
                MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
                MOI.ScalarQuadraticFunction(
                    MOI.ScalarQuadraticTerm.(1.0, x, y),
                    MOI.ScalarAffineTerm{Float64}[],
                    0.0,
                )
            )
    end
    for _ in 1:M
        MOI.set.(model, POI.ParameterValue(), y, 0.5)
        POI.update_parameters!(model)
    end
    return nothing
end

function poi_add_sqf_parameters_parameters_obj(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N/2)
    y = first.(MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2)))))
    for _ in 1:M
        MOI.set(
                model,
                MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
                MOI.ScalarQuadraticFunction(
                    MOI.ScalarQuadraticTerm.(1.0, y, y),
                    MOI.ScalarAffineTerm{Float64}[],
                    0.0,
                )
            )
    end
    return nothing
end

function poi_add_sqf_parameters_parameters_obj_parameter_update(N::Int, M::Int)
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, N/2)
    y = first.(MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2)))))
    for _ in 1:M
        MOI.set(
                model,
                MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
                MOI.ScalarQuadraticFunction(
                    MOI.ScalarQuadraticTerm.(1.0, y, y),
                    MOI.ScalarAffineTerm{Float64}[],
                    0.0,
                )
            )
    end
    for _ in 1:M
        MOI.set.(model, POI.ParameterValue(), y, 0.5)
        POI.update_parameters!(model)
    end
    return nothing
end

function run_benchmarks(N::Int, M::Int)
    println("Pkg status:")
    Pkg.status()
    println("")
    GC.gc()
    println("variables on a MOIU.Model.")
    @btime moi_add_variables($N)
    GC.gc()
    println("variables on a POI.Optimizer.")
    @btime poi_add_variables($N)
    GC.gc()
    println("parameters on a POI.Optimizer.")
    @btime poi_add_parameters($N)
    GC.gc()
    println("parameters and variables on a POI.Optimizer.")
    @btime poi_add_parameters_and_variables($N)
    GC.gc()
    println("alternating parameters and variables on a POI.Optimizer.")
    @btime poi_add_parameters_and_variables_alternating($N)
    GC.gc()
    println("SAF constraint with variables on a MOIU.Model.")
    @btime moi_add_saf_ctr($N, $M)
    GC.gc()
    println("SAF constraint with variables on a POI.Optimizer.")
    @btime poi_add_saf_ctr($N, $M)
    GC.gc()
    println("SAF constraint with variables and parameters on a POI.Optimizer.")
    @btime poi_add_saf_variables_and_parameters_ctr($N, $M)
    GC.gc()
    println("SQF constraint with variables on a MOIU.Model{Float64}.")
    @btime moi_add_sqf_variables_ctr($N, $M)
    GC.gc()
    println("SQF constraint with variables on a POI.Optimizer.")
    @btime poi_add_sqf_variables_ctr($N, $M)
    GC.gc()
    println("SQF constraint with product of variables and parameters on a POI.Optimizer.")
    @btime poi_add_sqf_variables_parameters_ctr($N, $M)
    GC.gc()
    println("SQF constraint with product of parameters on a POI.Optimizer.")
    @btime poi_add_sqf_parameters_parameters_ctr($N, $M)
    GC.gc()
    println("SAF objective with variables on a MOIU.Model.")
    @btime moi_add_saf_obj($N, $M)
    GC.gc()
    println("SAF objective with variables on a POI.Optimizer.")
    @btime poi_add_saf_obj($N, $M)
    GC.gc()
    println("SAF objective with variables and parameters on a POI.Optimizer.")
    @btime poi_add_saf_variables_and_parameters_obj($N, $M)
    GC.gc()
    println("SQF objective with variables on a MOIU.Model.")
    @btime moi_add_sqf_variables_obj($N, $M)
    GC.gc()
    println("SQF objective with variables on a POI.Optimizer.")
    @btime poi_add_sqf_variables_obj($N, $M)
    GC.gc()
    println("SQF objective with product of variables and parameters on a POI.Optimizer.")
    @btime poi_add_sqf_variables_parameters_obj($N, $M)
    GC.gc()
    println("SQF objective with product of parameters on a POI.Optimizer.")
    @btime poi_add_sqf_parameters_parameters_obj($N, $M)
    GC.gc()
    println("Update parameters in SAF constraint with variables and parameters on a POI.Optimizer.")
    @btime poi_add_saf_variables_and_parameters_ctr_parameter_update($N, $M)
    GC.gc()
    println("Update parameters in SAF objective with variables and parameters on a POI.Optimizer.")
    @btime poi_add_saf_variables_and_parameters_obj_parameter_update($N, $M)
    GC.gc()
    println("Update parameters in SQF constraint with product of variables and parameters on a POI.Optimizer.")
    @btime poi_add_sqf_variables_parameters_ctr_parameter_update($N, $M)
    GC.gc()
    println("Update parameters in SQF constraint with product of parameters on a POI.Optimizer.")
    @btime poi_add_sqf_parameters_parameters_ctr_parameter_update($N, $M)
    GC.gc()
    println("Update parameters in SQF objective with product of variables and parameters on a POI.Optimizer.")
    @btime poi_add_sqf_variables_parameters_obj_parameter_update($N, $M)
    GC.gc()
    println("Update parameters in SQF objective with product of parameters on a POI.Optimizer.")
    @btime poi_add_sqf_parameters_parameters_obj_parameter_update($N, $M)
    return nothing
end

N = 10_000
M = 100
run_benchmarks(N, M)
