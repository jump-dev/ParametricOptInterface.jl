using MathOptInterface
using ParametricOptInterface
using BenchmarkTools
using JuMP
const MOI = MathOptInterface
const POI = ParametricOptInterface
import Pkg

function moi_add_variables(N::Int)
    model = direct_model(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}()))
    @variable(model, x[i=1:N])
    return nothing
end

function poi_add_variables(N::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:N])
    return nothing
end

function poi_add_parameters(N::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:N] in POI.Parameter(1))
    return nothing
end

function poi_add_parameters_and_variables(N::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:N] in POI.Parameter(1))
    return nothing
end

function poi_add_parameters_and_variables_alternating(N::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    for i in 1:Int(N/2)
        @variable(model)
        @variable(model, set = POI.Parameter(1))
    end
    return nothing
end

function moi_add_saf_ctr(N::Int, M::Int)
    model = direct_model(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}()))
    @variable(model, x[i=1:N])
    @constraint(model, cons[i=1:M], sum(x) >= 1)
    return nothing
end

function poi_add_saf_ctr(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:N])
    @constraint(model, cons[i=1:M], sum(x) >= 1)
    return nothing
end

function poi_add_saf_variables_and_parameters_ctr(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:Int(N/2)])
    @variable(model, p[i=1:Int(N/2)] in POI.Parameter.(0))
    @constraint(model, con[i=1:M], sum(x) + sum(p) >= 1)
    return nothing
end

function poi_add_saf_variables_and_parameters_ctr_parameter_update(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:Int(N/2)])
    @variable(model, p[i=1:Int(N/2)] in POI.Parameter.(0))
    @constraint(model, con[i=1:M], sum(x) + sum(p) >= 1)
    MOI.set.(model, POI.ParameterValue(), p, 0.5)
    POI.update_parameters!(model.moi_backend)
    return nothing
end

function moi_add_sqf_variables_ctr(N::Int, M::Int)
    model = direct_model(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}()))
    @variable(model, x[i=1:N])
    @constraint(model, con[i=1:M], sum(x.^2) >= 1)
    return nothing
end

function poi_add_sqf_variables_ctr(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:N])
    @constraint(model, con[i=1:M], sum(x.^2) >= 1)
    return nothing
end

function poi_add_sqf_variables_parameters_ctr(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:Int(N/2)])
    @variable(model, p[i=1:Int(N/2)] in POI.Parameter.(1))
    @constraint(model, con[i=1:M], sum(x.*p) >= 1)
    return nothing
end

function poi_add_sqf_variables_parameters_ctr_parameter_update(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:Int(N/2)])
    @variable(model, p[i=1:Int(N/2)] in POI.Parameter.(1))
    @constraint(model, con[i=1:M], sum(x.*p) >= 1)
    MOI.set.(model, POI.ParameterValue(), p, 0.5)
    POI.update_parameters!(model.moi_backend)
    return nothing
end

function poi_add_sqf_parameters_parameters_ctr(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:Int(N/2)])
    @variable(model, p[i=1:Int(N/2)] in POI.Parameter.(1))
    @constraint(model, con[i=1:M], sum(p.^2) >= 1)
    return nothing
end

function poi_add_sqf_parameters_parameters_ctr_parameter_update(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:Int(N/2)])
    @variable(model, p[i=1:Int(N/2)] in POI.Parameter.(1))
    @constraint(model, con[i=1:M], sum(p.^2) >= 1)
    MOI.set.(model, POI.ParameterValue(), p, 0.5)
    POI.update_parameters!(model.moi_backend)
    return nothing
end

function moi_add_saf_obj(N::Int, M::Int)
    model = direct_model(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}()))
    @variable(model, x[i=1:N])
    @objective(model, Min, sum(x))
    return nothing
end

function poi_add_saf_obj(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:N])
    for _ in 1:M
        @objective(model, Min, sum(x))
    end
    return nothing
end

function poi_add_saf_variables_and_parameters_obj(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:Int(N/2)])
    @variable(model, p[i=1:Int(N/2)] in POI.Parameter.(1))
    for _ in 1:M
        @objective(model, Min, sum(x)+sum(p))
    end
    return nothing
end

function poi_add_saf_variables_and_parameters_obj_parameter_update(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:Int(N/2)])
    @variable(model, p[i=1:Int(N/2)] in POI.Parameter.(1))
    for _ in 1:M
        @objective(model, Min, sum(x)+sum(p))
    end
    for _ in 1:M
        MOI.set.(model, POI.ParameterValue(), p, 0.5)
        POI.update_parameters!(model.moi_backend)
    end
    return nothing
end

function moi_add_sqf_variables_obj(N::Int, M::Int)
    model = direct_model(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}()))
    @variable(model, x[i=1:N])
    for _ in 1:M
        @objective(model, Min, sum(x.^2))
    end
    return nothing
end

function poi_add_sqf_variables_obj(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:N])
    for _ in 1:M
        @objective(model, Min, sum(x.^2))
    end
    return nothing
end

function poi_add_sqf_variables_parameters_obj(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:Int(N/2)])
    @variable(model, p[i=1:Int(N/2)] in POI.Parameter.(1))
    for _ in 1:M
        @objective(model, Min, sum(x.*p))
    end
    return nothing
end

function poi_add_sqf_variables_parameters_obj_parameter_update(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:Int(N/2)])
    @variable(model, p[i=1:Int(N/2)] in POI.Parameter.(1))
    for _ in 1:M
        @objective(model, Min, sum(x.*p))
    end
    for _ in 1:M
        MOI.set.(model, POI.ParameterValue(), p, 0.5)
        POI.update_parameters!(model.moi_backend)
    end
    return nothing
end

function poi_add_sqf_parameters_parameters_obj(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:Int(N/2)])
    @variable(model, p[i=1:Int(N/2)] in POI.Parameter.(1))
    for _ in 1:M
        @objective(model, Min, sum(p.^2))
    end
    return nothing
end

function poi_add_sqf_parameters_parameters_obj_parameter_update(N::Int, M::Int)
    model = direct_model(POI.Optimizer(MOIU.MockOptimizer(MOI.Utilities.Model{Float64}())))
    @variable(model, x[i=1:Int(N/2)])
    @variable(model, p[i=1:Int(N/2)] in POI.Parameter.(1))
    for _ in 1:M
        @objective(model, Min, sum(p.^2))
    end
    for _ in 1:M
        MOI.set.(model, POI.ParameterValue(), p, 0.5)
        POI.update_parameters!(model.moi_backend)
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
