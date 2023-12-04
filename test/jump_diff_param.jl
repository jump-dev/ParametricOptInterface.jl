# using Revise

using JuMP
using DiffOpt
using Test
import ParametricOptInterface as POI

function test_diff_rhs()
    model = Model(() ->
        POI.Optimizer(
            DiffOpt.Optimizer(
                GLPK.Optimizer()
            )
        )
    )
    @variable(model, x)
    @variable(model, p in MOI.Parameter(3.0))
    @constraint(model, cons, x >= 3 * p)
    @objective(model, Min, 2x)
    optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 9
    # the function is
    # x(p) = 3p, hence x'(p) = 3
    # differentiate w.r.t. p
    MOI.set(model, POI.ForwardParameter(), p, 1)
    DiffOpt.forward_differentiate!(model)
    @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 3
    # again with different "direction"
    MOI.set(model, POI.ForwardParameter(), p, 2)
    DiffOpt.forward_differentiate!(model)
    @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 6
    #
    MOI.set(model, POI.ParameterValue(), p, 2.0)
    optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 6
    # differentiate w.r.t. p
    MOI.set(model, POI.ForwardParameter(), p, 1)
    DiffOpt.forward_differentiate!(model)
    @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 3
    # again with different "direction"
    MOI.set(model, POI.ForwardParameter(), p, 2)
    DiffOpt.forward_differentiate!(model)
    @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 6
    return
end

function test_affine_changes()
    model = Model(() ->
        POI.Optimizer(
            DiffOpt.Optimizer(
                GLPK.Optimizer()
            )
        )
    )
    p_val = 3.0
    pc_val = 1.0
    @variable(model, x)
    @variable(model, p in MOI.Parameter(p_val))
    @variable(model, pc in MOI.Parameter(pc_val))
    @constraint(model, cons, pc * x >= 3 * p)
    @objective(model, Min, 2x)
    optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 3 * p_val / pc_val
    # the function is
    # x(p, pc) = 3p / pc, hence dx/dp = 3 / pc, dx/dpc = -3p / pc^2
    # differentiate w.r.t. p
    for direction_p = 1:2
        MOI.set(model, POI.ForwardParameter(), p, direction_p)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ direction_p * 3 / pc_val
    end
    # update p
    p_val = 2.0
    MOI.set(model, POI.ParameterValue(), p, p_val)
    optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 3 * p_val / pc_val
    # differentiate w.r.t. p
    for direction_p = 1:2
        MOI.set(model, POI.ForwardParameter(), p, direction_p)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ direction_p * 3 / pc_val
    end
    # differentiate w.r.t. pc
    # stop differentiating with respect to p
    direction_p = 0.0
    MOI.set(model, POI.ForwardParameter(), p, direction_p)
    for direction_pc = 1:2
        MOI.set(model, POI.ForwardParameter(), pc, direction_pc)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ - direction_pc * 3 * p_val / pc_val^2
    end
    # update pc
    pc_val = 2.0
    MOI.set(model, POI.ParameterValue(), pc, pc_val)
    optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 3 * p_val / pc_val
    for direction_pc = 1:2
        MOI.set(model, POI.ForwardParameter(), pc, direction_pc)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ - direction_pc * 3 * p_val / pc_val^2
    end
    # test combines directions
    for direction_pc = 1:2, direction_p = 1:2
        MOI.set(model, POI.ForwardParameter(), p, direction_p)
        MOI.set(model, POI.ForwardParameter(), pc, direction_pc)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
            - direction_pc * 3 * p_val / pc_val^2 + direction_p * 3 / pc_val
    end
    return
end

function test_affine_changes_compact()
    model = Model(() ->
        POI.Optimizer(
            DiffOpt.Optimizer(
                GLPK.Optimizer()
            )
        )
    )
    p_val = 3.0
    pc_val = 1.0
    @variable(model, x)
    @variable(model, p in MOI.Parameter(p_val))
    @variable(model, pc in MOI.Parameter(pc_val))
    @constraint(model, cons, pc * x >= 3 * p)
    @objective(model, Min, 2x)
    # the function is
    # x(p, pc) = 3p / pc, hence dx/dp = 3 / pc, dx/dpc = -3p / pc^2
    for p_val = 1:3, pc_val = 1:3
        MOI.set(model, POI.ParameterValue(), p, p_val)
        MOI.set(model, POI.ParameterValue(), pc, pc_val)
        optimize!(model)
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 3 * p_val / pc_val
        for direction_pc = 0:2, direction_p = 0:2
            MOI.set(model, POI.ForwardParameter(), p, direction_p)
            MOI.set(model, POI.ForwardParameter(), pc, direction_pc)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
                - direction_pc * 3 * p_val / pc_val^2 + direction_p * 3 / pc_val
        end
    end
    return
end