# using Revise

using JuMP
using DiffOpt
using Test
import ParametricOptInterface as POI
using GLPK
using Ipopt
using HiGHS

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
    #
    # test reverse mode
    #
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, 1)
    DiffOpt.reverse_differentiate!(model)
    @test MOI.get(model, POI.ReverseParameter(), p) ≈ 3
    # again with different "direction"
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, 2)
    DiffOpt.reverse_differentiate!(model)
    @test MOI.get(model, POI.ReverseParameter(), p) ≈ 6
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
        for direction_x in 0:2
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, direction_x)
            DiffOpt.reverse_differentiate!(model)
            @test MOI.get(model, POI.ReverseParameter(), p) ≈ direction_x * 3 / pc_val
            @test MOI.get(model, POI.ReverseParameter(), pc) ≈ - direction_x * 3 * p_val / pc_val^2
        end
    end
    return
end

function test_quadratic_rhs_changes()
    model = Model(() ->
        POI.Optimizer(
            DiffOpt.Optimizer(
                GLPK.Optimizer()
            )
        )
    )
    p_val = 2.0
    q_val = 2.0
    r_val = 2.0
    s_val = 2.0
    t_val = 2.0
    @variable(model, x)
    @variable(model, p in MOI.Parameter(p_val))
    @variable(model, q in MOI.Parameter(q_val))
    @variable(model, r in MOI.Parameter(r_val))
    @variable(model, s in MOI.Parameter(s_val))
    @variable(model, t in MOI.Parameter(t_val))
    @constraint(model, cons, 11 * t * x >= 1 + 3 * p * q + 5 * r ^ 2 + 7 * s)
    @objective(model, Min, 2x)
    # the function is
    # x(p, q, r, s, t) = (1 + 3pq + 5r^2 + 7s) / (11t)
    # hence
    # dx/dp = 3q / (11t)
    # dx/dq = 3p / (11t)
    # dx/dr = 10r / (11t)
    # dx/ds = 7 / (11t)
    # dx/dt = - (1 + 3pq + 5r^2 + 7s) / (11t^2)
    optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈
        (1 + 3 * p_val * q_val + 5 * r_val ^ 2 + 7 * s_val) / (11 * t_val)
    for p_val = 2:3, q_val = 2:3, r_val = 2:3, s_val = 2:3, t_val = 2:3
        MOI.set(model, POI.ParameterValue(), p, p_val)
        MOI.set(model, POI.ParameterValue(), q, q_val)
        MOI.set(model, POI.ParameterValue(), r, r_val)
        MOI.set(model, POI.ParameterValue(), s, s_val)
        MOI.set(model, POI.ParameterValue(), t, t_val)
        optimize!(model)
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈
            (1 + 3 * p_val * q_val + 5 * r_val ^ 2 + 7 * s_val) / (11 * t_val)
        for dir_p = 0:2, dir_q = 0:2, dir_r = 0:2, dir_s = 0:2, dir_t = 0:2
            MOI.set(model, POI.ForwardParameter(), p, dir_p)
            MOI.set(model, POI.ForwardParameter(), q, dir_q)
            MOI.set(model, POI.ForwardParameter(), r, dir_r)
            MOI.set(model, POI.ForwardParameter(), s, dir_s)
            MOI.set(model, POI.ForwardParameter(), t, dir_t)
            DiffOpt.forward_differentiate!(model)
            @test isapprox(MOI.get(model, DiffOpt.ForwardVariablePrimal(), x),
                dir_p * 3 * q_val / (11 * t_val) +
                dir_q * 3 * p_val / (11 * t_val) +
                dir_r * 10 * r_val / (11 * t_val) +
                dir_s * 7 / (11 * t_val) +
                dir_t * (- (1 + 3 * p_val * q_val + 5 * r_val ^ 2 + 7 * s_val) /
                    (11 * t_val ^ 2)),
                atol = 1e-10,
            )
        end
    end
    return
end

function test_affine_changes_compact_max()
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
    @objective(model, Max, -2x)
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

function test_diff_affine_objective()
    model = Model(() ->
        POI.Optimizer(
            DiffOpt.Optimizer(
                GLPK.Optimizer()
            )
        )
    )
    p_val = 3.0
    @variable(model, x)
    @variable(model, p in MOI.Parameter(p_val))
    @constraint(model, cons, x >= 3)
    @objective(model, Min, 2x + 3p)
    # x(p, pc) = 3, hence dx/dp = 0
    for p_val = 1:2
        MOI.set(model, POI.ParameterValue(), p, p_val)
        optimize!(model)
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 3
        for direction_p = 0:2
            MOI.set(model, POI.ForwardParameter(), p, direction_p)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 0.0
        end
    end
    return
end

function test_diff_quadratic_objective()
    model = Model(() ->
        POI.Optimizer(
            DiffOpt.Optimizer(
                GLPK.Optimizer()
            )
        )
    )
    p_val = 3.0
    @variable(model, x)
    @variable(model, p in MOI.Parameter(p_val))
    @constraint(model, cons, x >= 3)
    @objective(model, Min, p * x)
    # x(p, pc) = 3, hence dx/dp = 0
    for p_val = 1:2
        MOI.set(model, POI.ParameterValue(), p, p_val)
        optimize!(model)
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 3
        for direction_p = 0:2
            MOI.set(model, POI.ForwardParameter(), p, direction_p)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 0.0
        end
    end
    return
end

function test_quadratic_objective_qp()
    model = Model(() ->
        POI.Optimizer(
            DiffOpt.Optimizer(
                HiGHS.Optimizer()
            )
        )
    )
    set_silent(model)
    p_val = 3.0
    @variable(model, x)
    @variable(model, p in MOI.Parameter(p_val))
    @constraint(model, cons, x >= -10)
    @objective(model, Min, 3 * p * x + x*x)
    # 2x + 3p = 0, hence x = -3p/2
    # hence dx/dp = -3/2
    for p_val = 3:3
        MOI.set(model, POI.ParameterValue(), p, p_val)
        optimize!(model)
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ -3p_val/2
        for direction_p = 0:2
            MOI.set(model, POI.ForwardParameter(), p, direction_p)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ direction_p * (-3/2)
            # reverse mode
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, direction_p)
            DiffOpt.reverse_differentiate!(model)
            @test MOI.get(model, POI.ReverseParameter(), p) ≈ direction_p * (-3/2)
        end
    end
    return
end

# TODO: try with Ipopt
# TODO: make highs support obj change