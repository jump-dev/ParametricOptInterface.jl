# using Revise

using JuMP
using DiffOpt
using Test
import ParametricOptInterface as POI
using HiGHS
using SCS

function test_diff_rhs()
    model = Model(() -> POI.Optimizer(DiffOpt.Optimizer(HiGHS.Optimizer())))
    set_silent(model)
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

function test_diff_vector_rhs()
    model = direct_model(POI.Optimizer(DiffOpt.diff_optimizer(SCS.Optimizer)))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in MOI.Parameter(3.0))
    @constraint(model, cons, [x - 3 * p] in MOI.Zeros(1))

    # FIXME
    @constraint(model, fake_soc, [0, 0, 0] in SecondOrderCone())

    @objective(model, Min, 2x)
    optimize!(model)
    @test isapprox(MOI.get(model, MOI.VariablePrimal(), x), 9, atol = 1e-3)
    # the function is
    # x(p) = 3p, hence x'(p) = 3
    # differentiate w.r.t. p
    for p_val in 0:3
        MOI.set(model, POI.ParameterValue(), p, p_val)
        optimize!(model)
        @test isapprox(
            MOI.get(model, MOI.VariablePrimal(), x),
            3 * p_val,
            atol = 1e-3,
        )
        for direction in 0:3
            MOI.set(model, POI.ForwardParameter(), p, direction)
            DiffOpt.forward_differentiate!(model)
            @test isapprox(
                MOI.get(model, DiffOpt.ForwardVariablePrimal(), x),
                direction * 3,
                atol = 1e-3,
            )
            # reverse mode
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, direction)
            DiffOpt.reverse_differentiate!(model)
            @test isapprox(
                MOI.get(model, POI.ReverseParameter(), p),
                direction * 3,
                atol = 1e-3,
            )
        end
    end
    return
end

function test_affine_changes()
    model = Model(() -> POI.Optimizer(DiffOpt.Optimizer(HiGHS.Optimizer())))
    set_silent(model)
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
    for direction_p in 1:2
        MOI.set(model, POI.ForwardParameter(), p, direction_p)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
              direction_p * 3 / pc_val
    end
    # update p
    p_val = 2.0
    MOI.set(model, POI.ParameterValue(), p, p_val)
    optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 3 * p_val / pc_val
    # differentiate w.r.t. p
    for direction_p in 1:2
        MOI.set(model, POI.ForwardParameter(), p, direction_p)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
              direction_p * 3 / pc_val
    end
    # differentiate w.r.t. pc
    # stop differentiating with respect to p
    direction_p = 0.0
    MOI.set(model, POI.ForwardParameter(), p, direction_p)
    for direction_pc in 1:2
        MOI.set(model, POI.ForwardParameter(), pc, direction_pc)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
              -direction_pc * 3 * p_val / pc_val^2
    end
    # update pc
    pc_val = 2.0
    MOI.set(model, POI.ParameterValue(), pc, pc_val)
    optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 3 * p_val / pc_val
    for direction_pc in 1:2
        MOI.set(model, POI.ForwardParameter(), pc, direction_pc)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
              -direction_pc * 3 * p_val / pc_val^2
    end
    # test combines directions
    for direction_pc in 1:2, direction_p in 1:2
        MOI.set(model, POI.ForwardParameter(), p, direction_p)
        MOI.set(model, POI.ForwardParameter(), pc, direction_pc)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
              -direction_pc * 3 * p_val / pc_val^2 + direction_p * 3 / pc_val
    end
    return
end

function test_affine_changes_compact()
    model = Model(() -> POI.Optimizer(DiffOpt.Optimizer(HiGHS.Optimizer())))
    set_silent(model)
    p_val = 3.0
    pc_val = 1.0
    @variable(model, x)
    @variable(model, p in MOI.Parameter(p_val))
    @variable(model, pc in MOI.Parameter(pc_val))
    @constraint(model, cons, pc * x >= 3 * p)
    @objective(model, Min, 2x)
    # the function is
    # x(p, pc) = 3p / pc, hence dx/dp = 3 / pc, dx/dpc = -3p / pc^2
    for p_val in 1:3, pc_val in 1:3
        MOI.set(model, POI.ParameterValue(), p, p_val)
        MOI.set(model, POI.ParameterValue(), pc, pc_val)
        optimize!(model)
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 3 * p_val / pc_val
        for direction_pc in 0:2, direction_p in 0:2
            MOI.set(model, POI.ForwardParameter(), p, direction_p)
            MOI.set(model, POI.ForwardParameter(), pc, direction_pc)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
                  -direction_pc * 3 * p_val / pc_val^2 +
                  direction_p * 3 / pc_val
        end
        for direction_x in 0:2
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, direction_x)
            DiffOpt.reverse_differentiate!(model)
            @test MOI.get(model, POI.ReverseParameter(), p) ≈
                  direction_x * 3 / pc_val
            @test MOI.get(model, POI.ReverseParameter(), pc) ≈
                  -direction_x * 3 * p_val / pc_val^2
        end
    end
    return
end

function test_quadratic_rhs_changes()
    model = Model(() -> POI.Optimizer(DiffOpt.Optimizer(HiGHS.Optimizer())))
    set_silent(model)
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
    @constraint(model, cons, 11 * t * x >= 1 + 3 * p * q + 5 * r^2 + 7 * s)
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
          (1 + 3 * p_val * q_val + 5 * r_val^2 + 7 * s_val) / (11 * t_val)
    for p_val in 2:3, q_val in 2:3, r_val in 2:3, s_val in 2:3, t_val in 2:3
        MOI.set(model, POI.ParameterValue(), p, p_val)
        MOI.set(model, POI.ParameterValue(), q, q_val)
        MOI.set(model, POI.ParameterValue(), r, r_val)
        MOI.set(model, POI.ParameterValue(), s, s_val)
        MOI.set(model, POI.ParameterValue(), t, t_val)
        optimize!(model)
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈
              (1 + 3 * p_val * q_val + 5 * r_val^2 + 7 * s_val) / (11 * t_val)
        for dir_p in 0:2, dir_q in 0:2, dir_r in 0:2, dir_s in 0:2, dir_t in 0:2
            MOI.set(model, POI.ForwardParameter(), p, dir_p)
            MOI.set(model, POI.ForwardParameter(), q, dir_q)
            MOI.set(model, POI.ForwardParameter(), r, dir_r)
            MOI.set(model, POI.ForwardParameter(), s, dir_s)
            MOI.set(model, POI.ForwardParameter(), t, dir_t)
            DiffOpt.forward_differentiate!(model)
            @test isapprox(
                MOI.get(model, DiffOpt.ForwardVariablePrimal(), x),
                dir_p * 3 * q_val / (11 * t_val) +
                dir_q * 3 * p_val / (11 * t_val) +
                dir_r * 10 * r_val / (11 * t_val) +
                dir_s * 7 / (11 * t_val) +
                dir_t * (
                    -(1 + 3 * p_val * q_val + 5 * r_val^2 + 7 * s_val) /
                    (11 * t_val^2)
                ),
                atol = 1e-10,
            )
        end
        for dir_x in 0:3
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, dir_x)
            DiffOpt.reverse_differentiate!(model)
            @test isapprox(
                MOI.get(model, POI.ReverseParameter(), p),
                dir_x * 3 * q_val / (11 * t_val),
                atol = 1e-10,
            )
            @test isapprox(
                MOI.get(model, POI.ReverseParameter(), q),
                dir_x * 3 * p_val / (11 * t_val),
                atol = 1e-10,
            )
            @test isapprox(
                MOI.get(model, POI.ReverseParameter(), r),
                dir_x * 10 * r_val / (11 * t_val),
                atol = 1e-10,
            )
            @test isapprox(
                MOI.get(model, POI.ReverseParameter(), s),
                dir_x * 7 / (11 * t_val),
                atol = 1e-10,
            )
            @test isapprox(
                MOI.get(model, POI.ReverseParameter(), t),
                dir_x * (
                    -(1 + 3 * p_val * q_val + 5 * r_val^2 + 7 * s_val) /
                    (11 * t_val^2)
                ),
                atol = 1e-10,
            )
        end
    end
    return
end

function test_affine_changes_compact_max()
    model = Model(() -> POI.Optimizer(DiffOpt.Optimizer(HiGHS.Optimizer())))
    set_silent(model)
    p_val = 3.0
    pc_val = 1.0
    @variable(model, x)
    @variable(model, p in MOI.Parameter(p_val))
    @variable(model, pc in MOI.Parameter(pc_val))
    @constraint(model, cons, pc * x >= 3 * p)
    @objective(model, Max, -2x)
    # the function is
    # x(p, pc) = 3p / pc, hence dx/dp = 3 / pc, dx/dpc = -3p / pc^2
    for p_val in 1:3, pc_val in 1:3
        MOI.set(model, POI.ParameterValue(), p, p_val)
        MOI.set(model, POI.ParameterValue(), pc, pc_val)
        optimize!(model)
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 3 * p_val / pc_val
        for direction_pc in 0:2, direction_p in 0:2
            MOI.set(model, POI.ForwardParameter(), p, direction_p)
            MOI.set(model, POI.ForwardParameter(), pc, direction_pc)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
                  -direction_pc * 3 * p_val / pc_val^2 +
                  direction_p * 3 / pc_val
        end
    end
    return
end

function test_diff_affine_objective()
    model = Model(() -> POI.Optimizer(DiffOpt.Optimizer(HiGHS.Optimizer())))
    set_silent(model)
    p_val = 3.0
    @variable(model, x)
    @variable(model, p in MOI.Parameter(p_val))
    @constraint(model, cons, x >= 3)
    @objective(model, Min, 2x + 3p)
    # x(p, pc) = 3, hence dx/dp = 0
    for p_val in 1:2
        MOI.set(model, POI.ParameterValue(), p, p_val)
        optimize!(model)
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 3
        for direction_p in 0:2
            MOI.set(model, POI.ForwardParameter(), p, direction_p)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 0.0
            # reverse mode
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, direction_p)
            DiffOpt.reverse_differentiate!(model)
            @test MOI.get(model, POI.ReverseParameter(), p) ≈ 0.0
        end
    end
    return
end

function test_diff_quadratic_objective()
    model = Model(() -> POI.Optimizer(DiffOpt.Optimizer(HiGHS.Optimizer())))
    set_silent(model)
    p_val = 3.0
    @variable(model, x)
    @variable(model, p in MOI.Parameter(p_val))
    @constraint(model, cons, x >= 3)
    @objective(model, Min, p * x)
    # x(p, pc) = 3, hence dx/dp = 0
    for p_val in 1:2
        MOI.set(model, POI.ParameterValue(), p, p_val)
        optimize!(model)
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 3
        for direction_p in 0:2
            MOI.set(model, POI.ForwardParameter(), p, direction_p)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 0.0
            # reverse mode
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, direction_p)
            DiffOpt.reverse_differentiate!(model)
            @test MOI.get(model, POI.ReverseParameter(), p) ≈ 0.0
        end
    end
    return
end

function test_quadratic_objective_qp()
    model = Model(() -> POI.Optimizer(DiffOpt.Optimizer(HiGHS.Optimizer())))
    set_silent(model)
    p_val = 3.0
    @variable(model, x)
    @variable(model, p in MOI.Parameter(p_val))
    @constraint(model, cons, x >= -10)
    @objective(model, Min, 3 * p * x + x * x + 5 * p + 7 * p^2)
    # 2x + 3p = 0, hence x = -3p/2
    # hence dx/dp = -3/2
    for p_val in 3:3
        MOI.set(model, POI.ParameterValue(), p, p_val)
        optimize!(model)
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ -3p_val / 2
        for direction_p in 0:2
            MOI.set(model, POI.ForwardParameter(), p, direction_p)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
                  direction_p * (-3 / 2)
            # reverse mode
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, direction_p)
            DiffOpt.reverse_differentiate!(model)
            @test MOI.get(model, POI.ReverseParameter(), p) ≈
                  direction_p * (-3 / 2)
        end
    end
    return
end

function test_diff_errors()
    model = Model(() -> POI.Optimizer(DiffOpt.Optimizer(HiGHS.Optimizer())))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in MOI.Parameter(3.0))
    @constraint(model, cons, x >= 3 * p)
    @objective(model, Min, 2x)
    optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 9

    @test_throws ErrorException MOI.set(model, POI.ForwardParameter(), x, 1)
    @test_throws ErrorException MOI.set(
        model,
        DiffOpt.ReverseVariablePrimal(),
        p,
        1,
    )
    @test_throws ErrorException MOI.get(
        model,
        DiffOpt.ForwardVariablePrimal(),
        p,
    )
    @test_throws ErrorException MOI.get(model, POI.ReverseParameter(), x)

    return
end

function test_diff_projection()
    num_A = 2
    ##### SecondOrderCone #####
    _x_hat = rand(num_A)
    μ = rand(num_A) * 10
    Σ_12 = rand(num_A, num_A)
    Σ = Σ_12 * Σ_12' + 0.1 * I
    γ = 1.0
    model = direct_model(POI.Optimizer(DiffOpt.diff_optimizer(SCS.Optimizer)))
    set_silent(model)
    @variable(model, x[1:num_A])
    @variable(model, x_hat[1:num_A] in MOI.Parameter.(_x_hat))
    @variable(model, norm_2)
    # (x - x_hat)^T Σ^-1 (x - x_hat) <= γ
    @constraint(
        model,
        (x - μ)' * inv(Σ) * (x - μ) <= γ,
    )
    # norm_2 >= ||x - x_hat||_2
    @constraint(model, [norm_2; x - x_hat] in SecondOrderCone())
    @objective(model, Min, norm_2)
    optimize!(model)
    MOI.set.(model, POI.ForwardParameter(), x_hat, ones(num_A))
    DiffOpt.forward_differentiate!(model) # ERROR
    #@test TBD
    return
end

using JuMP
using DiffOpt
using SCS

function fallback_error()
    num_A = 2
    ##### SecondOrderCone #####
    x_hat = rand(num_A)
    μ = rand(num_A) * 10
    Σ_12 = rand(num_A, num_A)
    Σ = Σ_12 * Σ_12' + 0.1 * I
    γ = 1.0
    model = direct_model(DiffOpt.diff_optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x[1:num_A])
    @variable(model, norm_2)
    # (x - x_hat)^T Σ^-1 (x - x_hat) <= γ
    @constraint(
        model,
        (x - μ)' * inv(Σ) * (x - μ) <= γ,
    )
    # norm_2 >= ||x - x_hat||_2
    @constraint(model, ctr, [norm_2; x - x_hat] in SecondOrderCone())
    @objective(model, Min, norm_2)
    optimize!(model)
    MOI.set(model, DiffOpt.ForwardConstraintFunction(), ctr,
        MOI.VectorAffineFunction{Float64}(MOI.VectorOfVariables(x))
    )
    DiffOpt.forward_differentiate!(model) # ERROR
    return
end

#
function scalarize_bridge_error()
    model = direct_model(DiffOpt.diff_optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x)
    # @variable(model, p in MOI.Parameter(3.0))
    p = 3.0
    @constraint(model, cons, [x - 3 * p] in MOI.Zeros(1))

    # it works it this is uncommented
    # @constraint(model, fake_soc, [0, 0, 0] in SecondOrderCone())

    @objective(model, Min, 2x)
    optimize!(model)

    MOI.set(model, DiffOpt.ForwardConstraintFunction(), cons,
        MOI.VectorAffineFunction{Float64}(MOI.VectorOfVariables([x]))
    )
    DiffOpt.forward_differentiate!(model) # ERROR

    return
end