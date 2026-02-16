# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestCubic

using Test
using JuMP

import HiGHS
import ParametricOptInterface as POI

const ATOL = 1e-4

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

# ============================================================================
# Parser Tests
# ============================================================================

function test_cubic_parse_single_pvv_term()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    p = POI.v_idx(POI.ParameterIndex(1))

    # 2 * x * y * p
    f = MOI.ScalarNonlinearFunction(:*, Any[2.0, x, y, p])
    result = POI._parse_cubic_expression(f, Float64)

    @test result !== nothing
    pvv = result.pvv
    @test length(pvv) == 1
    @test pvv[1].coefficient == 2.0
    @test pvv[1].index_1 == p
    @test pvv[1].index_2 == x
    @test pvv[1].index_3 == y
    return
end

function test_cubic_parse_squared_variable()
    x = MOI.VariableIndex(1)
    p = POI.v_idx(POI.ParameterIndex(1))

    # 3 * p * x^2 using power operator
    f = MOI.ScalarNonlinearFunction(
        :*,
        Any[3.0, p, MOI.ScalarNonlinearFunction(:^, Any[x, 2])],
    )
    result = POI._parse_cubic_expression(f, Float64)

    @test result !== nothing
    pvv = result.pvv
    @test length(pvv) == 1
    @test pvv[1].coefficient == 3.0
    # Check that both variables are x (squared variable)
    v1 = pvv[1].index_2
    v2 = pvv[1].index_3
    @test v1 == x
    @test v2 == x
    return
end

function test_cubic_parse_parenthesis_variations()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    p = POI.v_idx(POI.ParameterIndex(1))

    # Flat: 2 * x * y * p
    f1 = MOI.ScalarNonlinearFunction(:*, Any[2.0, x, y, p])

    # Left-associative: ((2*x)*y)*p
    f2 = MOI.ScalarNonlinearFunction(
        :*,
        Any[
            MOI.ScalarNonlinearFunction(
                :*,
                Any[MOI.ScalarNonlinearFunction(:*, Any[2.0, x]), y],
            ),
            p,
        ],
    )

    # Grouped: (2*p) * (x*y)
    f3 = MOI.ScalarNonlinearFunction(
        :*,
        Any[
            MOI.ScalarNonlinearFunction(:*, Any[2.0, p]),
            MOI.ScalarNonlinearFunction(:*, Any[x, y]),
        ],
    )

    r1 = POI._parse_cubic_expression(f1, Float64)
    r2 = POI._parse_cubic_expression(f2, Float64)
    r3 = POI._parse_cubic_expression(f3, Float64)

    # All should parse to equivalent results
    for r in [r1, r2, r3]
        @test r !== nothing
        pvv = r.pvv
        @test length(pvv) == 1
        @test pvv[1].coefficient == 2.0
    end
    return
end

function test_cubic_parse_ppv_term()
    x = MOI.VariableIndex(1)
    p = POI.v_idx(POI.ParameterIndex(1))
    q = POI.v_idx(POI.ParameterIndex(2))

    # 2 * p * q * x
    f = MOI.ScalarNonlinearFunction(:*, Any[2.0, p, q, x])
    result = POI._parse_cubic_expression(f, Float64)

    @test result !== nothing
    ppv = result.ppv
    @test length(ppv) == 1
    @test ppv[1].coefficient == 2.0
    return
end

function test_cubic_parse_ppp_term()
    p = POI.v_idx(POI.ParameterIndex(1))
    q = POI.v_idx(POI.ParameterIndex(2))
    r = POI.v_idx(POI.ParameterIndex(3))

    # 3 * p * q * r
    f = MOI.ScalarNonlinearFunction(:*, Any[3.0, p, q, r])
    result = POI._parse_cubic_expression(f, Float64)

    @test result !== nothing
    ppp = result.ppp
    @test length(ppp) == 1
    @test ppp[1].coefficient == 3.0
    return
end

function test_cubic_parse_invalid_degree_4()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    z = MOI.VariableIndex(3)
    p = POI.v_idx(POI.ParameterIndex(1))

    # x * y * z * p (degree 4) should return nothing
    f = MOI.ScalarNonlinearFunction(:*, Any[x, y, z, p])
    result = POI._parse_cubic_expression(f, Float64)

    @test result === nothing
    return
end

function test_cubic_parse_three_vars_no_param()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    z = MOI.VariableIndex(3)

    # x * y * z (3 variables, 0 parameters) should be rejected
    f = MOI.ScalarNonlinearFunction(:*, Any[x, y, z])
    result = POI._parse_cubic_expression(f, Float64)

    @test result === nothing
    return
end

function test_cubic_parse_subtraction()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    p = POI.v_idx(POI.ParameterIndex(1))

    # x*y*p - 2*x (one cubic, one affine with negative coef)
    f = MOI.ScalarNonlinearFunction(
        :-,
        Any[
            MOI.ScalarNonlinearFunction(:*, Any[x, y, p]),
            MOI.ScalarNonlinearFunction(:*, Any[2.0, x]),
        ],
    )
    result = POI._parse_cubic_expression(f, Float64)

    @test result !== nothing
    pvv = result.pvv
    @test length(pvv) == 1
    # Check affine term via quadratic_func
    affine = result.v
    @test length(affine) == 1
    @test affine[1].coefficient == -2.0
    return
end

function test_cubic_parse_unary_minus()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    p = POI.v_idx(POI.ParameterIndex(1))

    # -x*y*p (negation of cubic term)
    f = MOI.ScalarNonlinearFunction(
        :-,
        Any[MOI.ScalarNonlinearFunction(:*, Any[x, y, p])],
    )
    result = POI._parse_cubic_expression(f, Float64)

    @test result !== nothing
    pvv = result.pvv
    @test length(pvv) == 1
    @test pvv[1].coefficient == -1.0
    return
end

function test_cubic_parse_term_combination()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    p = POI.v_idx(POI.ParameterIndex(1))

    # x*y*p + 2*x*y*p = 3*x*y*p (should combine into single term)
    f = MOI.ScalarNonlinearFunction(
        :+,
        Any[
            MOI.ScalarNonlinearFunction(:*, Any[x, y, p]),
            MOI.ScalarNonlinearFunction(:*, Any[2.0, x, y, p]),
        ],
    )
    result = POI._parse_cubic_expression(f, Float64)

    @test result !== nothing
    pvv = result.pvv
    @test length(pvv) == 1  # combined into single term
    @test pvv[1].coefficient == 3.0
    return
end

function test_cubic_parse_non_polynomial_rejected()
    x = MOI.VariableIndex(1)
    p = POI.v_idx(POI.ParameterIndex(1))

    # sin(x) * p - should be rejected
    f = MOI.ScalarNonlinearFunction(
        :*,
        Any[MOI.ScalarNonlinearFunction(:sin, Any[x]), p],
    )
    result = POI._parse_cubic_expression(f, Float64)

    @test result === nothing

    # sin(x) * p - should be rejected
    f = MOI.ScalarNonlinearFunction(
        :+,
        Any[MOI.ScalarNonlinearFunction(:sin, Any[x]), p],
    )
    result = POI._parse_cubic_expression(f, Float64)

    @test result === nothing
    return
end

function test_parse_nonlinear_with_saf()
    saf = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, MOI.VariableIndex(1))],
        1.3,
    )
    f = MOI.ScalarNonlinearFunction(:+, Any[saf, 1.0])
    result = POI._parse_cubic_expression(f, Float64)
    @test result.constant == 2.3
    return
end

# ============================================================================
# JuMP Integration Tests
# ============================================================================

function test_jump_cubic_pvv_basic()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, -1 <= y <= 10)
    @variable(model, p in MOI.Parameter(1.0))

    # a convex quadratic with cross terms
    # Minimize: x ^ 2 + 2 * x * y + y ^ 2 - 3 x
    # Subject to: x + y >= 2
    @constraint(model, x + y >= 0)
    @objective(model, Min, x^2 + p * x * y + y^2 - 3 * x)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -3.0 atol = ATOL
    @test value(x) ≈ 2.0 atol = ATOL
    @test value(y) ≈ -1.0 atol = ATOL

    # Change p to 0.5
    set_parameter_value(p, 0.5)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -2.4 atol = ATOL
    @test value(x) ≈ 1.6 atol = ATOL
    @test value(y) ≈ -0.4 atol = ATOL

    # Change p to 0 (removes cross term)
    set_parameter_value(p, 0.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -9 / 4 atol = ATOL
    @test value(x) ≈ 3 / 2 atol = ATOL
    @test value(y) ≈ 0.0 atol = ATOL
    return
end

function test_jump_cubic_pvv_same()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, p in MOI.Parameter(1.0))

    # a convex quadratic with cross terms
    # Minimize: p x ^ 2 - 3 x
    # Subject to: x >= 0
    @constraint(model, x >= 0)
    @objective(model, Min, p * x^2 - 3 * x)

    # Optimize with p=1
    # Optimal at x=3/2, obj = -9/4
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -9 / 4 atol = ATOL
    @test value(x) ≈ 3 / 2 atol = ATOL

    # Change p to 0.5
    # Optimal at x=3.0, obj = -9/2
    set_parameter_value(p, 0.5)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -9 / 2 atol = ATOL
    @test value(x) ≈ 3 atol = ATOL

    # Change p to 0 (removes cross term)
    set_parameter_value(p, 0.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -30.0 atol = ATOL
    @test value(x) ≈ 10.0 atol = ATOL
    return
end

function test_jump_cubic_ppv_basic()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(2.0))
    @variable(model, q in MOI.Parameter(3.0))

    # Minimize: x + p*q*x = x * (1 + p*q)
    # With p=2, q=3: minimize x * (1 + 6) = 7x
    # Subject to: x >= 1
    @constraint(model, x >= 1)
    @objective(model, Min, x + p * q * x)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    # Optimal at x=1, obj = 7
    @test objective_value(model) ≈ 7.0 atol = ATOL

    # Change p=1, q=1: minimize x*(1+1) = 2x
    set_parameter_value(p, 1.0)
    set_parameter_value(q, 1.0)
    optimize!(model)
    @test objective_value(model) ≈ 2.0 atol = ATOL
end

function test_jump_cubic_ppv_same()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(2.0))

    # Minimize: x + p^2 * x = x * (1 + p^2)
    # With p=2: minimize x * (1 + 4) = 5x
    # Subject to: x >= 1
    @constraint(model, x >= 1)
    @objective(model, Min, x + p * p * x)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    # Optimal at x=1, obj = 5
    @test objective_value(model) ≈ 5.0 atol = ATOL

    # Change p=1: minimize x*(1+1) = 2x
    set_parameter_value(p, 1.0)
    optimize!(model)
    @test objective_value(model) ≈ 2.0 atol = ATOL
end

function test_jump_cubic_ppp_basic()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(2.0))
    @variable(model, q in MOI.Parameter(3.0))
    @variable(model, r in MOI.Parameter(4.0))

    # Minimize: x + p*q*r
    # With p=2, q=3, r=4: minimize x + 24
    # Subject to: x >= 1
    @constraint(model, x >= 1)
    @objective(model, Min, x + p * q * r)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    # Optimal at x=1, obj = 1 + 24 = 25
    @test objective_value(model) ≈ 25.0 atol = ATOL

    # Change p=1, q=1, r=1: minimize x + 1
    set_parameter_value(p, 1.0)
    set_parameter_value(q, 1.0)
    set_parameter_value(r, 1.0)
    optimize!(model)
    @test objective_value(model) ≈ 2.0 atol = ATOL
    return
end

function test_jump_cubic_ppp_same()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(2.0))

    # Minimize: x + p^3
    # With p=2: minimize x + 8
    # Subject to: x >= 1
    @constraint(model, x >= 1)
    @objective(model, Min, x + p * p * p)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    # Optimal at x=1, obj = 1 + 8 = 9
    @test objective_value(model) ≈ 9.0 atol = ATOL

    # Change p=1: minimize x + 1
    set_parameter_value(p, 1.0)
    optimize!(model)
    @test objective_value(model) ≈ 2.0 atol = ATOL
    return
end

function test_jump_cubic_parameter_initially_zero()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, -1 <= y <= 10)
    @variable(model, p in MOI.Parameter(0.0))

    # a convex quadratic with cross terms
    # Minimize: x ^ 2 + 2 * x * y + y ^ 2 - 3 x
    # Subject to: x + y >= 2
    @constraint(model, x + y >= 0)
    @objective(model, Min, x^2 + p * x * y + y^2 - 3 * x)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -9 / 4 atol = ATOL
    @test value(x) ≈ 3 / 2 atol = ATOL
    @test value(y) ≈ 0.0 atol = ATOL

    # Change p to 0.5
    set_parameter_value(p, 0.5)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -2.4 atol = ATOL
    @test value(x) ≈ 1.6 atol = ATOL
    @test value(y) ≈ -0.4 atol = ATOL

    return
end

function test_jump_cubic_parameter_division_by_constant()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, 0 <= y <= 10)
    @variable(model, p in MOI.Parameter(0.0))

    # a convex quadratic with cross terms
    # Minimize: x ^ 2 + 2 * x * y + y ^ 2 - 3 x
    # Subject to: x + y >= 2
    @constraint(model, x + y >= 0)
    @objective(model, Min, x^2 + p * x * y / 1 + y^2 - 3 * x)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -9 / 4 atol = ATOL
    @test value(x) ≈ 3 / 2 atol = ATOL
    @test value(y) ≈ 0.0 atol = ATOL

    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, -1 <= y <= 10)
    @variable(model, p in MOI.Parameter(1.0))

    # a convex quadratic with cross terms
    # Minimize: x ^ 2 + 0.5 * x * y + y ^ 2 - 3 x
    # Subject to: x + y >= 2
    @constraint(model, x + y >= 0)
    @objective(model, Min, x^2 + p * x * y / 2 + y^2 - 3 * x)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -2.4 atol = ATOL
    @test value(x) ≈ 1.6 atol = ATOL
    @test value(y) ≈ -0.4 atol = ATOL

    return
end

end  # module

TestCubic.runtests()
