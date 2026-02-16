# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestCubic

using Test
using JuMP

import HiGHS
import ParametricOptInterface as POI
import MathOptInterface as MOI

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
# Helper mock optimizer
# ============================================================================

"""
Mock optimizer that rejects ScalarQuadraticFunction objectives.
Defined in its own module to avoid method-table invalidation.
"""
struct NoQuadObjModel <: MOI.ModelLike
    inner::MOI.Utilities.Model{Float64}
end

NoQuadObjModel() = NoQuadObjModel(MOI.Utilities.Model{Float64}())

MOI.add_variable(m::NoQuadObjModel) = MOI.add_variable(m.inner)
MOI.is_empty(m::NoQuadObjModel) = MOI.is_empty(m.inner)
MOI.empty!(m::NoQuadObjModel) = MOI.empty!(m.inner)
function MOI.is_valid(m::NoQuadObjModel, vi::MOI.VariableIndex)
    return MOI.is_valid(m.inner, vi)
end

function MOI.set(
    ::NoQuadObjModel,
    ::MOI.ObjectiveFunction{<:MOI.ScalarQuadraticFunction},
    ::MOI.ScalarQuadraticFunction,
)
    return error("Quadratic objectives not supported")
end

function MOI.set(m::NoQuadObjModel, attr::MOI.ObjectiveSense, v)
    return MOI.set(m.inner, attr, v)
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

# ============================================================================
# Parser Tests - Additional Coverage
# ============================================================================

function test_cubic_parse_quadratic_function_input()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    p = POI.v_idx(POI.ParameterIndex(1))

    # ScalarQuadraticFunction as argument inside a nonlinear expression
    sqf = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(2.0, x, y)],  # off-diagonal: 2*x*y
        [MOI.ScalarAffineTerm(3.0, x)],
        1.5,
    )
    f = MOI.ScalarNonlinearFunction(:+, Any[sqf, p])
    result = POI._parse_cubic_expression(f, Float64)
    @test result !== nothing
    @test length(result.vv) == 1
    @test result.vv[1].coefficient == 2.0
    @test length(result.v) == 1
    @test result.v[1].coefficient == 3.0
    @test length(result.p) == 1
    @test result.constant == 1.5
    return
end

function test_cubic_parse_quadratic_function_diagonal()
    x = MOI.VariableIndex(1)

    # ScalarQuadraticFunction with diagonal term: x^2
    # MOI convention: diagonal coefficient C means (C/2)*x^2, so C=2 means x^2
    sqf = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(2.0, x, x)],  # diagonal: (2/2)*x^2 = x^2
        MOI.ScalarAffineTerm{Float64}[],
        0.0,
    )
    f = MOI.ScalarNonlinearFunction(:+, Any[sqf, 0.0])
    result = POI._parse_cubic_expression(f, Float64)
    @test result !== nothing
    @test length(result.vv) == 1
    # After parsing: coef should have MOI convention applied
    return
end

function test_cubic_parse_power_zero_exponent()
    x = MOI.VariableIndex(1)

    # x^0 = 1 (constant)
    f = MOI.ScalarNonlinearFunction(
        :+,
        Any[MOI.ScalarNonlinearFunction(:^, Any[x, 0]), 5.0],
    )
    result = POI._parse_cubic_expression(f, Float64)
    @test result !== nothing
    @test result.constant == 6.0
    return
end

function test_cubic_parse_power_negative_exponent()
    x = MOI.VariableIndex(1)

    # x^(-1) should be rejected
    f = MOI.ScalarNonlinearFunction(:^, Any[x, -1])
    result = POI._parse_cubic_expression(f, Float64)
    @test result === nothing
    return
end

function test_cubic_parse_power_non_integer_exponent()
    x = MOI.VariableIndex(1)

    # x^1.5 should be rejected
    f = MOI.ScalarNonlinearFunction(:^, Any[x, 1.5])
    result = POI._parse_cubic_expression(f, Float64)
    @test result === nothing
    return
end

function test_cubic_parse_division_by_zero()
    x = MOI.VariableIndex(1)

    # x / 0 should be rejected
    f = MOI.ScalarNonlinearFunction(:/, Any[x, 0.0])
    result = POI._parse_cubic_expression(f, Float64)
    @test result === nothing
    return
end

function test_cubic_parse_division_by_variable()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)

    # x / y should be rejected (variable denominator)
    f = MOI.ScalarNonlinearFunction(:/, Any[x, y])
    result = POI._parse_cubic_expression(f, Float64)
    @test result === nothing
    return
end

function test_cubic_parse_division_wrong_arity()
    x = MOI.VariableIndex(1)

    # Division with 1 or 3 args should be rejected
    f = MOI.ScalarNonlinearFunction(:/, Any[x])
    result = POI._parse_cubic_expression(f, Float64)
    @test result === nothing

    f = MOI.ScalarNonlinearFunction(:/, Any[x, 2.0, 3.0])
    result = POI._parse_cubic_expression(f, Float64)
    @test result === nothing
    return
end

function test_cubic_parse_power_wrong_arity()
    x = MOI.VariableIndex(1)

    # Power with 1 or 3 args should be rejected
    f = MOI.ScalarNonlinearFunction(:^, Any[x])
    result = POI._parse_cubic_expression(f, Float64)
    @test result === nothing

    f = MOI.ScalarNonlinearFunction(:^, Any[x, 2, 3])
    result = POI._parse_cubic_expression(f, Float64)
    @test result === nothing
    return
end

function test_cubic_parse_convenience_method()
    x = MOI.VariableIndex(1)
    p = POI.v_idx(POI.ParameterIndex(1))

    # Test the convenience method without specifying type
    f = MOI.ScalarNonlinearFunction(:*, Any[2.0, p, x, x])
    result = POI._parse_cubic_expression(f)
    @test result !== nothing
    @test length(result.pvv) == 1
    @test result.pvv[1].coefficient == 2.0
    return
end

function test_cubic_parse_pp_terms()
    p = POI.v_idx(POI.ParameterIndex(1))
    q = POI.v_idx(POI.ParameterIndex(2))

    # p * q (quadratic in parameters only)
    f = MOI.ScalarNonlinearFunction(:*, Any[3.0, p, q])
    result = POI._parse_cubic_expression(f, Float64)
    @test result !== nothing
    @test length(result.pp) == 1
    @test result.pp[1].coefficient == 3.0
    return
end

function test_cubic_parse_pp_same_parameter()
    p = POI.v_idx(POI.ParameterIndex(1))

    # p^2 (diagonal quadratic in parameters)
    f = MOI.ScalarNonlinearFunction(:^, Any[p, 2])
    result = POI._parse_cubic_expression(f, Float64)
    @test result !== nothing
    @test length(result.pp) == 1
    return
end

function test_cubic_parse_pv_terms()
    x = MOI.VariableIndex(1)
    p = POI.v_idx(POI.ParameterIndex(1))

    # 4 * p * x (quadratic with one parameter and one variable)
    f = MOI.ScalarNonlinearFunction(:*, Any[4.0, p, x])
    result = POI._parse_cubic_expression(f, Float64)
    @test result !== nothing
    @test length(result.pv) == 1
    @test result.pv[1].coefficient == 4.0
    return
end

function test_cubic_parse_mixed_all_degrees()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    p = POI.v_idx(POI.ParameterIndex(1))
    q = POI.v_idx(POI.ParameterIndex(2))

    # 2*p*x*y + 3*p*q*x + p*q*p + x^2 + 5*p*x + 7*x + 2*p + 10
    f = MOI.ScalarNonlinearFunction(
        :+,
        Any[
            MOI.ScalarNonlinearFunction(:*, Any[2.0, p, x, y]),   # pvv
            MOI.ScalarNonlinearFunction(:*, Any[3.0, p, q, x]),   # ppv
            MOI.ScalarNonlinearFunction(:*, Any[1.0, p, q, p]),   # ppp
            MOI.ScalarNonlinearFunction(:^, Any[x, 2]),           # vv
            MOI.ScalarNonlinearFunction(:*, Any[5.0, p, x]),      # pv
            MOI.ScalarNonlinearFunction(:*, Any[7.0, x]),         # v
            MOI.ScalarNonlinearFunction(:*, Any[2.0, p]),         # p
            10.0,                                                 # constant
        ],
    )
    result = POI._parse_cubic_expression(f, Float64)
    @test result !== nothing
    @test length(result.pvv) == 1
    @test result.pvv[1].coefficient == 2.0
    @test length(result.ppv) == 1
    @test result.ppv[1].coefficient == 3.0
    @test length(result.ppp) == 1
    @test length(result.vv) == 1
    @test length(result.pv) == 1
    @test length(result.v) == 1
    @test result.v[1].coefficient == 7.0
    @test length(result.p) == 1
    @test result.p[1].coefficient == 2.0
    @test result.constant == 10.0
    return
end

function test_cubic_parse_zero_coefficient_elimination()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    p = POI.v_idx(POI.ParameterIndex(1))

    # p*x*y - p*x*y = 0 (coefficients cancel)
    f = MOI.ScalarNonlinearFunction(
        :-,
        Any[
            MOI.ScalarNonlinearFunction(:*, Any[1.0, p, x, y]),
            MOI.ScalarNonlinearFunction(:*, Any[1.0, p, x, y]),
        ],
    )
    result = POI._parse_cubic_expression(f, Float64)
    @test result !== nothing
    @test isempty(result.pvv)
    return
end

function test_cubic_parse_multiple_pvv_different_vars()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    z = MOI.VariableIndex(3)
    p = POI.v_idx(POI.ParameterIndex(1))

    # p*x*y + 2*p*x*z (two different pvv terms)
    f = MOI.ScalarNonlinearFunction(
        :+,
        Any[
            MOI.ScalarNonlinearFunction(:*, Any[1.0, p, x, y]),
            MOI.ScalarNonlinearFunction(:*, Any[2.0, p, x, z]),
        ],
    )
    result = POI._parse_cubic_expression(f, Float64)
    @test result !== nothing
    @test length(result.pvv) == 2
    return
end

function test_cubic_parse_subtraction_multiple_args()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    p = POI.v_idx(POI.ParameterIndex(1))

    # p*x*y - x - 1 (binary subtraction, second operand is sum)
    f = MOI.ScalarNonlinearFunction(
        :-,
        Any[
            MOI.ScalarNonlinearFunction(:*, Any[p, x, y]),
            MOI.ScalarNonlinearFunction(:+, Any[x, 1.0]),
        ],
    )
    result = POI._parse_cubic_expression(f, Float64)
    @test result !== nothing
    @test length(result.pvv) == 1
    @test length(result.v) == 1
    @test result.v[1].coefficient == -1.0
    @test result.constant == -1.0
    return
end

# ============================================================================
# Cubic Types - Direct Unit Tests
# ============================================================================

function test_normalize_cubic_indices_mixed()
    x = MOI.VariableIndex(2)
    y = MOI.VariableIndex(1)
    p = POI.v_idx(POI.ParameterIndex(1))

    n1, n2, n3 = POI._normalize_cubic_indices(x, p, y)
    # Parameter should come first
    @test POI._is_parameter(n1)
    @test !POI._is_parameter(n2)
    @test !POI._is_parameter(n3)
    return
end

function test_make_cubic_term_normalization()
    x = MOI.VariableIndex(2)
    y = MOI.VariableIndex(1)
    p = POI.v_idx(POI.ParameterIndex(1))

    term = POI._make_cubic_term(3.0, x, p, y)
    @test term.coefficient == 3.0
    # index_1 should be parameter
    @test POI._is_parameter(term.index_1)
    return
end

# ============================================================================
# ParametricCubicFunction - Unit Tests
# ============================================================================

function test_parametric_cubic_function_constructor_with_ppv()
    x = MOI.VariableIndex(1)
    p = POI.v_idx(POI.ParameterIndex(1))
    q = POI.v_idx(POI.ParameterIndex(2))

    # p*q*x + 2*x (ppv term + affine term on same variable)
    f = MOI.ScalarNonlinearFunction(
        :+,
        Any[
            MOI.ScalarNonlinearFunction(:*, Any[1.0, p, q, x]),
            MOI.ScalarNonlinearFunction(:*, Any[2.0, x]),
        ],
    )
    parsed = POI._parse_cubic_expression(f, Float64)
    @test parsed !== nothing
    pcf = POI.ParametricCubicFunction(parsed)
    @test length(pcf.ppv) == 1
    # x should be in affine_data since it's shared with ppv
    @test haskey(pcf.affine_data, x)
    @test pcf.affine_data[x] == 2.0
    return
end

function test_parametric_cubic_function_constructor_np_affine()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    p = POI.v_idx(POI.ParameterIndex(1))

    # p*x^2 + 3*y (pvv term on x, non-parametric affine on y)
    f = MOI.ScalarNonlinearFunction(
        :+,
        Any[
            MOI.ScalarNonlinearFunction(:*, Any[1.0, p, x, x]),
            MOI.ScalarNonlinearFunction(:*, Any[3.0, y]),
        ],
    )
    parsed = POI._parse_cubic_expression(f, Float64)
    @test parsed !== nothing
    pcf = POI.ParametricCubicFunction(parsed)
    # y should be in affine_data_np since it's not related to any pv or ppv term
    @test haskey(pcf.affine_data_np, y)
    @test pcf.affine_data_np[y] == 3.0
    return
end

# ============================================================================
# MOI Objective Interface Tests
# ============================================================================

function test_cubic_objective_supports()
    model = POI.Optimizer(HiGHS.Optimizer())
    @test MOI.supports(
        model,
        MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}(),
    )
    return
end

function test_cubic_objective_set_invalid_expression()
    model = POI.Optimizer(HiGHS.Optimizer())

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    z = MOI.add_variable(model)

    # x*y*z (3 variables, no parameter) should error
    f = MOI.ScalarNonlinearFunction(:*, Any[x, y, z])
    @test_throws ErrorException MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}(),
        f,
    )
    return
end

function test_cubic_objective_get_no_cache()
    model = POI.Optimizer(HiGHS.Optimizer())

    @test_throws ErrorException MOI.get(
        model,
        MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}(),
    )
    return
end

function test_cubic_objective_get_save_original()
    model = POI.Optimizer(
        HiGHS.Optimizer();
        save_original_objective_and_constraints = true,
    )
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)
    p, _ = MOI.add_constrained_variable(model, MOI.Parameter(2.0))

    # Set a cubic objective: p*x^2
    p_v = POI.v_idx(POI.p_idx(p))
    f = MOI.ScalarNonlinearFunction(:*, Any[1.0, p_v, x, x])
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}(), f)

    # Should be able to retrieve
    retrieved =
        MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}())
    @test retrieved isa MOI.ScalarNonlinearFunction
    return
end

function test_cubic_objective_get_no_save()
    model = POI.Optimizer(
        HiGHS.Optimizer();
        save_original_objective_and_constraints = false,
    )
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)
    p, _ = MOI.add_constrained_variable(model, MOI.Parameter(2.0))

    # Set a cubic objective: p*x^2
    p_v = POI.v_idx(POI.p_idx(p))
    f = MOI.ScalarNonlinearFunction(:*, Any[1.0, p_v, x, x])
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}(), f)

    # Should error since save_original is false
    @test_throws ErrorException MOI.get(
        model,
        MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}(),
    )
    return
end

# ============================================================================
# JuMP Integration Tests - Additional Coverage
# ============================================================================

function test_jump_cubic_mixed_pvv_ppv_ppp()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, 0 <= y <= 10)
    @variable(model, p in MOI.Parameter(1.0))
    @variable(model, q in MOI.Parameter(2.0))

    # Minimize: p*x*y + p*q*x + p*q*p + x^2 + y^2
    # With p=1, q=2: x*y + 2*x + 2 + x^2 + y^2
    @constraint(model, x >= 0)
    @constraint(model, y >= 0)
    @objective(model, Min, p * x * y + p * q * x + p * q * p + x^2 + y^2)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    obj1 = objective_value(model)
    x1 = value(x)
    y1 = value(y)
    # Verify objective matches the formula
    @test obj1 ≈ 1.0 * x1 * y1 + 2.0 * x1 + 2.0 + x1^2 + y1^2 atol = ATOL

    # Change p=2, q=3: 2*x*y + 6*x + 12 + x^2 + y^2
    set_parameter_value(p, 2.0)
    set_parameter_value(q, 3.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    obj2 = objective_value(model)
    x2 = value(x)
    y2 = value(y)
    @test obj2 ≈ 2.0 * x2 * y2 + 6.0 * x2 + 12.0 + x2^2 + y2^2 atol = ATOL
    return
end

function test_jump_cubic_negative_parameters()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, p in MOI.Parameter(-1.0))

    # Minimize: p*x^2 - x
    # With p=-1: -x^2 - x (concave, so min at boundary)
    # Actually this is concave so HiGHS may struggle.
    # Use a positive leading term: x^2 + p*x^2 = (1+p)*x^2
    # With p=-0.5: 0.5*x^2 - x, optimal at x=1, obj=-0.5
    @constraint(model, x >= 0)
    @objective(model, Min, x^2 + p * x^2 - x)

    set_parameter_value(p, -0.5)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -0.5 atol = ATOL
    @test value(x) ≈ 1.0 atol = ATOL

    # Change p to 0: x^2 - x, optimal at x=0.5, obj=-0.25
    set_parameter_value(p, 0.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -0.25 atol = ATOL
    @test value(x) ≈ 0.5 atol = ATOL
    return
end

function test_jump_cubic_multiple_parameter_updates()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, p in MOI.Parameter(1.0))

    @constraint(model, x >= 0)
    @objective(model, Min, p * x^2 - 4 * x)

    # p=1: x^2 - 4x, optimal at x=2, obj=-4
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test value(x) ≈ 2.0 atol = ATOL
    @test objective_value(model) ≈ -4.0 atol = ATOL

    # p=2: 2*x^2 - 4x, optimal at x=1, obj=-2
    set_parameter_value(p, 2.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test value(x) ≈ 1.0 atol = ATOL
    @test objective_value(model) ≈ -2.0 atol = ATOL

    # p=4: 4*x^2 - 4x, optimal at x=0.5, obj=-1
    set_parameter_value(p, 4.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test value(x) ≈ 0.5 atol = ATOL
    @test objective_value(model) ≈ -1.0 atol = ATOL

    # p=0.5: 0.5*x^2 - 4x, optimal at x=4, obj=-8
    set_parameter_value(p, 0.5)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test value(x) ≈ 4.0 atol = ATOL
    @test objective_value(model) ≈ -8.0 atol = ATOL
    return
end

function test_jump_cubic_ppv_negative_params()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(-1.0))
    @variable(model, q in MOI.Parameter(2.0))

    # Minimize: x + p*q*x = x*(1 + p*q) = x*(1-2) = -x
    # Subject to: 0 <= x <= 5
    @constraint(model, x <= 5)
    @objective(model, Min, x + p * q * x)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    # min(-x) with 0<=x<=5 is at x=5, obj=-5+5=-5... wait
    # f(x) = x + (-1)(2)(x) = x - 2x = -x, so min at x=5, obj=-5
    @test objective_value(model) ≈ -5.0 atol = ATOL
    @test value(x) ≈ 5.0 atol = ATOL

    # Change p=1, q=1: x*(1+1) = 2x, min at x=0
    set_parameter_value(p, 1.0)
    set_parameter_value(q, 1.0)
    optimize!(model)
    @test objective_value(model) ≈ 0.0 atol = ATOL
    @test value(x) ≈ 0.0 atol = ATOL
    return
end

function test_jump_cubic_ppp_negative_params()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(-2.0))
    @variable(model, q in MOI.Parameter(3.0))
    @variable(model, r in MOI.Parameter(1.0))

    # Minimize: x + p*q*r = x + (-2)(3)(1) = x - 6
    # Subject to: x >= 1
    @constraint(model, x >= 1)
    @objective(model, Min, x + p * q * r)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -5.0 atol = ATOL
    @test value(x) ≈ 1.0 atol = ATOL

    # Change to positive: p=1,q=1,r=1 -> x+1, min at x=1, obj=2
    set_parameter_value(p, 1.0)
    set_parameter_value(q, 1.0)
    set_parameter_value(r, 1.0)
    optimize!(model)
    @test objective_value(model) ≈ 2.0 atol = ATOL
    return
end

function test_jump_cubic_partial_parameter_update()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(2.0))
    @variable(model, q in MOI.Parameter(3.0))

    # Minimize: x + p*q*x = x*(1 + p*q)
    # With p=2, q=3: x*(1+6)=7x, min at x=0 -> obj=0
    # Subject to: x >= 1
    @constraint(model, x >= 1)
    @objective(model, Min, x + p * q * x)

    optimize!(model)
    @test objective_value(model) ≈ 7.0 atol = ATOL

    # Only update p to 0, keep q=3: x*(1+0)=x, min at x=1 -> obj=1
    set_parameter_value(p, 0.0)
    optimize!(model)
    @test objective_value(model) ≈ 1.0 atol = ATOL
    return
end

function test_jump_cubic_all_term_types_combined()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, 0 <= y <= 10)
    @variable(model, p in MOI.Parameter(1.0))
    @variable(model, q in MOI.Parameter(1.0))

    # f = p*x*y + p*q*x + p*q*p + x^2 + y^2 + p*x + 2*p + 3*x + 5
    # With p=1,q=1:
    #   x*y + x + 1 + x^2 + y^2 + x + 2 + 3x + 5 = x^2 + y^2 + x*y + 5x + 8
    @constraint(model, x >= 0)
    @constraint(model, y >= 0)
    @objective(
        model,
        Min,
        p * x * y +
        p * q * x +
        p * q * p +
        x^2 +
        y^2 +
        p * x +
        2 * p +
        3 * x +
        5,
    )

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    x1 = value(x)
    y1 = value(y)
    obj1 = objective_value(model)
    expected1 = x1^2 + y1^2 + x1 * y1 + 5 * x1 + 8
    @test obj1 ≈ expected1 atol = ATOL

    # p=2, q=0.5:
    #   2*x*y + 1*x + 2 + x^2 + y^2 + 2*x + 4 + 3*x + 5
    #   = x^2 + y^2 + 2*x*y + 6*x + 11
    set_parameter_value(p, 2.0)
    set_parameter_value(q, 0.5)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    x2 = value(x)
    y2 = value(y)
    obj2 = objective_value(model)
    expected2 = x2^2 + y2^2 + 2 * x2 * y2 + 6 * x2 + 11
    @test obj2 ≈ expected2 atol = ATOL
    return
end

function test_jump_cubic_pvv_with_constant_and_affine()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, 0 <= y <= 10)
    @variable(model, p in MOI.Parameter(2.0))

    # Minimize: p*x*y + x^2 + y^2 - 6*x - 6*y + 10
    # With p=2: 2*x*y + x^2 + y^2 - 6x - 6y + 10 = (x+y)^2 - 6(x+y) + 10
    # Let s=x+y. f = s^2 - 6s + 10 = (s-3)^2 + 1
    # Optimal at x+y=3, many solutions (e.g. x=1.5, y=1.5), obj=1
    @constraint(model, x >= 0)
    @constraint(model, y >= 0)
    @objective(model, Min, p * x * y + x^2 + y^2 - 6 * x - 6 * y + 10)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 1.0 atol = ATOL
    @test value(x) + value(y) ≈ 3.0 atol = ATOL

    # Change p=0: x^2 + y^2 - 6x - 6y + 10
    # Optimal at x=3, y=3, obj = 9+9-18-18+10 = -8
    set_parameter_value(p, 0.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -8.0 atol = ATOL
    @test value(x) ≈ 3.0 atol = ATOL
    @test value(y) ≈ 3.0 atol = ATOL
    return
end

function test_jump_cubic_direct_model_ppv()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(3.0))
    @variable(model, q in MOI.Parameter(2.0))

    # Minimize: x + p*q*x = x*(1 + 6) = 7x
    @constraint(model, x >= 1)
    @objective(model, Min, x + p * q * x)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 7.0 atol = ATOL

    # Update p=0: x*(1+0)=x, obj=1
    set_parameter_value(p, 0.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 1.0 atol = ATOL
    return
end

function test_jump_cubic_direct_model_ppp()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(2.0))
    @variable(model, q in MOI.Parameter(3.0))
    @variable(model, r in MOI.Parameter(4.0))

    # Minimize: x + p*q*r = x + 24
    @constraint(model, x >= 1)
    @objective(model, Min, x + p * q * r)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 25.0 atol = ATOL

    set_parameter_value(p, 0.0)
    optimize!(model)
    @test objective_value(model) ≈ 1.0 atol = ATOL
    return
end

function test_jump_cubic_pvv_multiple_cross_terms()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, 0 <= y <= 10)
    @variable(model, 0 <= z <= 10)
    @variable(model, p in MOI.Parameter(1.0))

    # Minimize: x^2 + y^2 + z^2 + p*x*y + p*x*z - 6x - 4y - 2z
    # With p=0: separable quadratics, x=3,y=2,z=1, obj=-9-4-1=-14
    @constraint(model, x >= 0)
    @constraint(model, y >= 0)
    @constraint(model, z >= 0)
    @objective(
        model,
        Min,
        x^2 + y^2 + z^2 + p * x * y + p * x * z - 6 * x - 4 * y - 2 * z,
    )

    set_parameter_value(p, 0.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -14.0 atol = ATOL

    # With p=1: coupled system
    set_parameter_value(p, 1.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    x1 = value(x)
    y1 = value(y)
    z1 = value(z)
    obj = objective_value(model)
    expected = x1^2 + y1^2 + z1^2 + x1 * y1 + x1 * z1 - 6x1 - 4y1 - 2z1
    @test obj ≈ expected atol = ATOL
    return
end

function test_jump_cubic_pp_in_objective()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(3.0))
    @variable(model, q in MOI.Parameter(2.0))

    # Minimize: x + p*q (pp quadratic in parameters contributes to constant)
    # With p=3, q=2: x + 6
    @constraint(model, x >= 1)
    @objective(model, Min, x + p * q)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 7.0 atol = ATOL

    # Change p=0, q=0: x + 0
    set_parameter_value(p, 0.0)
    set_parameter_value(q, 0.0)
    optimize!(model)
    @test objective_value(model) ≈ 1.0 atol = ATOL
    return
end

function test_jump_cubic_pv_in_objective()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, p in MOI.Parameter(2.0))

    # Minimize: x^2 + p*x - 4x = x^2 + (p-4)*x
    # With p=2: x^2 - 2x, optimal at x=1, obj=-1
    @constraint(model, x >= 0)
    @objective(model, Min, x^2 + p * x - 4 * x)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ -1.0 atol = ATOL
    @test value(x) ≈ 1.0 atol = ATOL

    # Change p=6: x^2 + 2x, optimal at x=0, obj=0
    set_parameter_value(p, 6.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 0.0 atol = ATOL
    @test value(x) ≈ 0.0 atol = ATOL
    return
end

function test_jump_cubic_p_affine_in_objective()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(5.0))

    # Minimize: x + 3*p (affine in parameter contributes to constant)
    # With p=5: x + 15
    @constraint(model, x >= 1)
    @objective(model, Min, x + 3 * p)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 16.0 atol = ATOL

    set_parameter_value(p, 0.0)
    optimize!(model)
    @test objective_value(model) ≈ 1.0 atol = ATOL
    return
end

function test_jump_cubic_division_in_ppv()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(4.0))
    @variable(model, q in MOI.Parameter(6.0))

    # Minimize: x + p*q*x/2 = x*(1 + p*q/2)
    # With p=4, q=6: x*(1+12)=13x
    @constraint(model, x >= 1)
    @objective(model, Min, x + p * q * x / 2)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 13.0 atol = ATOL
    return
end

function test_jump_cubic_pvv_update_no_change()
    # Test the case where parameter update results in no actual change
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, p in MOI.Parameter(1.0))

    @constraint(model, x >= 0)
    @objective(model, Min, p * x^2 - 2 * x)

    # First solve
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    obj1 = objective_value(model)

    # "Update" parameter to same value
    set_parameter_value(p, 1.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ obj1 atol = ATOL
    return
end

# ============================================================================
# Parser - Error propagation through subtraction and power
# ============================================================================

function test_cubic_parse_unary_minus_invalid()
    x = MOI.VariableIndex(1)

    # -(sin(x)) should propagate nothing from unary minus
    f = MOI.ScalarNonlinearFunction(
        :-,
        Any[MOI.ScalarNonlinearFunction(:sin, Any[x])],
    )
    result = POI._parse_cubic_expression(f, Float64)
    @test result === nothing
    return
end

function test_cubic_parse_binary_subtraction_invalid_rhs()
    x = MOI.VariableIndex(1)

    # x - sin(x) should propagate nothing from binary subtraction
    f = MOI.ScalarNonlinearFunction(
        :-,
        Any[x, MOI.ScalarNonlinearFunction(:sin, Any[x])],
    )
    result = POI._parse_cubic_expression(f, Float64)
    @test result === nothing
    return
end

function test_cubic_parse_power_of_invalid_base()
    x = MOI.VariableIndex(1)

    # sin(x)^2 should propagate nothing from power
    f = MOI.ScalarNonlinearFunction(
        :^,
        Any[MOI.ScalarNonlinearFunction(:sin, Any[x]), 2],
    )
    result = POI._parse_cubic_expression(f, Float64)
    @test result === nothing
    return
end

# ============================================================================
# Parser - Sorting branches for reverse-ordered indices
# ============================================================================

function test_cubic_parse_pv_param_first()
    # Test the branch where m.variables[1] is already the parameter
    p = POI.v_idx(POI.ParameterIndex(1))
    x = MOI.VariableIndex(1)

    # The monomial from _combine_like_monomials sorts by value, so the
    # variable (lower value) comes first. We construct a monomial explicitly
    # where the parameter is first in the raw expression to trigger both
    # branches of the pv classification.
    # With flat mult: p * x — monomials get combined with sorted key,
    # so let's just verify both orderings work.
    f1 = MOI.ScalarNonlinearFunction(:*, Any[p, x])
    f2 = MOI.ScalarNonlinearFunction(:*, Any[x, p])
    r1 = POI._parse_cubic_expression(f1, Float64)
    r2 = POI._parse_cubic_expression(f2, Float64)

    for r in [r1, r2]
        @test r !== nothing
        @test length(r.pv) == 1
        # Convention: variable_1 = parameter, variable_2 = variable
        @test POI._is_parameter(r.pv[1].variable_1)
        @test !POI._is_parameter(r.pv[1].variable_2)
    end
    return
end

# ============================================================================
# JuMP Integration - pp terms through cubic path
# ============================================================================

function test_jump_cubic_with_pp_terms()
    # To exercise pp handling in _parametric_constant and
    # _delta_parametric_constant, we need pp terms alongside a cubic term
    # so JuMP sends everything as ScalarNonlinearFunction.
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, p in MOI.Parameter(2.0))
    @variable(model, q in MOI.Parameter(3.0))

    # p*x^2 + p*q + x (pvv + pp + v)
    # With p=2, q=3: 2*x^2 + 6 + x
    # Optimal: 2x^2 + x + 6, d/dx = 4x + 1 = 0 -> x = -0.25 (bound at 0)
    # At x=0: obj = 6
    # Actually with 0<=x<=10: min at x=-1/4 but bounded, so x=0, obj=6
    @constraint(model, x >= 0)
    @objective(model, Min, p * x^2 + p * q + x)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 6.0 atol = ATOL
    @test value(x) ≈ 0.0 atol = ATOL

    # Change p=1, q=1: x^2 + 1 + x, at x=0: obj=1
    set_parameter_value(p, 1.0)
    set_parameter_value(q, 1.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 1.0 atol = ATOL
    @test value(x) ≈ 0.0 atol = ATOL
    return
end

function test_jump_cubic_with_pp_same_param()
    # p^2 term alongside a cubic term
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, p in MOI.Parameter(3.0))

    # p*x^2 + p*p + x (pvv + pp + v)
    # With p=3: 3*x^2 + 9 + x, at x=0: obj=9
    @constraint(model, x >= 0)
    @objective(model, Min, p * x^2 + p * p + x)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 9.0 atol = ATOL
    @test value(x) ≈ 0.0 atol = ATOL

    # Change p=0: 0 + 0 + x = x, at x=0: obj=0
    set_parameter_value(p, 0.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 0.0 atol = ATOL
    @test value(x) ≈ 0.0 atol = ATOL
    return
end

function test_jump_cubic_pvv_reverse_var_order()
    # Exercise the v1.value > v2.value sorting branches in
    # _parametric_quadratic_terms and _delta_parametric_quadratic_terms
    # by having pvv with y*x where y.index > x.index
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, -1 <= y <= 10)
    @variable(model, p in MOI.Parameter(2.0))

    # Minimize: x^2 + y^2 + p*y*x - 3x (y comes before x in term)
    @constraint(model, x + y >= 0)
    @objective(model, Min, x^2 + y^2 + p * y * x - 3 * x)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    x1 = value(x)
    y1 = value(y)
    @test objective_value(model) ≈ x1^2 + y1^2 + 2 * y1 * x1 - 3 * x1 atol =
        ATOL

    # Change p=0
    set_parameter_value(p, 0.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test value(x) ≈ 3 / 2 atol = ATOL
    @test value(y) ≈ 0.0 atol = ATOL
    return
end

function test_jump_cubic_p_affine_in_cubic_expr()
    # Exercise p affine terms inside a cubic expression
    # (p terms alongside pvv so it goes through cubic path)
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, p in MOI.Parameter(5.0))

    # p*x^2 + 3*p + x (pvv + p_affine + v)
    # With p=5: 5*x^2 + 15 + x, at x=0: obj=15
    @constraint(model, x >= 0)
    @objective(model, Min, p * x^2 + 3 * p + x)

    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 15.0 atol = ATOL
    @test value(x) ≈ 0.0 atol = ATOL

    # Change p=0: x, at x=0: obj=0
    set_parameter_value(p, 0.0)
    optimize!(model)
    @test termination_status(model) in (OPTIMAL, LOCALLY_SOLVED)
    @test objective_value(model) ≈ 0.0 atol = ATOL
    return
end

function test_cubic_objective_set_error_on_inner_optimizer()
    mock = NoQuadObjModel()
    model = POI.Optimizer(mock)

    x = MOI.add_variable(model)
    p, _ = MOI.add_constrained_variable(model, MOI.Parameter(2.0))
    p_v = POI.v_idx(POI.p_idx(p))

    # p * x^2 — parsed successfully but inner optimizer rejects SQF
    f = MOI.ScalarNonlinearFunction(:*, Any[1.0, p_v, x, x])
    err = try
        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}(), f)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("Failed to set cubic objective function", err.msg)
    @test occursin("Quadratic objectives not supported", err.msg)
    return
end

end  # module

TestCubic.runtests()
