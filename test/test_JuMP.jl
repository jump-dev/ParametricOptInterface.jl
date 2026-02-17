# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestJuMPTests

using Test
using JuMP

import HiGHS
import Ipopt
import LinearAlgebra
import ParametricOptInterface as POI
import SCS

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

function canonical_compare(f1, f2)
    return MOI.Utilities.canonical(f1) ≈ MOI.Utilities.canonical(f2)
end

function test_jump_direct_affine_parameters()
    optimizer = POI.Optimizer(HiGHS.Optimizer)
    model = direct_model(optimizer)
    set_silent(model)
    @variable(model, x[i=1:2] >= 0)
    @variable(model, y in Parameter(0.0))
    @variable(model, w in Parameter(0.0))
    @variable(model, z in Parameter(0.0))
    @constraint(model, 2 * x[1] + x[2] + y <= 4)
    @constraint(model, 1 * x[1] + 2 * x[2] + z <= 4)
    @objective(model, Max, 4 * x[1] + 3 * x[2] + w)
    optimize!(model)
    @test isapprox.(value(x[1]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(x[2]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(y), 0, atol = ATOL)
    # ===== Set parameter value =====
    MOI.set(model, POI.ParameterValue(), y, 2.0)
    optimize!(model)
    @test isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test isapprox.(value(y), 2.0, atol = ATOL)
    return
end

function test_jump_direct_parameter_times_variable()
    optimizer = POI.Optimizer(HiGHS.Optimizer)
    model = direct_model(optimizer)
    set_silent(model)
    @variable(model, x[i=1:2] >= 0)
    @variable(model, y in Parameter(0.0))
    @variable(model, w in Parameter(0.0))
    @variable(model, z in Parameter(0.0))
    @constraint(model, 2 * x[1] + x[2] + y <= 4)
    @constraint(model, (1 + y) * x[1] + 2 * x[2] + z <= 4)
    @objective(model, Max, 4 * x[1] + 3 * x[2] + w)
    optimize!(model)
    @test isapprox.(value(x[1]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(x[2]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(y), 0, atol = ATOL)
    # ===== Set parameter value =====
    MOI.set(model, POI.ParameterValue(), y, 2.0)
    optimize!(model)
    @test isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test isapprox.(value(y), 2.0, atol = ATOL)
    return
end

function test_jump_affine_parameters()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x[i=1:2] >= 0)
    @variable(model, y in Parameter(0.0))
    @variable(model, w in Parameter(0.0))
    @variable(model, z in Parameter(0.0))
    @constraint(model, 2 * x[1] + x[2] + y <= 4)
    @constraint(model, 1 * x[1] + 2 * x[2] + z <= 4)
    @objective(model, Max, 4 * x[1] + 3 * x[2] + w)
    optimize!(model)
    @test isapprox.(value(x[1]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(x[2]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(y), 0, atol = ATOL)
    # ===== Set parameter value =====
    MOI.set(model, POI.ParameterValue(), y, 2.0)
    optimize!(model)
    @test isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test isapprox.(value(y), 2.0, atol = ATOL)
    return
end

function test_jump_parameter_times_variable()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x[i=1:2] >= 0)
    @variable(model, y in Parameter(0.0))
    @variable(model, w in Parameter(0.0))
    @variable(model, z in Parameter(0.0))
    @test MOI.get(model, POI.ParameterValue(), y) == 0
    @constraint(model, 2 * x[1] + x[2] + y <= 4)
    @constraint(model, (1 + y) * x[1] + 2 * x[2] + z <= 4)
    @objective(model, Max, 4 * x[1] + 3 * x[2] + w)
    optimize!(model)
    @test isapprox.(value(x[1]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(x[2]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(y), 0, atol = ATOL)
    # ===== Set parameter value =====
    MOI.set(model, POI.ParameterValue(), y, 2.0)
    optimize!(model)
    @test isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test isapprox.(value(y), 2.0, atol = ATOL)
    return
end

function test_jump_constraintfunction_getter()
    model = direct_model(
        POI.Optimizer(
            MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
                MOI.Utilities.AUTOMATIC,
            ),
        ),
    )
    vx = @variable(model, x[i=1:2])
    vp = @variable(model, p[i=1:2] in Parameter(-1.0))
    c1 = @constraint(model, con, sum(x) + sum(p) >= 1)
    c2 = @constraint(model, conq, sum(x .* p) >= 1)
    c3 = @constraint(model, conqa, sum(x .* p) + x[1]^2 + x[1] + p[1] >= 1)
    @test MOI.Utilities.canonical(
        MOI.get(model, MOI.ConstraintFunction(), c1),
    ) ≈ MOI.Utilities.canonical(
        MOI.ScalarAffineFunction{Float64}(
            [
                MOI.ScalarAffineTerm{Float64}(1.0, MOI.VariableIndex(1)),
                MOI.ScalarAffineTerm{Float64}(1.0, MOI.VariableIndex(2)),
                MOI.ScalarAffineTerm{Float64}(
                    1.0,
                    MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1),
                ),
                MOI.ScalarAffineTerm{Float64}(
                    1.0,
                    MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 2),
                ),
            ],
            0.0,
        ),
    )
    @test canonical_compare(
        MOI.get(model, MOI.ConstraintFunction(), c2),
        MOI.ScalarQuadraticFunction{Float64}(
            [
                MOI.ScalarQuadraticTerm{Float64}(
                    1.0,
                    MOI.VariableIndex(1),
                    MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1),
                ),
                MOI.ScalarQuadraticTerm{Float64}(
                    1.0,
                    MOI.VariableIndex(2),
                    MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 2),
                ),
            ],
            [],
            0.0,
        ),
    )
    @test canonical_compare(
        MOI.get(model, MOI.ConstraintFunction(), c3),
        MOI.ScalarQuadraticFunction{Float64}(
            [
                MOI.ScalarQuadraticTerm{Float64}(
                    1.0,
                    MOI.VariableIndex(1),
                    MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1),
                ),
                MOI.ScalarQuadraticTerm{Float64}(
                    1.0,
                    MOI.VariableIndex(2),
                    MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 2),
                ),
                MOI.ScalarQuadraticTerm{Float64}(
                    2.0,
                    MOI.VariableIndex(1),
                    MOI.VariableIndex(1),
                ),
            ],
            [
                MOI.ScalarAffineTerm{Float64}(1.0, MOI.VariableIndex(1)),
                MOI.ScalarAffineTerm{Float64}(
                    1.0,
                    MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1),
                ),
            ],
            0.0,
        ),
    )
    o1 = @objective(model, Min, sum(x) + sum(p))
    F = MOI.get(model, MOI.ObjectiveFunctionType())
    @test canonical_compare(
        MOI.get(model, MOI.ObjectiveFunction{F}()),
        MOI.ScalarAffineFunction{Float64}(
            [
                MOI.ScalarAffineTerm{Float64}(1.0, MOI.VariableIndex(1)),
                MOI.ScalarAffineTerm{Float64}(1.0, MOI.VariableIndex(2)),
                MOI.ScalarAffineTerm{Float64}(
                    1.0,
                    MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1),
                ),
                MOI.ScalarAffineTerm{Float64}(
                    1.0,
                    MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 2),
                ),
            ],
            0.0,
        ),
    )
    o2 = @objective(model, Min, sum(x .* p) + 2)
    F = MOI.get(model, MOI.ObjectiveFunctionType())
    f = MOI.get(model, MOI.ObjectiveFunction{F}())
    f_ref = MOI.ScalarQuadraticFunction{Float64}(
        [
            MOI.ScalarQuadraticTerm{Float64}(
                1.0,
                MOI.VariableIndex(1),
                MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1),
            ),
            MOI.ScalarQuadraticTerm{Float64}(
                1.0,
                MOI.VariableIndex(2),
                MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 2),
            ),
        ],
        [],
        2.0,
    )
    @test canonical_compare(f, f_ref)
    o3 = @objective(model, Min, sum(x .* p) + x[1]^2 + x[1] + p[1])
    F = MOI.get(model, MOI.ObjectiveFunctionType())
    @test canonical_compare(
        MOI.get(model, MOI.ObjectiveFunction{F}()),
        MOI.ScalarQuadraticFunction{Float64}(
            [
                MOI.ScalarQuadraticTerm{Float64}(
                    1.0,
                    MOI.VariableIndex(1),
                    MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1),
                ),
                MOI.ScalarQuadraticTerm{Float64}(
                    1.0,
                    MOI.VariableIndex(2),
                    MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 2),
                ),
                MOI.ScalarQuadraticTerm{Float64}(
                    2.0,
                    MOI.VariableIndex(1),
                    MOI.VariableIndex(1),
                ),
            ],
            [
                MOI.ScalarAffineTerm{Float64}(1.0, MOI.VariableIndex(1)),
                MOI.ScalarAffineTerm{Float64}(
                    1.0,
                    MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1),
                ),
            ],
            0.0,
        ),
    )
    return
end

function test_jump_interpret_parameteric_bounds()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)
    @variable(model, x[i=1:2])
    @variable(model, p[i=1:2] in Parameter.(-1.0))
    @constraint(model, [i in 1:2], x[i] >= p[i])
    @objective(model, Min, sum(x))
    optimize!(model)
    expected = Tuple{Type,Type}[
        (MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}),
        (MOI.VariableIndex, MOI.Parameter{Float64}),
    ]
    result = MOI.get(model, MOI.ListOfConstraintTypesPresent())
    @test Set(result) == Set(expected)
    @test length(result) == length(expected)
    expected = Tuple{Type,Type}[(MOI.VariableIndex, MOI.GreaterThan{Float64})]
    result = MOI.get(
        backend(model).optimizer.model.optimizer,
        MOI.ListOfConstraintTypesPresent(),
    )
    @test Set(result) == Set(expected)
    @test length(result) == length(expected)
    @test objective_value(model) == -2
    MOI.set(model, POI.ParameterValue(), p[1], 4.0)
    optimize!(model)
    @test objective_value(model) == 3
    return
end

function test_jump_interpret_parameteric_bounds_expression()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)
    @variable(model, x[i=1:2])
    @variable(model, p[i=1:2] in Parameter.(-1.0))
    @constraint(model, [i in 1:2], x[i] >= p[i] + p[1])
    @objective(model, Min, sum(x))
    optimize!(model)
    expected = Tuple{Type,Type}[
        (MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}),
        (MOI.VariableIndex, MOI.Parameter{Float64}),
    ]
    result = MOI.get(model, MOI.ListOfConstraintTypesPresent())
    @test Set(result) == Set(expected)
    @test length(result) == length(expected)
    expected = Tuple{Type,Type}[(MOI.VariableIndex, MOI.GreaterThan{Float64})]
    result = MOI.get(
        backend(model).optimizer.model.optimizer,
        MOI.ListOfConstraintTypesPresent(),
    )
    @test Set(result) == Set(expected)
    @test length(result) == length(expected)
    @test objective_value(model) == -4
    MOI.set(model, POI.ParameterValue(), p[1], 4.0)
    optimize!(model)
    @test objective_value(model) == 11.0
    return
end

function test_jump_direct_interpret_parameteric_bounds()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)
    @variable(model, x[i=1:2])
    @variable(model, p[i=1:2] in Parameter(-1.0))
    @constraint(model, [i in 1:2], x[i] >= p[i])
    @objective(model, Min, sum(x))
    optimize!(model)
    expected = Tuple{Type,Type}[
        (MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}),
        (MOI.VariableIndex, MOI.Parameter{Float64}),
    ]
    result = MOI.get(model, MOI.ListOfConstraintTypesPresent())
    @test Set(result) == Set(expected)
    @test length(result) == length(expected)
    expected = Tuple{Type,Type}[(MOI.VariableIndex, MOI.GreaterThan{Float64})]
    result =
        MOI.get(backend(model).optimizer, MOI.ListOfConstraintTypesPresent())
    @test Set(result) == Set(expected)
    @test length(result) == length(expected)
    @test objective_value(model) == -2
    MOI.set(model, POI.ParameterValue(), p[1], 4.0)
    optimize!(model)
    @test objective_value(model) == 3
    return
end

function test_jump_direct_interpret_parameteric_bounds_no_interpretation()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_CONSTRAINTS)
    @variable(model, x[i=1:2])
    @variable(model, p[i=1:2] in Parameter(-1.0))
    @constraint(model, [i in 1:2], x[i] >= p[i])
    @objective(model, Min, sum(x))
    optimize!(model)
    expected = Tuple{Type,Type}[
        (MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}),
        (MOI.VariableIndex, MOI.Parameter{Float64}),
    ]
    result = MOI.get(model, MOI.ListOfConstraintTypesPresent())
    @test Set(result) == Set(expected)
    @test length(result) == length(expected)
    expected = Tuple{Type,Type}[(
        MOI.ScalarAffineFunction{Float64},
        MOI.GreaterThan{Float64},
    ),]
    result =
        MOI.get(backend(model).optimizer, MOI.ListOfConstraintTypesPresent())
    @test Set(result) == Set(expected)
    @test length(result) == length(expected)
    @test objective_value(model) == -2
    MOI.set(model, POI.ParameterValue(), p[1], 4.0)
    optimize!(model)
    @test objective_value(model) == 3
    return
end

function test_jump_direct_interpret_parameteric_bounds_change()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)
    @variable(model, x[i=1:2])
    @variable(model, p[i=1:2] in Parameter(-1.0))
    @constraint(model, [i in 1:2], x[i] >= p[i])
    @test_throws ErrorException @constraint(model, [i in 1:2], 2x[i] >= p[i])
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_CONSTRAINTS)
    @constraint(model, [i in 1:2], 2x[i] >= p[i])
    @objective(model, Min, sum(x))
    optimize!(model)
    @test objective_value(model) == -1
    MOI.set(model, POI.ParameterValue(), p[1], 4.0)
    optimize!(model)
    @test objective_value(model) == 3.5
    return
end

function test_jump_direct_interpret_parameteric_bounds_both()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    MOI.set(model, POI.ConstraintsInterpretation(), POI.BOUNDS_AND_CONSTRAINTS)
    @variable(model, x[i=1:2])
    @variable(model, p[i=1:2] in Parameter(-1.0))
    @constraint(model, [i in 1:2], x[i] >= p[i])
    @constraint(model, [i in 1:2], 2x[i] >= p[i])
    @objective(model, Min, sum(x))
    optimize!(model)
    @test objective_value(model) == -1
    MOI.set(model, POI.ParameterValue(), p[1], 4.0)
    optimize!(model)
    @test objective_value(model) == 3.5
    return
end

function test_jump_direct_interpret_parameteric_bounds_invalid()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)
    @variable(model, x[i=1:2])
    @variable(model, p[i=1:2] in Parameter(-1.0))
    @test_throws ErrorException @constraint(
        model,
        [i in 1:2],
        2x[i] >= p[i] + p[1]
    )
    return
end

function test_jump_set_variable_start_value()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, p in Parameter(0.0))
    set_start_value(x, 1.0)
    @test start_value(x) == 1
    @test_throws ErrorException(
        "The parameter $(index(p)) value is 0.0, but trying to set VariablePrimalStart 1.0",
    ) set_start_value(p, 1.0)
    @test start_value(p) == 0.0
    return
end

function test_jump_direct_get_parameter_value()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x, lower_bound = 0.0, upper_bound = 10.0)
    @variable(model, y, binary = true)
    @variable(model, z, set = MOI.Parameter(10.0))
    c = @constraint(model, 19.0 * x - z + 22.0 * y <= 1.0)
    @objective(model, Min, x + y)
    @test MOI.get(model, POI.ParameterValue(), z) == 10
    return
end

function test_jump_get_parameter_value()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x, lower_bound = 0.0, upper_bound = 10.0)
    @variable(model, y, binary = true)
    @variable(model, z, set = MOI.Parameter(10))
    c = @constraint(model, 19.0 * x - z + 22.0 * y <= 1.0)
    @objective(model, Min, x + y)
    @test MOI.get(model, POI.ParameterValue(), z) == 10
    return
end

function test_jump_sdp_scalar_parameter()
    m = Model(() -> POI.Optimizer(SCS.Optimizer))
    set_silent(m)
    @variable(m, p in Parameter(0.0))
    @variable(m, x[1:2, 1:2], Symmetric)
    @objective(m, Min, x[1, 1] + x[2, 2])
    @constraint(m, LinearAlgebra.Symmetric(x .- [1+p 0; 0 1+p]) in PSDCone())
    optimize!(m)
    @test all(isapprox.(value.(x), [1 0; 0 1], atol = ATOL))
    MOI.set(m, POI.ParameterValue(), p, 1)
    optimize!(m)
    @test all(isapprox.(value.(x), [2 0; 0 2], atol = ATOL))
    return
end

function test_jump_sdp_matrix_parameter()
    m = Model(() -> POI.Optimizer(SCS.Optimizer))
    set_silent(m)
    P1 = [1 2; 2 3]
    @variable(m, p[1:2, 1:2] in Parameter.(P1))
    @variable(m, x[1:2, 1:2], Symmetric)
    @objective(m, Min, x[1, 1] + x[2, 2])
    @constraint(m, LinearAlgebra.Symmetric(x - p) in PSDCone())
    optimize!(m)
    @test all(isapprox.(value.(x), P1, atol = ATOL))
    P2 = [1 2; 2 1]
    MOI.set.(m, POI.ParameterValue(), p, P2)
    optimize!(m)
    @test all(isapprox.(value.(x), P2, atol = ATOL))
    return
end

function test_jump_dual_basic()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x[1:2] in Parameter.(ones(2) .* 4.0))
    @variable(model, y[1:6])
    @constraint(model, ctr1, 3 * y[1] >= 2 - 7 * x[1])
    @objective(model, Min, 5 * y[1])
    optimize!(model)
    @test 5 / 3 ≈ dual(ctr1) atol = 1e-3
    @test [-35 / 3, 0.0] ≈ MOI.get.(model, POI.ParameterDual(), x) atol = 1e-3
    @test [-26 / 3, 0.0, 0.0, 0.0, 0.0, 0.0] ≈ value.(y) atol = 1e-3
    @test -130 / 3 ≈ objective_value(model) atol = 1e-3
    return
end

function test_jump_dual_multiplicative_fail()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    @constraint(model, cons, x * p >= 3)
    @objective(model, Min, 2x)
    optimize!(model)
    err = MOI.GetAttributeNotAllowed(
        MOI.ConstraintDual(),
        "Cannot compute the dual of a multiplicative parameter",
    )
    @test_throws err MOI.get(model, POI.ParameterDual(), p)
    return
end

function test_jump_dual_objective_min()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    @constraint(model, cons, x >= 3 * p)
    @objective(model, Min, 2x + p)
    optimize!(model)
    @test MOI.get(model, POI.ParameterDual(), p) == 7
    return
end

function test_jump_dual_objective_max()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    @constraint(model, cons, x >= 3 * p)
    @objective(model, Max, -2x + p)
    optimize!(model)
    @test MOI.get(model, POI.ParameterDual(), p) == 5
    return
end

function test_jump_dual_multiple_parameters_1()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x[1:6] in Parameter.(ones(6) .* 4.0))
    @variable(model, y[1:6] <= 0.0)
    @constraint(model, ctr1, 3 * y[1] >= 2 - 7 * x[3])
    @constraint(model, ctr2, 3 * y[1] >= 2 - 7 * x[3])
    @constraint(model, ctr3, 3 * y[1] >= 2 - 7 * x[3])
    @constraint(model, ctr4, 3 * y[1] >= 2 - 7 * x[3])
    @constraint(model, ctr5, 3 * y[1] >= 2 - 7 * x[3])
    @constraint(model, ctr6, 3 * y[1] >= 2 - 7 * x[3])
    @constraint(model, ctr7, sum(3 * y[i] + x[i] for i in 2:4) >= 2 - 7 * x[3])
    @constraint(
        model,
        ctr8,
        sum(3 * y[i] + 7.0 * x[i] - x[i] for i in 2:4) >= 2 - 7 * x[3]
    )
    @objective(model, Min, 5 * y[1])
    optimize!(model)
    @test 5 / 3 ≈
          dual(ctr1) +
          dual(ctr2) +
          dual(ctr3) +
          dual(ctr4) +
          dual(ctr5) +
          dual(ctr6) atol = 1e-3
    @test 0.0 ≈ dual(ctr7) atol = 1e-3
    @test 0.0 ≈ dual(ctr8) atol = 1e-3
    @test [0.0, 0.0, -35 / 3, 0.0, 0.0, 0.0] ≈
          MOI.get.(model, POI.ParameterDual(), x) atol = 1e-3
    @test [-26 / 3, 0.0, 0.0, 0.0, 0.0, 0.0] ≈ value.(y) atol = 1e-3
    @test -130 / 3 ≈ objective_value(model) atol = 1e-3
    return
end

function test_jump_duals_LessThan()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, α in Parameter(-1.0))
    @variable(model, x)
    cref = @constraint(model, x ≤ α)
    @objective(model, Max, x)
    optimize!(model)
    @test value(x) == -1.0
    @test dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0

    MOI.set(model, POI.ParameterValue(), α, 2.0)
    optimize!(model)
    @test value(x) == 2.0
    @test dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
    return
end

function test_jump_duals_EqualTo()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, α in Parameter(-1.0))
    @variable(model, x)
    cref = @constraint(model, x == α)
    @objective(model, Max, x)
    optimize!(model)
    @test value(x) == -1.0
    @test dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
    MOI.set(model, POI.ParameterValue(), α, 2.0)
    optimize!(model)
    @test value(x) == 2.0
    @test dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
    return
end

function test_jump_duals_GreaterThan()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, α in Parameter(1.0))
    MOI.set(model, POI.ParameterValue(), α, -1.0)
    @variable(model, x)
    cref = @constraint(model, x >= α)
    @objective(model, Min, x)
    optimize!(model)
    @test value(x) == -1.0
    @test dual(cref) == 1.0
    @test MOI.get(model, POI.ParameterDual(), α) == 1.0
    MOI.set(model, POI.ParameterValue(), α, 2.0)
    optimize!(model)
    @test value(x) == 2.0
    @test dual(cref) == 1.0
    @test MOI.get(model, POI.ParameterDual(), α) == 1.0
    return
end

function test_jump_dual_multiple_parameters_2()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, α[1:10] in Parameter.(ones(10)))
    @variable(model, x)
    cref = @constraint(model, x == sum(2 * α[i] for i in 1:10))
    @objective(model, Min, x)
    optimize!(model)
    @test value(x) == 20.0
    @test dual(cref) == 1.0
    @test MOI.get(model, POI.ParameterDual(), α[3]) == 2.0
    return
end

function test_jump_dual_mixing_params_and_vars_1()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, α[1:5] in Parameter.(ones(5)))
    @variable(model, x)
    cref = @constraint(model, sum(x for i in 1:5) == sum(2 * α[i] for i in 1:5))
    @objective(model, Min, x)
    optimize!(model)
    @test value(x) == 2.0
    @test dual(cref) == 1 / 5
    @test MOI.get(model, POI.ParameterDual(), α[3]) == 2 / 5
    return
end

function test_jump_dual_mixing_params_and_vars_2()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, α[1:5] in Parameter.(ones(5)))
    @variable(model, x)
    cref = @constraint(model, 0.0 == sum(-x + 2 * α[i] for i in 1:5))
    @objective(model, Min, x)
    optimize!(model)
    @test value(x) == 2.0
    @test dual(cref) == 1 / 5
    @test MOI.get(model, POI.ParameterDual(), α[3]) == 2 / 5
    return
end

function test_jump_dual_mixing_params_and_vars_3()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, α[1:5] in Parameter.(ones(5)))
    @variable(model, x)
    cref = @constraint(model, 0.0 == sum(-x + 2.0 + 2 * α[i] for i in 1:5))
    @objective(model, Min, x)
    optimize!(model)
    @test value(x) == 4.0
    @test dual(cref) == 1 / 5
    @test MOI.get(model, POI.ParameterDual(), α[3]) == 2 / 5
    return
end

function test_jump_dual_add_after_solve()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, α in Parameter(1.0))
    MOI.set(model, POI.ParameterValue(), α, -1.0)
    @variable(model, x)
    cref = @constraint(model, x <= α)
    @objective(model, Max, x)
    optimize!(model)
    @test value(x) == -1.0
    @test dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
    @variable(model, b in Parameter(-2.0))
    cref = @constraint(model, x <= b)
    optimize!(model)
    @test value(x) == -2.0
    @test dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == 0.0
    @test MOI.get(model, POI.ParameterDual(), b) == -1.0
    return
end

function test_jump_dual_add_ctr_alaternative()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, α in Parameter(-1.0))
    @variable(model, x)
    exp = x - α
    cref = @constraint(model, exp ≤ 0)
    @objective(model, Max, x)
    optimize!(model)
    @test value(x) == -1.0
    @test dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
    return
end

function test_jump_dual_delete_constraint()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, α in Parameter(-1.0))
    @variable(model, x)
    cref1 = @constraint(model, x ≤ α / 2)
    cref2 = @constraint(model, x ≤ α)
    cref3 = @constraint(model, x ≤ 2α)
    @objective(model, Max, x)
    delete(model, cref3)
    optimize!(model)
    @test value(x) == -1.0
    @test dual(cref1) == 0.0
    @test dual(cref2) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
    delete(model, cref2)
    optimize!(model)
    @test value(x) == -0.5
    @test dual(cref1) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -0.5
    return
end

function test_jump_dual_delete_constraint_2()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, α in Parameter(1.0))
    @variable(model, β in Parameter(0.0))
    @variable(model, x)
    list = []
    cref = @constraint(model, x >= 1 * 1)
    push!(list, cref)
    cref = @constraint(model, x >= 9 * α)
    push!(list, cref)
    cref = @constraint(model, x >= 8 * α + β^2)
    push!(list, cref)
    cref = @constraint(model, x >= 7 * 1)
    push!(list, cref)
    cref = @constraint(model, x >= 6 * α)
    push!(list, cref)
    cref = @constraint(model, x >= 5 * α + β^2)
    push!(list, cref)
    cref = @constraint(model, x >= 4 * 1)
    push!(list, cref)
    cref = @constraint(model, x >= 3 * α)
    push!(list, cref)
    cref = @constraint(model, x >= 2 * α + β^2)
    push!(list, cref)
    @objective(model, Min, x)
    cref1 = popfirst!(list)
    for i in 9:-1:2
        optimize!(model)
        @test value(x) == 1.0 * i
        @test dual(cref1) == 0.0
        for con in list[2:end]
            @test dual(con) == 0.0
        end
        @test dual(list[1]) == 1.0
        if i in [7, 4]
            @test MOI.get(model, POI.ParameterDual(), α) == 0.0
        else
            @test MOI.get(model, POI.ParameterDual(), α) == 1.0 * i
        end
        con = popfirst!(list)
        delete(model, con)
    end
    return
end

function test_jump_dual_delete_constraint_3()
    model = direct_model(POI.Optimizer(SCS.Optimizer))
    set_silent(model)
    list = []
    @variable(model, α in Parameter(1.0))
    @variable(model, β in Parameter(0.0))
    @variable(model, x)
    cref = @constraint(model, [x - 1 * 1] in MOI.Nonnegatives(1))
    push!(list, cref)
    cref = @constraint(model, [x - 9 * α] in MOI.Nonnegatives(1))
    push!(list, cref)
    # cref = @constraint(model, [x - 8 * α + β^2] in MOI.Nonnegatives(1))
    cref = @constraint(model, [x - 8 * α + β] in MOI.Nonnegatives(1))
    push!(list, cref)
    cref = @constraint(model, [x - 7 * 1] in MOI.Nonnegatives(1))
    push!(list, cref)
    cref = @constraint(model, [x - 6 * α] in MOI.Nonnegatives(1))
    push!(list, cref)
    # cref = @constraint(model, [x - 5 * α + β^2] in MOI.Nonnegatives(1))
    cref = @constraint(model, [x - 5 * α + β] in MOI.Nonnegatives(1))
    push!(list, cref)
    cref = @constraint(model, [x - 4 * 1] in MOI.Nonnegatives(1))
    push!(list, cref)
    cref = @constraint(model, [x - 3 * α] in MOI.Nonnegatives(1))
    push!(list, cref)
    # cref = @constraint(model, [x - 2 * α + β^2] in MOI.Nonnegatives(1))
    cref = @constraint(model, [x - 2 * α + β] in MOI.Nonnegatives(1))
    push!(list, cref)
    @objective(model, Min, 1.0 * x)
    cref1 = popfirst!(list)
    for i in 9:-1:2
        optimize!(model)
        @test value(x) ≈ 1.0 * i atol = 1e-5
        @test dual(cref1)[] ≈ 0.0 atol = 1e-5
        for con in list[2:end]
            @test dual(con)[] ≈ 0.0 atol = 1e-5
        end
        @test dual(list[1])[] ≈ 1.0 atol = 1e-5
        if i in [7, 4]
            @test MOI.get(model, POI.ParameterDual(), α) ≈ 0.0 atol = 1e-5
        else
            @test MOI.get(model, POI.ParameterDual(), α) ≈ 1.0 * i atol = 1e-5
        end
        con = popfirst!(list)
        delete(model, con)
    end
    return
end

function test_jump_nlp()
    model = Model(() -> POI.Optimizer(Ipopt.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, z in Parameter(10.0))
    @constraint(model, x >= z)
    @NLobjective(model, Min, x^2)
    @test_throws ErrorException optimize!(model)
    return
end

function test_jump_direct_vector_parameter_affine_nonnegatives()
    """
        min x + y
            x - t + 1 >= 0
            y - t + 2 >= 0

        opt
            x* = t-1
            y* = t-2
            obj = 2*t-3
    """
    model = direct_model(POI.Optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, y)
    @variable(model, t in Parameter(5.0))
    @constraint(model, [(x - t + 1), (y - t + 2)...] in MOI.Nonnegatives(2))
    @objective(model, Min, x + y)
    optimize!(model)
    @test isapprox.(value(x), 4.0, atol = ATOL)
    @test isapprox.(value(y), 3.0, atol = ATOL)
    MOI.set(model, POI.ParameterValue(), t, 6)
    optimize!(model)
    @test isapprox.(value(x), 5.0, atol = ATOL)
    @test isapprox.(value(y), 4.0, atol = ATOL)
    return
end

function test_jump_direct_vector_parameter_affine_nonpositives()
    """
        min x + y
            - x + t - 1 ≤ 0
            - y + t - 2 ≤ 0

        opt
            x* = t-1
            y* = t-2
            obj = 2*t-3
    """
    model = Model(() -> POI.Optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, y)
    @variable(model, t in Parameter(5.0))
    @constraint(model, [(-x + t - 1), (-y + t - 2)...] in MOI.Nonpositives(2))
    @objective(model, Min, x + y)
    optimize!(model)
    @test isapprox.(value(x), 4.0, atol = ATOL)
    @test isapprox.(value(y), 3.0, atol = ATOL)
    MOI.set(model, POI.ParameterValue(), t, 6)
    optimize!(model)
    @test isapprox.(value(x), 5.0, atol = ATOL)
    @test isapprox.(value(y), 4.0, atol = ATOL)
    return
end

function test_jump_direct_soc_parameters()
    """
        Problem SOC2 from MOI

        min  x
        s.t. y ≥ 1/√2
            (x-p)² + y² ≤ 1

        in conic form:

        min  x
        s.t.  -1/√2 + y ∈ R₊
            1 - t ∈ {0}
            (t, x-p ,y) ∈ SOC₃

        opt
            x* = p - 1/√2
            y* = 1/√2
    """
    model = direct_model(POI.Optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, y)
    @variable(model, t)
    @variable(model, p in Parameter(0.0))
    @constraint(model, [y - 1 / √2] in MOI.Nonnegatives(1))
    @constraint(model, [t - 1] in MOI.Zeros(1))
    @constraint(model, [t, (x - p), y...] in SecondOrderCone())
    @objective(model, Min, 1.0 * x)
    optimize!(model)
    @test objective_value(model) ≈ -1 / √2 atol = ATOL
    @test value(x) ≈ -1 / √2 atol = ATOL
    @test value(y) ≈ 1 / √2 atol = ATOL
    @test value(t) ≈ 1 atol = ATOL
    MOI.set(model, POI.ParameterValue(), p, 1)
    optimize!(model)
    @test objective_value(model) ≈ 1 - 1 / √2 atol = ATOL
    @test value(x) ≈ 1 - 1 / √2 atol = ATOL
    return
end

function test_jump_direct_rsoc_constraints()
    """
        Problem RSOC
        min  x
        s.t. y ≥ 1/√2
            x² + (y-p)² ≤ 1
        in conic form:
        min  x
        s.t.  -1/√2 + y ∈ R₊
            1 - t ∈ {0}
            (t, x ,y-p) ∈ RSOC
        opt
            x* = 1/2*(max{1/√2,p}-p)^2
            y* = max{1/√2,p}
    """
    model = Model(() -> POI.Optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, y)
    @variable(model, t)
    @variable(model, p in Parameter(0.0))
    @constraint(model, [y - 1 / √2] in MOI.Nonnegatives(1))
    @constraint(model, [t - 1] in MOI.Zeros(1))
    @constraint(model, [t, x, y - p] in RotatedSecondOrderCone())
    @objective(model, Min, 1.0 * x)
    optimize!(model)
    @test objective_value(model) ≈ 1 / 4 atol = ATOL
    @test value(x) ≈ 1 / 4 atol = ATOL
    @test value(y) ≈ 1 / √2 atol = ATOL
    @test value(t) ≈ 1 atol = ATOL
    MOI.set(model, POI.ParameterValue(), p, 2)
    optimize!(model)
    @test objective_value(model) ≈ 0.0 atol = ATOL
    @test value(x) ≈ 0.0 atol = ATOL
    @test value(y) ≈ 2 atol = ATOL
    return
end

function test_jump_quadratic_interval()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, p in Parameter(10.0))
    @variable(model, q in Parameter(4.0))
    @constraint(model, 0 <= x - p * y + q <= 0)
    @objective(model, Min, x + y)
    optimize!(model)
    @test value(x) ≈ 0 atol = ATOL
    @test value(y) ≈ 0.4 atol = ATOL
    MOI.set(model, POI.ParameterValue(), p, 20.0)
    optimize!(model)
    @test value(x) ≈ 0 atol = ATOL
    @test value(y) ≈ 0.2 atol = ATOL
    MOI.set(model, POI.ParameterValue(), q, 6.0)
    optimize!(model)
    @test value(x) ≈ 0 atol = ATOL
    @test value(y) ≈ 0.3 atol = ATOL
    return
end

function test_jump_quadratic_interval_cached()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    # optimizer = POI.Optimizer(HiGHS.Optimizer)
    # model = direct_model(optimizer)
    set_silent(model)
    # model = Model(() -> optimizer)
    # MOI.set(model, MOI.Silent(), true)
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, p in Parameter(10.0))
    @variable(model, q in Parameter(4.0))
    @constraint(model, 0 <= x - p * y + q <= 0)
    @objective(model, Min, x + y)
    optimize!(model)
    @test value(x) ≈ 0 atol = ATOL
    @test value(y) ≈ 0.4 atol = ATOL
    MOI.set(model, POI.ParameterValue(), p, 20.0)
    optimize!(model)
    @test value(x) ≈ 0 atol = ATOL
    @test value(y) ≈ 0.2 atol = ATOL
    MOI.set(model, POI.ParameterValue(), q, 6.0)
    optimize!(model)
    @test value(x) ≈ 0 atol = ATOL
    @test value(y) ≈ 0.3 atol = ATOL
    return
end

function test_affine_parametric_objective()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, p in Parameter(1.0))
    @variable(model, 0 <= x <= 1)
    @objective(model, Max, (p + 0.5) * x)
    optimize!(model)
    @test value(x) ≈ 1.0
    @test objective_value(model) ≈ 1.5
    @test value(objective_function(model)) ≈ 1.5
end

function test_abstract_optimizer_attributes()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    set_attribute(model, "time_limit", 60 * 1000.0)
    attr = MOI.RawOptimizerAttribute("time_limit")
    @test MOI.supports(unsafe_backend(model), attr)
    @test get_attribute(model, "time_limit") ≈ 60 * 1000.0
    return
end

function test_get_quadratic_constraint()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(2.0))
    @constraint(model, c, p * x <= 10)
    optimize!(model)
    @test value(c) ≈ 2.0 * value(x)
    return
end

function test_get_duals_from_multiplicative_parameters_1()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p1 in Parameter(2.0))
    @variable(model, p2 in Parameter(2.0))
    @constraint(model, c, 3 * x >= p1 * p2)
    @objective(model, Min, sum(x))
    optimize!(model)
    @test dual(c) ≈ 1.0 / 3.0
    @test MOI.get.(model, POI.ParameterDual(), p1) ≈ 2.0 / 3
    @test MOI.get.(model, POI.ParameterDual(), p2) ≈ 2.0 / 3
    return
end

function test_get_duals_from_multiplicative_parameters_2()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p1 in Parameter(40.0))
    @variable(model, p2 in Parameter(2.0))
    @constraint(model, c, 3 * x >= p1 * p2)
    @objective(model, Min, sum(x))
    optimize!(model)
    @test dual(c) ≈ 1.0 / 3.0
    @test MOI.get.(model, POI.ParameterDual(), p1) ≈ 2.0 / 3
    @test MOI.get.(model, POI.ParameterDual(), p2) ≈ 40.0 / 3
    return
end

function test_get_duals_from_multiplicative_parameters_3()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(4.0))
    @constraint(model, c, 3 * x >= p * p)
    @objective(model, Min, sum(x))
    optimize!(model)
    @test dual(c) ≈ 1.0 / 3.0
    @test MOI.get.(model, POI.ParameterDual(), p) ≈ 2 * 4.0 / 3
    return
end

function test_parameters_cannot_be_nan_1()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(NaN))
    @constraint(model, c, 3 * x >= p * p)
    @objective(model, Min, sum(x))
    @test_throws AssertionError optimize!(model)
    MOI.set(model, POI.ParameterValue(), p, 20.0)
    return
end

function test_parameters_cannot_be_nan_2()
    optimizer = POI.Optimizer(HiGHS.Optimizer)
    model = direct_model(optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    @test_throws AssertionError @variable(model, p in Parameter(NaN))
    @variable(model, p in Parameter(1.0))
    @constraint(model, c, 3 * x[1] + x[2] >= p * p)
    @objective(model, Min, sum(x))
    @test_throws AssertionError MOI.set(model, POI.ParameterValue(), p, NaN)
    return
end

function test_parameter_Cannot_be_inf_1()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(Inf))
    @constraint(model, c, 3 * x >= p * p)
    @objective(model, Min, sum(x))
    @test_throws AssertionError optimize!(model)
    MOI.set(model, POI.ParameterValue(), p, 20.0)
    return
end

function test_parameter_Cannot_be_inf_2()
    optimizer = POI.Optimizer(HiGHS.Optimizer)
    model = direct_model(optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    @test_throws AssertionError @variable(model, p in Parameter(Inf))
    @variable(model, p in Parameter(1.0))
    @constraint(model, c, 3 * x[1] + x[2] >= p * p)
    @objective(model, Min, sum(x))
    @test_throws AssertionError MOI.set(model, POI.ParameterValue(), p, Inf)
    return
end

function test_jump_psd_cone_with_parameter_pv()
    inner = POI.Optimizer(SCS.Optimizer; with_bridge_type = Float64)
    model = direct_model(inner)
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    # @constraint(
    #     model,
    #     con,
    #     [[0 (p * x - 1)]; [(p * x - 1) 0]] in PSDCone()
    # );
    # TODO: bridges do not support setting constraint function
    @constraint(
        model,
        con,
        [0, (p * x - 1), 0] in MOI.PositiveSemidefiniteConeTriangle(2)
    )
    @test MOI.get(backend(model), MOI.ConstraintFunction(), index(con)) isa
          MOI.VectorQuadraticFunction{Float64}
    @test MOI.get(backend(model), MOI.ConstraintSet(), index(con)) isa
          MOI.PositiveSemidefiniteConeTriangle
    # the above constraint is equivalent to
    # - (p * x -1)^2 >=0 -> (p * x -1)^2 <= 0 -> (p * x -1) == 0 -> p*x == 1
    @test is_valid(model, con)
    optimize!(model)
    @test value(x) ≈ 1.0 atol = 1e-5
    set_parameter_value(p, 3.0)
    optimize!(model)
    @test value(x) ≈ 1 / 3 atol = 1e-5
    delete(model, con)
    return
end

function test_jump_psd_cone_with_parameter_pp()
    model = Model(() -> POI.Optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    # @constraint(
    #     model,
    #     con,
    #     [[0 (x - p * p)]; [(x - p * p) 0]] in PSDCone()
    # )
    @constraint(
        model,
        con,
        [0, (x - p * p), 0] in MOI.PositiveSemidefiniteConeTriangle(2)
    )
    @test is_valid(model, con)
    optimize!(model)
    @test value(x) ≈ 1.0 atol = 1e-5
    set_parameter_value(p, 3.0)
    optimize!(model)
    @test value(x) ≈ 9.0 atol = 1e-5
    delete(model, con)
    return
end

function test_jump_psd_cone_with_parameter_p()
    model = Model(() -> POI.Optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    # @constraint(model, con, [[0 (x - p)]; [(x - p) 0]] in PSDCone())
    @constraint(
        model,
        con,
        [0, (x - p), 0] in MOI.PositiveSemidefiniteConeTriangle(2)
    )
    @test is_valid(model, con)
    optimize!(model)
    @test value(x) ≈ 1.0 atol = 1e-5
    set_parameter_value(p, 3.0)
    optimize!(model)
    @test value(x) ≈ 3.0 atol = 1e-5
    delete(model, con)
    return
end

function test_jump_psd_cone_with_parameter_pv_v_pv()
    model = Model(() -> POI.Optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    @constraint(
        model,
        con,
        [p * x, (2 * x - 3), p * 3 * x] in
        MOI.PositiveSemidefiniteConeTriangle(2)
    )
    # which is (p * x) * (p * 3 *x) - (2 * x - 3) ^ 2 >= 0
    # that simplifies to: p^2 * 3 * x^2 - 4 * x^2 + 12 * x - 9 >= 0
    # then: (p^2 * 3 - 4) * x^2 + 12 * x - 9 >= 0
    # for p == 1: -1 * x^2 + 12 * x - 9 >= 0
    # for p == 3: 23 * x^2 + 12 * x - 9 >= 0
    @objective(model, Min, x)
    @test is_valid(model, con)
    optimize!(model)
    @test value(x) ≈ 0.803845 atol = 1e-5
    @test value(x) ≈ 6 - 3 * sqrt(3) atol = 1e-5
    set_parameter_value(p, 3.0)
    optimize!(model)
    @test value(x) ≈ 0.416888 atol = 1e-5
    @test value(x) ≈ (9 * sqrt(3) - 6) / 23 atol = 1e-5
    delete(model, con)
    return
end

function test_jump_psd_cone_with_parameter_pp_v_pv()
    model = Model(() -> POI.Optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    @constraint(
        model,
        con,
        [p * p, (2 * x - 3), p * 3 * x] in
        MOI.PositiveSemidefiniteConeTriangle(2)
    )
    @objective(model, Min, x)
    @test is_valid(model, con)
    optimize!(model)
    @test value(x) ≈ 0.7499854 atol = 1e-5
    set_parameter_value(p, 3.0)
    optimize!(model)
    @test value(x) ≈ 0.0971795 atol = 1e-5
    delete(model, con)
    return
end

function test_jump_psd_cone_with_parameter_p_v_pv()
    model = Model(() -> POI.Optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    @constraint(
        model,
        con,
        [p, (2 * x - 3), p * 3 * x] in MOI.PositiveSemidefiniteConeTriangle(2)
    )
    @objective(model, Min, x * x - p * p)
    @test is_valid(model, con)
    optimize!(model)
    @test value(x) ≈ 0.7499854 atol = 1e-3
    set_parameter_value(p, 3.0)
    optimize!(model)
    @test value(x) ≈ 0.236506 atol = 1e-3
    delete(model, con)
    return
end

function test_jump_psd_cone_with_parameter_p_v_pp()
    model = Model(() -> POI.Optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    @constraint(
        model,
        con,
        [p, (2 * x - 3), p * 3 * p] in MOI.PositiveSemidefiniteConeTriangle(2)
    )
    @objective(model, Min, x - p)
    @test is_valid(model, con)
    optimize!(model)
    @test value(x) ≈ 0.633969 atol = 1e-5
    set_parameter_value(p, Float32(3.0))
    optimize!(model)
    @test value(x) ≈ -2.9999734 atol = 1e-5
    delete(model, con)
    return
end

function test_jump_psd_cone_without_parameter_v_and_vv()
    model = Model(() -> POI.Optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    @constraint(
        model,
        con,
        [x, (x - 1), x] in MOI.PositiveSemidefiniteConeTriangle(2)
    )
    @test is_valid(model, con)
    optimize!(model)
    @test value(x) ≈ 0.50000 atol = 1e-5
    return
end

function test_variable_and_constraint_not_registered()
    model1 = direct_model(POI.Optimizer(SCS.Optimizer))
    model2 = direct_model(POI.Optimizer(SCS.Optimizer))
    set_silent(model1)
    set_silent(model2)
    @variable(model1, x)
    @variable(model1, p in Parameter(1.0))
    @variable(model1, p1 in Parameter(1.0))
    @variable(model2, p2 in Parameter(1.0))
    @constraint(model1, con, [x - p] in MOI.Nonnegatives(1))
    @test !MOI.is_valid(backend(model2), index(x))
    @test_throws MOI.InvalidIndex MOI.get(
        backend(model2),
        MOI.ConstraintFunction(),
        index(ParameterRef(p1)),
    )
    @test_throws MOI.InvalidIndex MOI.get(
        backend(model2),
        MOI.ConstraintSet(),
        index(ParameterRef(p1)),
    )
    @test_throws MOI.InvalidIndex MOI.set(
        backend(model2),
        MOI.ConstraintPrimalStart(),
        index(ParameterRef(p1)),
        1.0,
    )
    @test_throws MOI.InvalidIndex MOI.get(
        backend(model2),
        MOI.ConstraintPrimalStart(),
        index(ParameterRef(p1)),
    )
    @test_throws MOI.InvalidIndex MOI.set(
        backend(model2),
        MOI.ObjectiveFunction{MOI.VariableIndex}(),
        index(x),
    )
    @test_throws ErrorException(
        "Cannot use a parameter as objective function alone in ParametricOptInterface.",
    ) MOI.set(
        backend(model2),
        MOI.ObjectiveFunction{MOI.VariableIndex}(),
        index(p),
    )
    MOI.set(backend(model2), MOI.VariablePrimalStart(), index(p2), 1.0)
    @test_throws ErrorException MOI.set(
        backend(model2),
        MOI.VariablePrimalStart(),
        index(p2),
        10.0,
    )
    @test_throws MOI.InvalidIndex MOI.set(
        backend(model2),
        MOI.VariablePrimalStart(),
        MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 100),
        10.0,
    )
    @test_throws MOI.InvalidIndex MOI.get(
        backend(model2),
        MOI.VariablePrimalStart(),
        MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 100),
    )
    @test_throws ErrorException MOI.delete(backend(model2), index(p2))
    @test_throws MOI.InvalidIndex MOI.delete(
        backend(model2),
        MOI.ConstraintIndex{
            MOI.VectorQuadraticFunction{Float64},
            MOI.Nonpositives,
        }(
            1,
        ),
    )
    @test_throws MOI.InvalidIndex MOI.delete(
        backend(model2),
        MOI.ConstraintIndex{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.EqualTo{Float64},
        }(
            1,
        ),
    )
    @test_throws MOI.InvalidIndex MOI.get(
        backend(model2),
        MOI.VariablePrimal(),
        MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 100),
    )
    return
end

function test_jump_errors()
    model = direct_model(POI.Optimizer(SCS.Optimizer))
    @test_throws MOI.UnsupportedAttribute MOI.get(
        backend(model),
        MOI.NLPBlock(),
    )

    MOI.get(
        backend(model),
        MOI.ListOfConstraintAttributesSet{
            MOI.VectorQuadraticFunction{Float64},
            MOI.Nonnegatives,
        }(),
    )

    MOI.get(
        backend(model),
        MOI.ListOfConstraintAttributesSet{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.LessThan{Float64},
        }(),
    )

    MOI.set(
        backend(model),
        POI._WarnIfQuadraticOfAffineFunctionAmbiguous(),
        false,
    )

    @test MOI.get(
        backend(model),
        POI._WarnIfQuadraticOfAffineFunctionAmbiguous(),
    ) == false

    MOI.get(
        backend(model),
        MOI.ListOfConstraintAttributesSet{
            MOI.VectorQuadraticFunction{Float64},
            MOI.Nonnegatives,
        }(),
    )

    MOI.get(
        backend(model),
        MOI.ListOfConstraintAttributesSet{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.LessThan{Float64},
        }(),
    )

    model = direct_model(POI.Optimizer(Ipopt.Optimizer))

    @test_throws MOI.GetAttributeNotAllowed MOI.get(
        backend(model),
        MOI.ListOfConstraintAttributesSet{
            MOI.VectorQuadraticFunction{Float64},
            MOI.Nonnegatives,
        }(),
    )

    MOI.get(
        backend(model),
        MOI.ListOfConstraintAttributesSet{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.LessThan{Float64},
        }(),
    )

    MOI.set(
        backend(model),
        POI._WarnIfQuadraticOfAffineFunctionAmbiguous(),
        false,
    )

    @test MOI.get(
        backend(model),
        POI._WarnIfQuadraticOfAffineFunctionAmbiguous(),
    ) == false

    @test_throws MOI.GetAttributeNotAllowed MOI.get(
        backend(model),
        MOI.ListOfConstraintAttributesSet{
            MOI.VectorQuadraticFunction{Float64},
            MOI.Nonnegatives,
        }(),
    )

    MOI.get(
        backend(model),
        MOI.ListOfConstraintAttributesSet{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.LessThan{Float64},
        }(),
    )

    return
end

function test_print()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, p in Parameter(1.0))
    @variable(model, x)
    @constraint(model, con, x >= p + p * p + p * x)
    @objective(model, Min, 1 + 2x + 0.1p + 1.0p^2)
    filename = tempdir() * "/test.lp"
    write_to_file(model, filename)
    @test readlines(filename) == [
        "minimize",
        "obj: 1 + 2 x + 0.1 p + [ 2 p ^ 2 ]/2",
        "subject to",
        "con: 1 x - 1 p + [ -1 p * x - 1 p ^ 2 ] >= 0",
        "Bounds",
        "x free",
        "p = 1",
        "End",
    ]
    return
end

function test_set_normalized_coefficient()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, p in Parameter(1.0))
    @variable(model, x)
    @constraint(model, con, x >= p)
    @constraint(model, con1, x >= 1)
    @constraint(model, con2, x >= x * p)
    @test_throws ErrorException set_normalized_coefficient(con, x, 2.0)
    set_normalized_coefficient(con1, x, 2.0)
    @test_throws ErrorException set_normalized_coefficient(con2, x, 2.0)
    return
end

function test_ListOfConstraintAttributesSet()
    cached = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.Utilities.AUTOMATIC,
    )
    optimizer = POI.Optimizer(cached)
    model = direct_model(optimizer)
    set_silent(model)
    @variable(model, p in Parameter(1.0))
    @variable(model, x)
    @constraint(model, con, [x * p] in MOI.Nonnegatives(1))
    ret = get_attribute(
        model,
        MOI.ListOfConstraintAttributesSet{
            MOI.VectorQuadraticFunction{Float64},
            MOI.Nonnegatives,
        }(),
    )
    @test ret == []
    set_attribute(model, POI._WarnIfQuadraticOfAffineFunctionAmbiguous(), false)
    ret = get_attribute(
        model,
        MOI.ListOfConstraintAttributesSet{
            MOI.VectorQuadraticFunction{Float64},
            MOI.Nonnegatives,
        }(),
    )
    @test ret == []
    return
end

end # module

TestJuMPTests.runtests()
