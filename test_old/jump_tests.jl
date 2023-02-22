# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function test_jump_direct_affine_parameters()
    optimizer = POI.Optimizer(GLPK.Optimizer())
    model = direct_model(optimizer)
    @variable(model, x[i = 1:2] >= 0)
    @variable(model, y in POI.Parameter(0))
    @variable(model, w in POI.Parameter(0))
    @variable(model, z in POI.Parameter(0))
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
    optimizer = POI.Optimizer(GLPK.Optimizer())
    model = direct_model(optimizer)
    @variable(model, x[i = 1:2] >= 0)
    @variable(model, y in POI.Parameter(0))
    @variable(model, w in POI.Parameter(0))
    @variable(model, z in POI.Parameter(0))
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
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, x[i = 1:2] >= 0)
    @variable(model, y in POI.Parameter(0))
    @variable(model, w in POI.Parameter(0))
    @variable(model, z in POI.Parameter(0))
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
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, x[i = 1:2] >= 0)
    @variable(model, y in POI.Parameter(0))
    @variable(model, w in POI.Parameter(0))
    @variable(model, z in POI.Parameter(0))
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
    vx = @variable(model, x[i = 1:2])
    vp = @variable(model, p[i = 1:2] in POI.Parameter.(-1))
    c1 = @constraint(model, con, sum(x) + sum(p) >= 1)
    c2 = @constraint(model, conq, sum(x .* p) >= 1)
    c3 = @constraint(model, conqa, sum(x .* p) + x[1]^2 + x[1] + p[1] >= 1)
    @test MOI.get(model, MOI.ConstraintFunction(), c1) ≈
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
    )
    @test MOI.get(model, MOI.ConstraintFunction(), c2) ≈
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
    )
    @test MOI.get(model, MOI.ConstraintFunction(), c3) ≈
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
    )
    o1 = @objective(model, Min, sum(x) + sum(p))
    F = MOI.get(model, MOI.ObjectiveFunctionType())
    @test MOI.get(model, MOI.ObjectiveFunction{F}()) ≈
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
    )
    o2 = @objective(model, Min, sum(x .* p) + 2)
    F = MOI.get(model, MOI.ObjectiveFunctionType())
    @test MOI.get(model, MOI.ObjectiveFunction{F}()) ≈
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
        2.0,
    )
    o3 = @objective(model, Min, sum(x .* p) + x[1]^2 + x[1] + p[1])
    F = MOI.get(model, MOI.ObjectiveFunctionType())
    @test MOI.get(model, MOI.ObjectiveFunction{F}()) ≈
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
    )
    return
end

function test_jump_interpret_parameteric_bounds()
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)
    @variable(model, x[i = 1:2])
    @variable(model, p[i = 1:2] in POI.Parameter.(-1))
    @constraint(model, [i in 1:2], x[i] >= p[i])
    @objective(model, Min, sum(x))
    optimize!(model)
    @test MOI.get(model, MOI.ListOfConstraintTypesPresent()) ==
          Tuple{Type,Type}[
        (MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}),
        (MOI.VariableIndex, POI.Parameter),
    ]
    @test MOI.get(
        backend(model).optimizer.model.optimizer,
        MOI.ListOfConstraintTypesPresent(),
    ) == Tuple{Type,Type}[(MOI.VariableIndex, MOI.GreaterThan{Float64})]
    @test objective_value(model) == -2
    MOI.set(model, POI.ParameterValue(), p[1], 4.0)
    optimize!(model)
    @test objective_value(model) == 3
    return
end

function test_jump_interpret_parameteric_bounds_expression()
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)
    @variable(model, x[i = 1:2])
    @variable(model, p[i = 1:2] in POI.Parameter.(-1))
    @constraint(model, [i in 1:2], x[i] >= p[i] + p[1])
    @objective(model, Min, sum(x))
    optimize!(model)
    @test MOI.get(model, MOI.ListOfConstraintTypesPresent()) ==
          Tuple{Type,Type}[
        (MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}),
        (MOI.VariableIndex, POI.Parameter),
    ]
    @test MOI.get(
        backend(model).optimizer.model.optimizer,
        MOI.ListOfConstraintTypesPresent(),
    ) == Tuple{Type,Type}[(MOI.VariableIndex, MOI.GreaterThan{Float64})]
    @test objective_value(model) == -4
    MOI.set(model, POI.ParameterValue(), p[1], 4.0)
    optimize!(model)
    @test objective_value(model) == 11.0
    return
end

function test_jump_direct_interpret_parameteric_bounds()
    model = direct_model(POI.Optimizer(GLPK.Optimizer()))
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)
    @variable(model, x[i = 1:2])
    @variable(model, p[i = 1:2] in POI.Parameter.(-1))
    @constraint(model, [i in 1:2], x[i] >= p[i])
    @objective(model, Min, sum(x))
    optimize!(model)
    @test MOI.get(model, MOI.ListOfConstraintTypesPresent()) ==
          Tuple{Type,Type}[(MOI.VariableIndex, MOI.GreaterThan{Float64})]
    @test MOI.get(
        backend(model).optimizer,
        MOI.ListOfConstraintTypesPresent(),
    ) == Tuple{Type,Type}[(MOI.VariableIndex, MOI.GreaterThan{Float64})]
    @test objective_value(model) == -2
    MOI.set(model, POI.ParameterValue(), p[1], 4.0)
    optimize!(model)
    @test objective_value(model) == 3
    return
end

function test_jump_direct_interpret_parameteric_bounds_no_interpretation()
    model = direct_model(POI.Optimizer(GLPK.Optimizer()))
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_CONSTRAINTS)
    @variable(model, x[i = 1:2])
    @variable(model, p[i = 1:2] in POI.Parameter.(-1))
    @constraint(model, [i in 1:2], x[i] >= p[i])
    @objective(model, Min, sum(x))
    optimize!(model)
    @test MOI.get(model, MOI.ListOfConstraintTypesPresent()) ==
          Tuple{Type,Type}[(
        MOI.ScalarAffineFunction{Float64},
        MOI.GreaterThan{Float64},
    )]
    @test MOI.get(
        backend(model).optimizer,
        MOI.ListOfConstraintTypesPresent(),
    ) == Tuple{Type,Type}[(
        MOI.ScalarAffineFunction{Float64},
        MOI.GreaterThan{Float64},
    )]
    @test objective_value(model) == -2
    MOI.set(model, POI.ParameterValue(), p[1], 4.0)
    optimize!(model)
    @test objective_value(model) == 3
    return
end

function test_jump_direct_interpret_parameteric_bounds_change()
    model = direct_model(POI.Optimizer(GLPK.Optimizer()))
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)
    @variable(model, x[i = 1:2])
    @variable(model, p[i = 1:2] in POI.Parameter.(-1))
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
    model = direct_model(POI.Optimizer(GLPK.Optimizer()))
    MOI.set(model, POI.ConstraintsInterpretation(), POI.BOUNDS_AND_CONSTRAINTS)
    @variable(model, x[i = 1:2])
    @variable(model, p[i = 1:2] in POI.Parameter.(-1))
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
    model = direct_model(POI.Optimizer(GLPK.Optimizer()))
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)
    @variable(model, x[i = 1:2])
    @variable(model, p[i = 1:2] in POI.Parameter.(-1))
    @test_throws ErrorException @constraint(
        model,
        [i in 1:2],
        2x[i] >= p[i] + p[1]
    )
    return
end

function test_jump_set_variable_start_value()
    cached = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            GLPK.Optimizer(),
        ),
        Float64,
    )
    optimizer = POI.Optimizer(cached)
    model = direct_model(optimizer)
    @variable(model, x >= 0)
    @variable(model, p in POI.Parameter(0))
    set_start_value(x, 1.0)
    @test start_value(x) == 1
    err = ErrorException(
        "MathOptInterface.VariablePrimalStart() is not supported for parameters",
    )
    @test_throws err set_start_value(p, 1.0)
    @test_throws err start_value(p)
    return
end

function test_jump_direct_get_parameter_value()
    model = direct_model(POI.Optimizer(GLPK.Optimizer()))
    @variable(model, x, lower_bound = 0.0, upper_bound = 10.0)
    @variable(model, y, binary = true)
    @variable(model, z, set = POI.Parameter(10))
    c = @constraint(model, 19.0 * x - z + 22.0 * y <= 1.0)
    @objective(model, Min, x + y)
    @test MOI.get(model, POI.ParameterValue(), z) == 10
    return
end

function test_jump_get_parameter_value()
    model = Model(() -> ParametricOptInterface.Optimizer(GLPK.Optimizer()))
    @variable(model, x, lower_bound = 0.0, upper_bound = 10.0)
    @variable(model, y, binary = true)
    @variable(model, z, set = POI.Parameter(10))
    c = @constraint(model, 19.0 * x - z + 22.0 * y <= 1.0)
    @objective(model, Min, x + y)
    @test MOI.get(model, POI.ParameterValue(), z) == 10
    return
end

function test_jump_sdp_scalar_parameter()
    cached = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            SCS.Optimizer(),
        ),
        Float64,
    )
    optimizer = POI.Optimizer(cached)
    m = direct_model(optimizer)
    set_silent(m)
    @variable(m, p in POI.Parameter(0))
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
    cached = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            SCS.Optimizer(),
        ),
        Float64,
    )
    optimizer = POI.Optimizer(cached)
    m = direct_model(optimizer)
    set_silent(m)
    P1 = [1 2; 2 3]
    @variable(m, p[1:2, 1:2] in POI.Parameter.(P1))
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
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, x[1:2] in POI.Parameter.(ones(2) .* 4.0))
    @variable(model, y[1:6])
    @constraint(model, ctr1, 3 * y[1] >= 2 - 7 * x[1])
    @objective(model, Min, 5 * y[1])
    JuMP.optimize!(model)
    @test 5 / 3 ≈ JuMP.dual(ctr1) atol = 1e-3
    @test [-35 / 3, 0.0] ≈ MOI.get.(model, POI.ParameterDual(), x) atol = 1e-3
    @test [-26 / 3, 0.0, 0.0, 0.0, 0.0, 0.0] ≈ JuMP.value.(y) atol = 1e-3
    @test -130 / 3 ≈ JuMP.objective_value(model) atol = 1e-3
    return
end

function test_jump_dual_multiple_parameters_1()
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, x[1:6] in POI.Parameter.(ones(6) .* 4.0))
    @variable(model, y[1:6])
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
    JuMP.optimize!(model)
    @test 5 / 3 ≈
          JuMP.dual(ctr1) +
          JuMP.dual(ctr2) +
          JuMP.dual(ctr3) +
          JuMP.dual(ctr4) +
          JuMP.dual(ctr5) +
          JuMP.dual(ctr6) atol = 1e-3
    @test 0.0 ≈ JuMP.dual(ctr7) atol = 1e-3
    @test 0.0 ≈ JuMP.dual(ctr8) atol = 1e-3
    @test [0.0, 0.0, -35 / 3, 0.0, 0.0, 0.0] ≈
          MOI.get.(model, POI.ParameterDual(), x) atol = 1e-3
    @test [-26 / 3, 0.0, 0.0, 0.0, 0.0, 0.0] ≈ JuMP.value.(y) atol = 1e-3
    @test -130 / 3 ≈ JuMP.objective_value(model) atol = 1e-3
    return
end

function test_jump_duals_LessThan()
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α in POI.Parameter(-1.0))
    @variable(model, x)
    cref = @constraint(model, x ≤ α)
    @objective(model, Max, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -1.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0

    MOI.set(model, POI.ParameterValue(), α, 2.0)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 2.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
    return
end

function test_jump_duals_EqualTo()
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α in POI.Parameter(-1.0))
    @variable(model, x)
    cref = @constraint(model, x == α)
    @objective(model, Max, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -1.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
    MOI.set(model, POI.ParameterValue(), α, 2.0)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 2.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
    return
end

function test_jump_duals_GreaterThan()
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α in POI.Parameter(1.0))
    MOI.set(model, POI.ParameterValue(), α, -1.0)
    @variable(model, x)
    cref = @constraint(model, x >= α)
    @objective(model, Min, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -1.0
    @test JuMP.dual(cref) == 1.0
    @test MOI.get(model, POI.ParameterDual(), α) == 1.0
    MOI.set(model, POI.ParameterValue(), α, 2.0)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 2.0
    @test JuMP.dual(cref) == 1.0
    @test MOI.get(model, POI.ParameterDual(), α) == 1.0
    return
end

function test_jump_dual_multiple_parameters_2()
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α[1:10] in POI.Parameter.(ones(10)))
    @variable(model, x)
    cref = @constraint(model, x == sum(2 * α[i] for i in 1:10))
    @objective(model, Min, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 20.0
    @test JuMP.dual(cref) == 1.0
    @test MOI.get(model, POI.ParameterDual(), α[3]) == 2.0
    return
end

function test_jump_dual_mixing_params_and_vars_1()
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α[1:5] in POI.Parameter.(ones(5)))
    @variable(model, x)
    cref = @constraint(model, sum(x for i in 1:5) == sum(2 * α[i] for i in 1:5))
    @objective(model, Min, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 2.0
    @test JuMP.dual(cref) == 1 / 5
    @test MOI.get(model, POI.ParameterDual(), α[3]) == 2 / 5
    return
end

function test_jump_dual_mixing_params_and_vars_2()
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α[1:5] in POI.Parameter.(ones(5)))
    @variable(model, x)
    cref = @constraint(model, 0.0 == sum(-x + 2 * α[i] for i in 1:5))
    @objective(model, Min, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 2.0
    @test JuMP.dual(cref) == 1 / 5
    @test MOI.get(model, POI.ParameterDual(), α[3]) == 2 / 5
    return
end

function test_jump_dual_mixing_params_and_vars_3()
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α[1:5] in POI.Parameter.(ones(5)))
    @variable(model, x)
    cref = @constraint(model, 0.0 == sum(-x + 2.0 + 2 * α[i] for i in 1:5))
    @objective(model, Min, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 4.0
    @test JuMP.dual(cref) == 1 / 5
    @test MOI.get(model, POI.ParameterDual(), α[3]) == 2 / 5
    return
end

function test_jump_dual_add_after_solve()
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α in POI.Parameter(1.0))
    MOI.set(model, POI.ParameterValue(), α, -1.0)
    @variable(model, x)
    cref = @constraint(model, x <= α)
    @objective(model, Max, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -1.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
    @variable(model, b in POI.Parameter(-2.0))
    cref = @constraint(model, x <= b)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -2.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == 0.0
    @test MOI.get(model, POI.ParameterDual(), b) == -1.0
    return
end

function test_jump_dual_add_ctr_alaternative()
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α in POI.Parameter(-1.0))
    @variable(model, x)
    exp = x - α
    cref = @constraint(model, exp ≤ 0)
    @objective(model, Max, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -1.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
    return
end

function test_jump_dual_delete_constraint()
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α in POI.Parameter(-1.0))
    @variable(model, x)
    cref1 = @constraint(model, x ≤ α / 2)
    cref2 = @constraint(model, x ≤ α)
    cref3 = @constraint(model, x ≤ 2α)
    @objective(model, Max, x)
    JuMP.delete(model, cref3)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -1.0
    @test JuMP.dual(cref1) == 0.0
    @test JuMP.dual(cref2) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
    JuMP.delete(model, cref2)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -0.5
    @test JuMP.dual(cref1) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -0.5
    return
end

function test_jump_nlp()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    cached =
        () -> MOI.Bridges.full_bridge_optimizer(
            MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
                ipopt,
            ),
            Float64,
        )
    POI_cached_optimizer() = ParametricOptInterface.Optimizer(cached())
    model = Model(() -> POI_cached_optimizer())
    @variable(model, x)
    @variable(model, y)
    @variable(model, z in ParametricOptInterface.Parameter(10))
    @constraint(model, x + y >= z)
    @NLobjective(model, Min, x^2 + y^2)
    optimize!(model)
    objective_value(model)
    @test value(x) ≈ 5
    MOI.get(model, ParametricOptInterface.ParameterDual(), z)
    MOI.set(model, ParametricOptInterface.ParameterValue(), z, 2.0)
    optimize!(model)
    @test objective_value(model) ≈ 2 atol = 1e-3
    @test value(x) ≈ 1
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    model = Model(() -> ParametricOptInterface.Optimizer(ipopt))
    @variable(model, x)
    @variable(model, z in ParametricOptInterface.Parameter(10))
    MOI.set(
        model,
        ParametricOptInterface.ConstraintsInterpretation(),
        ParametricOptInterface.ONLY_BOUNDS,
    )
    @constraint(model, x >= z)
    @NLobjective(model, Min, x^2)
    optimize!(model)
    objective_value(model)
    @test value(x) ≈ 10
    MOI.get(model, ParametricOptInterface.ParameterDual(), z)
    MOI.set(model, ParametricOptInterface.ParameterValue(), z, 2.0)
    optimize!(model)
    @test objective_value(model) ≈ 4 atol = 1e-3
    @test value(x) ≈ 2
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
    cached = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        SCS.Optimizer(),
    )
    optimizer = POI.Optimizer(cached)
    model = direct_model(optimizer)
    @variable(model, x)
    @variable(model, y)
    @variable(model, t in POI.Parameter(5))
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
    cached = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            SCS.Optimizer(),
        ),
        Float64,
    )
    optimizer = POI.Optimizer(cached)
    model = direct_model(optimizer)
    @variable(model, x)
    @variable(model, y)
    @variable(model, t in POI.Parameter(5))
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
    cached = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        SCS.Optimizer(),
    )
    optimizer = POI.Optimizer(cached)
    model = direct_model(optimizer)
    @variable(model, x)
    @variable(model, y)
    @variable(model, t)
    @variable(model, p in POI.Parameter(0))
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

function test_jump_direct_qp_objective()
    optimizer = POI.Optimizer(Ipopt.Optimizer())
    model = direct_model(optimizer)
    MOI.set(model, MOI.Silent(), true)
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, p in POI.Parameter(1.0))
    @constraint(model, 2x + y <= 4)
    @constraint(model, x + 2y <= 4)
    @objective(model, Max, (x^2 + y^2) / 2)
    MOI.set(
        backend(model),
        POI.QuadraticObjectiveCoef(),
        (index(x), index(y)),
        2index(p) + 3,
    )
    optimize!(model)
    @test MOI.get(
        backend(model),
        POI.QuadraticObjectiveCoef(),
        (index(x), index(y)),
    ) ≈ MOI.ScalarAffineFunction{Int64}(
        MOI.ScalarAffineTerm{Int64}[MOI.ScalarAffineTerm{Int64}(
            2,
            MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1),
        )],
        3,
    )
    @test objective_value(model) ≈ 32 / 3 atol = ATOL
    @test value(x) ≈ 4 / 3 atol = ATOL
    @test value(y) ≈ 4 / 3 atol = ATOL
    MOI.set(model, POI.ParameterValue(), p, 2.0)
    optimize!(model)
    @test objective_value(model) ≈ 128 / 9 atol = ATOL
    @test value(x) ≈ 4 / 3 atol = ATOL
    @test value(y) ≈ 4 / 3 atol = ATOL
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
    cached = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            SCS.Optimizer(),
        ),
        Float64,
    )
    optimizer = POI.Optimizer(cached)
    model = direct_model(optimizer)
    MOI.set(model, MOI.Silent(), true)
    @variable(model, x)
    @variable(model, y)
    @variable(model, t)
    @variable(model, p in POI.Parameter(0))
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
