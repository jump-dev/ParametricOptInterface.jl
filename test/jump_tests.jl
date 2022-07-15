@testset "JuMP direct model - Linear Constraints - Affine parameters" begin
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
end

@testset "JuMP direct model - Linear Constraints - Parameter x variable" begin
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
end

@testset "JuMP - Linear Constraints - Affine parameters" begin
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
end

@testset "JuMP - Linear Constraints - Parameter x variable" begin
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
end

@testset "JuMP ConstraintFunction and ObjectiveFunction getters" begin
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
end

@testset "JuMP Interpret parametric bounds simple example" begin
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
end

@testset "JuMP Interpret parametric bounds parametric expression" begin
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
end

@testset "JuMP Interpret parametric bounds direct model" begin
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
end

@testset "JuMP Interpret parametric bounds direct model no interpretation" begin
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
end

@testset "JuMP Interpret parametric bounds direct model change interpretation" begin
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
end

@testset "JuMP Interpret parametric bounds direct model both interpretations" begin
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
end

@testset "JuMP Interpret parametric bounds parametric not a valid bound" begin
    model = direct_model(POI.Optimizer(GLPK.Optimizer()))
    MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)

    @variable(model, x[i = 1:2])
    @variable(model, p[i = 1:2] in POI.Parameter.(-1))
    @test_throws ErrorException @constraint(
        model,
        [i in 1:2],
        2x[i] >= p[i] + p[1]
    )
end

@testset "JuMP Set Variable Start Value" begin
    cached = MOI.Bridges.full_bridge_optimizer(
        MOIU.CachingOptimizer(
            MOIU.UniversalFallback(MOIU.Model{Float64}()),
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

    @test_throws ErrorException(
        "MathOptInterface.VariablePrimalStart() is not supported for parameters",
    ) set_start_value(p, 1.0)
    @test_throws ErrorException(
        "MathOptInterface.VariablePrimalStart() is not supported for parameters",
    ) start_value(p)
end
