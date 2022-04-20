@testset "Vector Constraints - Nonnegatives - Parameter in affine part" begin
    """
        min x + y
            x - t + 1 >= 0
            y - t + 2 >= 0

        opt
            x* = t-1
            y* = t-2
            obj = 2*t-3
    """
    cached = MOIU.CachingOptimizer(
        MOIU.UniversalFallback(MOIU.Model{Float64}()),
        ECOS.Optimizer(),
    )
    model = POI.Optimizer(cached)

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    t, ct = MOI.add_constrained_variable(model, POI.Parameter(5))

    A = [1.0 0 -1; 0 1 -1]
    b = [1.0; 2]
    terms =
        MOI.VectorAffineTerm.(
            1:2,
            MOI.ScalarAffineTerm.(A, reshape([x, y, t], 1, 3)),
        )
    f = MOI.VectorAffineFunction(vec(terms), b)
    set = MOI.Nonnegatives(2)

    cnn = MOI.add_constraint(model, f, MOI.Nonnegatives(2))

    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], [y, x]),
            0.0,
        ),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 3 atol = ATOL
    @test MOI.get(model, MOI.ConstraintPrimal(), cnn) ≈ [0.0, 0.0] atol = ATOL
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 7 atol = ATOL
    @test MOI.get(model, MOI.DualObjectiveValue()) ≈ 7 atol = ATOL
    @test MOI.get(model, MOI.ConstraintDual(), cnn) ≈ [1.0, 1.0] atol = ATOL

    MOI.set(model, POI.ParameterValue(), t, 6)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 5 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 atol = ATOL
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 9 atol = ATOL
end

@testset "JuMP direct model - Vector Constraints - Nonnegatives - Parameter in affine part" begin
    """
        min x + y
            x - t + 1 >= 0
            y - t + 2 >= 0

        opt
            x* = t-1
            y* = t-2
            obj = 2*t-3
    """

    cached = MOIU.CachingOptimizer(
        MOIU.UniversalFallback(MOIU.Model{Float64}()),
        ECOS.Optimizer(),
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
end

@testset "Vector Constraints - Nonpositives - Parameter in affine part" begin
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
        MOIU.CachingOptimizer(
            MOIU.UniversalFallback(MOIU.Model{Float64}()),
            ECOS.Optimizer(),
        ),
        Float64,
    )
    model = POI.Optimizer(cached)

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    t, ct = MOI.add_constrained_variable(model, POI.Parameter(5))

    A = [-1.0 0 1; 0 -1 1]
    b = [-1.0; -2]
    terms =
        MOI.VectorAffineTerm.(
            1:2,
            MOI.ScalarAffineTerm.(A, reshape([x, y, t], 1, 3)),
        )
    f = MOI.VectorAffineFunction(vec(terms), b)
    set = MOI.Nonnegatives(2)

    cnn = MOI.add_constraint(model, f, MOI.Nonpositives(2))

    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], [y, x]),
            0.0,
        ),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 3 atol = ATOL
    @test MOI.get(model, MOI.ConstraintPrimal(), cnn) ≈ [0.0, 0.0] atol = ATOL
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 7 atol = ATOL
    @test MOI.get(model, MOI.DualObjectiveValue()) ≈ 7 atol = ATOL
    @test MOI.get(model, MOI.ConstraintDual(), cnn) ≈ [-1.0, -1.0] atol = ATOL

    MOI.set(model, POI.ParameterValue(), t, 6)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 5 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 atol = ATOL
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 9 atol = ATOL
end

@testset "JuMP direct model - Vector Constraints - Nonnegatives - Parameter in affine part" begin
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
        MOIU.CachingOptimizer(
            MOIU.UniversalFallback(MOIU.Model{Float64}()),
            ECOS.Optimizer(),
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
end

@testset "Vector Constraints - SOC - Parameter in affine part" begin
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
    cached = MOIU.CachingOptimizer(
        MOIU.UniversalFallback(MOIU.Model{Float64}()),
        ECOS.Optimizer(),
    )
    model = POI.Optimizer(cached)

    x, y, t = MOI.add_variables(model, 3)
    p, cp = MOI.add_constrained_variable(model, POI.Parameter(0))

    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    cnon = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, y))],
            [-1 / √2],
        ),
        MOI.Nonnegatives(1),
    )

    ceq = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(-1.0, t))],
            [1.0],
        ),
        MOI.Zeros(1),
    )

    A = [
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 -1
        0.0 0.0 1.0 0.0
    ]
    f = MOI.VectorAffineFunction(
        vec(
            MOI.VectorAffineTerm.(
                1:3,
                MOI.ScalarAffineTerm.(A, reshape([t, x, y, p], 1, 4)),
            ),
        ),
        zeros(3),
    )
    csoc = MOI.add_constraint(model, f, MOI.SecondOrderCone(3))

    f_error = MOI.VectorOfVariables([t, p, y])
    @test_throws ErrorException MOI.add_constraint(
        model,
        f_error,
        MOI.SecondOrderCone(3),
    )

    MOI.optimize!(model)

    @test MOI.get(model, MOI.ObjectiveValue()) ≈ -1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ -1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), t) ≈ 1 atol = ATOL

    MOI.set(model, POI.ParameterValue(), p, 1)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1 - 1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 1 - 1 / √2 atol = ATOL
end

@testset "Vector Constraints - SOC - No parameters" begin
    """
        Problem SOC2 from MOI

        min  x
        s.t. y ≥ 1/√2
            x² + y² ≤ 1

        in conic form:

        min  x
        s.t.  -1/√2 + y ∈ R₊
            1 - t ∈ {0}
            (t, x ,y) ∈ SOC₃

        opt
            x* = 1/√2
            y* = 1/√2
    """
    cached = MOI.Bridges.full_bridge_optimizer(
        MOIU.CachingOptimizer(
            MOIU.UniversalFallback(MOIU.Model{Float64}()),
            ECOS.Optimizer(),
        ),
        Float64,
    )
    model = POI.Optimizer(cached)

    x, y, t = MOI.add_variables(model, 3)

    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    cnon = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, y))],
            [-1 / √2],
        ),
        MOI.Nonnegatives(1),
    )

    ceq = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(-1.0, t))],
            [1.0],
        ),
        MOI.Zeros(1),
    )

    f = MOI.VectorOfVariables([t, x, y])
    csoc = MOI.add_constraint(model, f, MOI.SecondOrderCone(3))

    MOI.optimize!(model)

    @test MOI.get(model, MOI.ObjectiveValue()) ≈ -1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ -1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), t) ≈ 1 atol = ATOL
end

@testset "JuMP direct model - Vector Constraints - SOC - Parameter in affine part" begin
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

    cached = MOIU.CachingOptimizer(
        MOIU.UniversalFallback(MOIU.Model{Float64}()),
        ECOS.Optimizer(),
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
end
