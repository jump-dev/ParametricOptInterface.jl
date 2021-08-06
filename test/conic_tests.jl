@testset "CONIC - Nonnegatives - Parameter in affine part" begin
    """
        min x + y
            x - t + 1 >= 0
            y - t + 2 >= 0

        opt
            x* = t-1
            y* = t-2
            obj = 2*t-3
    """
    cached = MOIU.CachingOptimizer(MOIU.UniversalFallback(MOIU.Model{Float64}()), ECOS.Optimizer())
    model = POI.ParametricOptimizer(cached)

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    t, ct = MOI.add_constrained_variable(model, POI.Parameter(5))

    A = [1. 0 -1; 0 1 -1]
    b = [1.; 2]
    terms = MOI.VectorAffineTerm.(1:2, MOI.ScalarAffineTerm.(A, reshape([x, y, t], 1, 3)))
    f = MOI.VectorAffineFunction(vec(terms), b)
    set = MOI.Nonnegatives(2)

    cnn = MOI.add_constraint(
        model,
        f,
        MOI.Nonnegatives(2)
    )

    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], [y, x]), 0.0,
        ),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT 
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 3  atol = ATOL
    @test MOI.get(model, MOI.ConstraintPrimal(), cnn) ≈ [0.0, 0.0] atol = ATOL 
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 7 atol = ATOL
    @test MOI.get(model, MOI.DualObjectiveValue()) ≈ 7 atol = ATOL 
    @test MOI.get(model, MOI.ConstraintDual(), cnn) ≈ [1.0, 1.0] atol = ATOL 

    MOI.set(model, POI.ParameterValue(), t, 6)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 5 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4  atol = ATOL
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 9 atol = ATOL
end