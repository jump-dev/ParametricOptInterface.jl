@testset "Production Problem" begin

    optimizer = POI.ParametricOptimizer(GLPK.Optimizer())

    c = [4.0, 3.0]
    A1 = [2.0, 1.0, 1.0]
    A2 = [1.0, 2.0, 1.0]
    b1 = 4.0
    b2 = 4.0

    x = MOI.add_variables(optimizer, length(c))

    @test typeof(x[1]) == MOI.VariableIndex

    w, cw = MOI.add_constrained_variable(optimizer, POI.Parameter(5))
    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(1))
    z, cz = MOI.add_constrained_variable(optimizer, POI.Parameter(1))

    @test MOI.get(optimizer, MOI.VariablePrimal(), w) == 5


    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(0.0))
    end

    cons1 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A1, [x[1], x[2], y]), 0.0)
    MOI.add_constraint(optimizer, cons1, MOI.LessThan(b1))

    cons2 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A2, [x[1], x[2], z]), 0.0)
    MOI.add_constraint(optimizer, cons2, MOI.LessThan(b2))

    @test cons1.terms[1].coefficient == 2
    @test cons2.terms[3].variable_index == MOI.VariableIndex(5)

    obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([c[1], c[2], 1.0], [x[1], x[2], w]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    MOI.get(optimizer, MOI.TerminationStatus())

    MOI.get(optimizer, MOI.PrimalStatus())

    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 12.0
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [1.0, 1.0] 

    MOI.get(optimizer, MOI.VariablePrimal(), w)
    MOI.get(optimizer, MOI.VariablePrimal(), y)
    MOI.get(optimizer, MOI.VariablePrimal(), z)

    MOI.set(optimizer, MOI.ConstraintSet(), cw, POI.Parameter(1.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(-1.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(3.0))

    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.VariablePrimal(), w) == 1.0
    @test MOI.get(optimizer, MOI.VariablePrimal(), y) == -1.0
    @test MOI.get(optimizer, MOI.VariablePrimal(), z) == 3.0
    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 5.0
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [1.0, 0.0] 



end