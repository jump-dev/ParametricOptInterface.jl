@testset "Production Problem" begin
    optimizer = POI.ParametricOptimizer(GLPK.Optimizer())
    
    c = [4.0, 3.0]
    A1 = [2.0, 1.0, 1.0]
    A2 = [1.0, 2.0, 1.0]
    b1 = 4.0
    b2 = 4.0

    x = MOI.add_variables(optimizer, length(c))

    @test typeof(x[1]) == MOI.VariableIndex

    w, cw = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
    z, cz = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    @test MOI.get(optimizer, MOI.VariablePrimal(), w) == 0

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(0.0))
    end

    cons1 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A1, [x[1], x[2], y]), 0.0)
    MOI.add_constraint(optimizer, cons1, MOI.LessThan(b1))

    cons2 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A2, [x[1], x[2], z]), 0.0)
    MOI.add_constraint(optimizer, cons2, MOI.LessThan(b2))

    @test cons1.terms[1].coefficient == 2
    @test POI.is_parameter_in_model(optimizer, cons2.terms[3].variable_index)

    obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([c[1], c[2], 3.0], [x[1], x[2], w]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    MOI.get(optimizer, MOI.TerminationStatus())

    MOI.get(optimizer, MOI.PrimalStatus())

    @test isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 28/3, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4/3, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 4/3, atol = ATOL)

    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cy), -5/3, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cz), -2/3, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cw), 3.0, atol = ATOL)

    MOI.get(optimizer, MOI.VariablePrimal(), w)
    MOI.get(optimizer, MOI.VariablePrimal(), y)
    MOI.get(optimizer, MOI.VariablePrimal(), z)

    MOI.set(optimizer, MOI.ConstraintSet(), cw, POI.Parameter(2.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(1.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(1.0))

    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.VariablePrimal(), w) == 2.0
    @test MOI.get(optimizer, MOI.VariablePrimal(), y) == 1.0
    @test MOI.get(optimizer, MOI.VariablePrimal(), z) == 1.0
    
    @test isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 13.0, atol = ATOL)
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [1.0, 1.0]

    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cy), -5/3, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cz), -2/3, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cw), 3.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cw, POI.Parameter(0.0))
    
    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.VariablePrimal(), w) == 0.0
    @test MOI.get(optimizer, MOI.VariablePrimal(), y) == 1.0
    @test MOI.get(optimizer, MOI.VariablePrimal(), z) == 1.0
    @test isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 7, atol = ATOL)
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [1.0, 1.0]


    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(-5.0))
    MOI.optimize!(optimizer)
    @test isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 12.0, atol = ATOL)
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [3.0, 0.0]

    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cy), 0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cz), -4, atol = ATOL)
   
end


@testset "Production Problem variation on parameters for duals" begin
    optimizer = POI.ParametricOptimizer(GLPK.Optimizer())
    
    c = [4.0, 3.0]
    A1 = [2.0, 1.0, 3.0]
    A2 = [1.0, 2.0, 0.5]
    b1 = 4.0
    b2 = 4.0

    x = MOI.add_variables(optimizer, length(c))

    @test typeof(x[1]) == MOI.VariableIndex

    w, cw = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
    z, cz = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    @test MOI.get(optimizer, MOI.VariablePrimal(), w) == 0

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(0.0))
    end

    cons1 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A1, [x[1], x[2], y]), 0.0)
    MOI.add_constraint(optimizer, cons1, MOI.LessThan(b1))

    cons2 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A2, [x[1], x[2], z]), 0.0)
    MOI.add_constraint(optimizer, cons2, MOI.LessThan(b2))

    @test cons1.terms[1].coefficient == 2
    @test cons2.terms[3].variable_index == MOI.VariableIndex(5)

    obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([c[1], c[2], 2.0], [x[1], x[2], w]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    MOI.get(optimizer, MOI.TerminationStatus())

    MOI.get(optimizer, MOI.PrimalStatus())

    @test isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 28/3, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4/3, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 4/3, atol = ATOL)

    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cy), -5, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cz), -2/6, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cw), 2.0, atol = ATOL)


    MOI.set(optimizer, MOI.ConstraintSet(), cw, POI.Parameter(2.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(1.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(1.0))

    MOI.optimize!(optimizer)

    @test isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 7.0, atol = ATOL)
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [0.0, 1.0]

    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cy), -9.0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cz), -0.0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cw), 2.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cw, POI.Parameter(0.0))
    
    MOI.optimize!(optimizer)

    @test isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 3.0, atol = ATOL)
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [0.0, 1.0]

    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cy), -9.0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cz), -0.0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cw), 2.0, atol = ATOL)


    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(-5.0))
    MOI.optimize!(optimizer)
    @test isapprox.(MOI.get(optimizer, MOI.ObjectiveValue()), 14.0, atol = ATOL)
    @test isapprox(MOI.get.(optimizer, MOI.VariablePrimal(), x),[3.5, 0.0], atol = ATOL)

    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cy), 0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cz), -2, atol = ATOL)
   
end