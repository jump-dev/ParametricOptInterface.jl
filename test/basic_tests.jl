@testset "Basic tests" begin
    optimizer = POI.ParametricOptimizer(GLPK.Optimizer())

    MOI.set(optimizer, MOI.Silent(), true)

    x = MOI.add_variables(optimizer,2)
    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    @test_throws ErrorException("Cannot constrain a parameter") MOI.add_constraint(optimizer, MOI.SingleVariable(y), MOI.EqualTo(0.0))

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(0.0))
    end

    cons1 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], [x[1], y]), 0.0)
    
    MOI.add_constraint(optimizer, cons1, MOI.EqualTo(2.0))

    obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], [x[1], y]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 2

    @test MOI.get(optimizer, MOI.VariablePrimal(), x[1]) == 2

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(1.0))

    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 2

    @test MOI.get(optimizer, MOI.VariablePrimal(), x[1]) == 1


    new_obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], [x[1], x[2]]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), new_obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 1

end