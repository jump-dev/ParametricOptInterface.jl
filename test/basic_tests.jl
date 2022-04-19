@testset "Basic tests" begin
    """
        min x₁ + y
            x₁ + y = 2
            x₁,x₂ ≥ 0

        opt
            x* = {2-y,0}
            obj = 2
    """

    optimizer = POI.Optimizer(GLPK.Optimizer())

    MOI.set(optimizer, MOI.Silent(), true)

    x = MOI.add_variables(optimizer, 2)
    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
    z = MOI.VariableIndex(4)
    cz = MOI.ConstraintIndex{MOI.VariableIndex,POI.Parameter}(4)

    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end

    @test_throws ErrorException("Cannot constrain a parameter") MOI.add_constraint(
        optimizer,
        y,
        MOI.EqualTo(0.0),
    )

    @test_throws ErrorException("Variable not in the model") MOI.add_constraint(
        optimizer,
        z,
        MOI.GreaterThan(0.0),
    )

    cons1 = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], [x[1], y]),
        0.0,
    )

    c1 = MOI.add_constraint(optimizer, cons1, MOI.EqualTo(2.0))

    obj_func = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], [x[1], y]),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 2

    @test MOI.get(optimizer, MOI.VariablePrimal(), x[1]) == 2

    @test_throws ErrorException("Variable not in the model") MOI.get(
        optimizer,
        MOI.VariablePrimal(),
        z,
    )

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(1.0))

    @test_throws ErrorException("Parameter not in the model") MOI.set(
        optimizer,
        MOI.ConstraintSet(),
        cz,
        POI.Parameter(1.0),
    )

    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 2

    @test MOI.get(optimizer, MOI.VariablePrimal(), x[1]) == 1

    """
        min x₁ + x₂
            x₁ + y = 2
            x₁,x₂ ≥ 0

        opt
            x* = {2-y,0}
            obj = 2-y
    """

    new_obj_func = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], [x[1], x[2]]),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        new_obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 1

    @test MOI.supports(optimizer, MOI.VariableName(), MOI.VariableIndex)
    @test MOI.get(optimizer, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    @test MOI.get(optimizer, MOI.VariableName(), x[1]) == ""
    @test MOI.get(optimizer, MOI.ConstraintName(), c1) == ""
end

@testset "Special cases of getters" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)

    A = [2.0 1.0; 1.0 2.0]
    a = [1.0, 1.0]
    c = [2.0, 1.0]

    x = MOI.add_variables(optimizer, 2)

    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], x[2], y))

    constraint_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, [x[1], y]),
        0.0,
    )

    cons_index =
        MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(25.0))

    obj_func = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(A[2, 2], x[2], y)],
        MOI.ScalarAffineTerm.(c, [x[1], x[2]]),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    @test MOI.get(optimizer, MOI.ObjectiveFunctionType()) ==
          MOI.ScalarQuadraticFunction{Float64}
    @test MOI.get(optimizer, MOI.NumberOfVariables()) == 3
end
