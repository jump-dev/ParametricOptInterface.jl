@testset "Basic tests" begin
    optimizer = POI.ParametricOptimizer(GLPK.Optimizer())

    MOI.set(optimizer, MOI.Silent(), true)

    x = MOI.add_variables(optimizer,2)
    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
    z = MOI.VariableIndex(4)
    cz = MOI.ConstraintIndex{MOI.SingleVariable, POI.Parameter}(4)

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(0.0))
    end

    @test_throws ErrorException("Cannot constrain a parameter") MOI.add_constraint(optimizer, MOI.SingleVariable(y), MOI.EqualTo(0.0))

    @test_throws ErrorException("Variable not in the model") MOI.add_constraint(optimizer, MOI.SingleVariable(z), MOI.GreaterThan(0.0))

    cons1 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], [x[1], y]), 0.0)
    
    MOI.add_constraint(optimizer, cons1, MOI.EqualTo(2.0))

    obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], [x[1], y]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 2

    @test MOI.get(optimizer, MOI.VariablePrimal(), x[1]) == 2

    @test_throws ErrorException("Variable not in the model") MOI.get(optimizer, MOI.VariablePrimal(), z)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(1.0))

    @test_throws ErrorException("Parameter not in the model") MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(1.0))

    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 2

    @test MOI.get(optimizer, MOI.VariablePrimal(), x[1]) == 1


    new_obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], [x[1], x[2]]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), new_obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 1

end

@testset "Quadratic objective parameter x parameter" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawParameter("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.ParametricOptimizer(opt_in)

    A = [0.0 1.0; 1.0 0.0]
    a = [1.0, 1.0]

    x = MOI.add_variables(optimizer, 2)

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(0.0))
    end

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(1))
    z, cz = MOI.add_constrained_variable(optimizer, POI.Parameter(1))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1,2], y, z))

    objective_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(a, x),
                            quad_terms,
                            0.0
                        )

    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 1.0, atol = ATOL)
    @test MOI.get(optimizer, MOI.VariablePrimal(), x[1]) == 0

    @test_throws ErrorException("Cannot calculate the dual of a multiplicative parameter") MOI.get(optimizer, MOI.ConstraintDual(), cy)
    @test_throws ErrorException("Cannot calculate the dual of a multiplicative parameter") MOI.get(optimizer, MOI.ConstraintDual(), cz)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 2.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(3.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 6.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(5.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(5.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 25.0, atol = ATOL)

end

@testset "Quadratic objective parameter in affine part" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawParameter("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.ParametricOptimizer(opt_in)

    A = [0.0 1.0; 1.0 0.0]
    a = [2.0, 1.0]

    x = MOI.add_variables(optimizer, 2)

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(0.0))
    end

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(1))
    z, cz = MOI.add_constrained_variable(optimizer, POI.Parameter(1))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1,1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1,2], x[1], x[2]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2,2], x[2], x[2]))

    objective_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(a, [y,z]),
                            quad_terms,
                            0.0
                        )

    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 3.0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 0, atol = ATOL)

    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cy), 2.0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cz), 1.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 5.0, atol = ATOL)

    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cy), 2.0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cz), 1.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(3.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 7.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(5.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(5.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 15.0, atol = ATOL)

end

@testset "Quadratic constraint parameter x parameter" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawParameter("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.ParametricOptimizer(opt_in)

    A = [2.0 1.0; 1.0 2.0]
    a = [1.0, 1.0]

    c = [2.0, 1.0]

    x = MOI.add_variables(optimizer, 2)

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(0.0))
    end

    MOI.add_constraint(optimizer, MOI.SingleVariable(x[1]), MOI.LessThan(20.0))

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
    z, cz = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1,1], y, y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1,2], y, z))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2,2], z, z))

    constraint_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(a, x),
                            quad_terms,
                            0.0
                        )

    MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(30.0))

    
    obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 50.0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 20.0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 10.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 42.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(1.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 36.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(-1.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(-1.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 45.0, atol = ATOL)

end

@testset "Quadratic constraint no parameters" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawParameter("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.ParametricOptimizer(opt_in)

    A = [0.0 1.0; 1.0 0.0]
    a = [0.0, 0.0]
    c = [2.0, 1.0]

    x = MOI.add_variables(optimizer, 2)

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(1.0))
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.LessThan(5.0))
    end


    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1,2], x[1], x[2]))

    constraint_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(a, x),
                            [MOI.ScalarQuadraticTerm(A[1,2], x[1], x[2])],
                            0.0
                        )

    MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(9.0))

    
    obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 11.8, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 5.0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 1.8, atol = ATOL)

end

@testset "Quadratic constraint parameter x variable" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawParameter("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.ParametricOptimizer(opt_in)

    A = [2.0 1.0; 1.0 2.0]
    a = [1.0, 1.0]

    c = [2.0, 1.0]

    x = MOI.add_variables(optimizer, 2)

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(0.0))
    end

    MOI.add_constraint(optimizer, MOI.SingleVariable(x[1]), MOI.LessThan(20.0))

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1,1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1,2], x[1], y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2,2], y, y))

    constraint_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(a, x),
                            quad_terms,
                            0.0
                        )

    MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(30.0))

    obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 30.25, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 0.5, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 29.25, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 22.0, atol = ATOL)
end

@testset "Quadratic constraint variable x parameter" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawParameter("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.ParametricOptimizer(opt_in)

    A = [2.0 1.0; 1.0 2.0]
    a = [1.0, 1.0]

    c = [2.0, 1.0]

    x = MOI.add_variables(optimizer, 2)

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(0.0))
    end

    MOI.add_constraint(optimizer, MOI.SingleVariable(x[1]), MOI.LessThan(20.0))

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2,2], y, y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1,2], y, x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2,2], x[1], x[1]))

    constraint_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(a, x),
                            quad_terms,
                            0.0
                        )

    MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(30.0))

    obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 30.25, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 0.5, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 29.25, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 22.0, atol = ATOL)
end

@testset "Quadratic constraint variable x variable + parameter in affine part" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawParameter("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.ParametricOptimizer(opt_in)

    A = [2.0 1.0; 1.0 2.0]
    a = [1.0, 1.0]
    c = [2.0, 1.0]

    x = MOI.add_variables(optimizer, 2)

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(0.0))
    end

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1,1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1,2], x[1], x[2]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2,2], x[2], x[2]))

    constraint_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(a, [x[1], y]),
                            quad_terms,
                            0.0
                        )

    cons_index = MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(25.0))

    obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 9.0664, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4.3665, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 1/3, atol = ATOL)

    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cy), MOI.get(optimizer, MOI.ConstraintDual(), cons_index), atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 8.6609, atol = ATOL)
end


@testset "Quadratic constraint variable x variable + parameter in affine part - variation to assess duals" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawParameter("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.ParametricOptimizer(opt_in)

    A = [2.0 1.0; 1.0 2.0]
    a = [1.0, 2.0]
    c = [2.0, 1.0]

    x = MOI.add_variables(optimizer, 2)

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(0.0))
    end

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1,1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1,2], x[1], x[2]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2,2], x[2], x[2]))

    constraint_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(a, [x[1], y]),
                            quad_terms,
                            0.0
                        )

    cons_index = MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(25.0))

    obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 9.0664, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4.3665, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 1/3, atol = ATOL)

    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), cy), 2*MOI.get(optimizer, MOI.ConstraintDual(), cons_index), atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 8.2376, atol = ATOL)
end




