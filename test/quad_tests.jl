@testset "QP - No parameters 1" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawParameter("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.ParametricOptimizer(opt_in)

    Q = [4.0 1.0; 1.0 2.0]
    q = [1.0; 1.0]
    G = [1.0 1.0; 1.0 0.0; 0.0 1.0; -1.0 -1.0; -1.0 0.0; 0.0 -1.0]
    h = [1.0; 0.7; 0.7; -1.0; 0.0; 0.0]

    x = MOI.add_variables(optimizer, 2)

    # define objective
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i = 1:2
        for j = i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
        end
    end

    objective_function =
        MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm.(q, x), quad_terms, 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add constraints
    for i = 1:6
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i, :], x), 0.0),
            MOI.LessThan(h[i]),
        )
    end

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 1.88, atol = ATOL)

    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 0.3, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 0.7, atol = ATOL)

end

@testset "QP - No parameters 2" begin
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

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2]))

    constraint_function = MOI.ScalarQuadraticFunction(
        MOI.ScalarAffineTerm.(a, x),
        [MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2])],
        0.0,
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

@testset "QP - Parameter in affine constraint" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawParameter("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.ParametricOptimizer(opt_in)

    Q = [3.0 2.0; 2.0 1.0]
    q = [1.0, 6.0]
    G = [2.0 3.0 1.0; 1.0 1.0 1.0]
    h = [4.0; 3.0]

    x = MOI.add_variables(optimizer, 2)

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    # define objective
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i = 1:2
        for j = i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
        end
    end

    objective_function =
        MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm.(q, x), quad_terms, 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add constraints
    for i = 1:2
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i, :], [x[1], x[2], y]), 0.0),
            MOI.GreaterThan(h[i]),
        )
    end

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 12.5, atol = ATOL)

    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 5.0, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), -2.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(1.0))
    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 5.0, atol = ATOL)

    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 3.0, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), -1.0, atol = ATOL)

end

@testset "QP - Parameter in affine part of quadratic constraint" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawParameter("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.ParametricOptimizer(opt_in)

    Q = [3.0 2.0; 2.0 1.0]
    q = [1.0, 6.0, 1.0]
    G = [2.0 3.0 1.0 0.0; 1.0 1.0 0.0 1.0]
    h = [4.0; 3.0]

    x = MOI.add_variables(optimizer, 2)

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
    w, cw = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    # define objective
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i = 1:2
        for j = i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
        end
    end

    objective_function = MOI.ScalarQuadraticFunction(
        MOI.ScalarAffineTerm.(q, [x[1], x[2], y]),
        quad_terms,
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1, :], [x[1], x[2], y, w]), 0.0),
        MOI.GreaterThan(h[1]),
    )
    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[2, :], [x[1], x[2], y, w]), 0.0),
        MOI.GreaterThan(h[2]),
    )


    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 12.5, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 5.0, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), -2.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(1.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cw, POI.Parameter(2.0))
    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 5.7142, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 2.1428, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), -0.4285, atol = ATOL)

end

@testset "QP - Quadratic constraint variable x variable + parameter in affine part" begin
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

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], x[2], x[2]))

    constraint_function =
        MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm.(a, [x[1], y]), quad_terms, 0.0)

    cons_index = MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(25.0))

    obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 9.0664, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4.3665, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 1 / 3, atol = ATOL)

    @test isapprox(
        MOI.get(optimizer, MOI.ConstraintDual(), cy),
        MOI.get(optimizer, MOI.ConstraintDual(), cons_index),
        atol = ATOL,
    )

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 8.6609, atol = ATOL)
end

@testset "QP - Quadratic constraint variable x variable + parameter in affine part - variation to assess duals" begin
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

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], x[2], x[2]))

    constraint_function =
        MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm.(a, [x[1], y]), quad_terms, 0.0)

    cons_index = MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(25.0))

    obj_func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 9.0664, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4.3665, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 1 / 3, atol = ATOL)

    @test isapprox(
        MOI.get(optimizer, MOI.ConstraintDual(), cy),
        2 * MOI.get(optimizer, MOI.ConstraintDual(), cons_index),
        atol = ATOL,
    )

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 8.2376, atol = ATOL)
end

@testset "QP - Quadratic constraint parameter x variable" begin
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

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], y, y))

    constraint_function =
        MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm.(a, x), quad_terms, 0.0)

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

@testset "QP - Quadratic constraint variable x parameter" begin
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

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], y, y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], y, x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))

    constraint_function =
        MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm.(a, x), quad_terms, 0.0)

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

@testset "QP - Quadratic constraint parameter x parameter" begin
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

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], y, y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], y, z))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], z, z))

    constraint_function =
        MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm.(a, x), quad_terms, 0.0)

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

@testset "QP - Quadratic parameter becomes constant" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawParameter("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.ParametricOptimizer(opt_in)

    Q = [3.0 2.0 0.0; 2.0 1.0 0.0; 0.0 0.0 1.0]
    q = [1.0, 6.0, 0.0]
    G = [2.0 3.0; 1.0 1.0]
    h = [4.0; 3.0]

    x = MOI.add_variables(optimizer, 2)

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    # define objective
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i = 1:2
        for j = i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
        end
    end
    # adding terms associated with parameter
    push!(quad_terms, MOI.ScalarQuadraticTerm(Q[1, 3], x[1], y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(Q[2, 3], x[2], y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(Q[3, 3], y, y))


    objective_function = MOI.ScalarQuadraticFunction(
        MOI.ScalarAffineTerm.(q, [x[1], x[2], y]),
        quad_terms,
        0.0,
    )

    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1, :], x), 0.0),
        MOI.GreaterThan(h[1]),
    )
    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[2, :], x), 0.0),
        MOI.GreaterThan(h[2]),
    )

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 12.5, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 5.0, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), -2.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(1.0))
    MOI.optimize!(optimizer)

    # @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 5.7142, atol = ATOL)
    # @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 2.1428, atol = ATOL)
    # @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), -0.4285, atol = ATOL)

end

@testset "QP - Quadratic objective parameter x parameter" begin
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
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], y, z))

    objective_function =
        MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm.(a, x), quad_terms, 0.0)

    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 1.0, atol = ATOL)
    @test MOI.get(optimizer, MOI.VariablePrimal(), x[1]) == 0

    @test_throws ErrorException("Cannot calculate the dual of a multiplicative parameter") MOI.get(
        optimizer,
        MOI.ConstraintDual(),
        cy,
    )
    @test_throws ErrorException("Cannot calculate the dual of a multiplicative parameter") MOI.get(
        optimizer,
        MOI.ConstraintDual(),
        cz,
    )

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 2.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(3.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 6.0, atol = ATOL)

    MOI.set(optimizer, POI.ParameterValue(), y, 5)
    MOI.set(optimizer, POI.ParameterValue(), z, 5.0)
    @test_throws ErrorException MOI.set(
        optimizer,
        POI.ParameterValue(),
        MOI.VariableIndex(10872368175),
        5.0,
    )
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 25.0, atol = ATOL)

end

@testset "QP - Quadratic objective parameter in affine part" begin
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
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], x[2], x[2]))

    objective_function =
        MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm.(a, [y, z]), quad_terms, 0.0)

    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
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


