
@testset "QP - No parameters" begin
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
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
        end
    end

    objective_function = MOI.ScalarQuadraticFunction(
        MOI.ScalarAffineTerm.(q, x),
        quad_terms,
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add constraints
    for i in 1:6
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i, :], x), 0.0),
            MOI.LessThan(h[i]),
        )
    end

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 1.88, atol = ATOL)

    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        0.3,
        atol = ATOL,
    )
    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        0.7,
        atol = ATOL,
    )
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
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
        end
    end

    objective_function = MOI.ScalarQuadraticFunction(
        MOI.ScalarAffineTerm.(q, x),
        quad_terms,
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add constraints
    for i in 1:2
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.(G[i, :], [x[1], x[2], y]),
                0.0,
            ),
            MOI.GreaterThan(h[i]),
        )
    end

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 12.5, atol = ATOL)

    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        5.0,
        atol = ATOL,
    )
    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        -2.0,
        atol = ATOL,
    )

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(1.0))
    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 5.0, atol = ATOL)

    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        3.0,
        atol = ATOL,
    )
    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        -1.0,
        atol = ATOL,
    )
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
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
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
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.(G[1, :], [x[1], x[2], y, w]),
            0.0,
        ),
        MOI.GreaterThan(h[1]),
    )
    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.(G[2, :], [x[1], x[2], y, w]),
            0.0,
        ),
        MOI.GreaterThan(h[2]),
    )

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 12.5, atol = ATOL)
    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        5.0,
        atol = ATOL,
    )
    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        -2.0,
        atol = ATOL,
    )

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(1.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cw, POI.Parameter(2.0))
    MOI.optimize!(optimizer)

    @test isapprox(
        MOI.get(optimizer, MOI.ObjectiveValue()),
        5.7142,
        atol = ATOL,
    )
    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        2.1428,
        atol = ATOL,
    )
    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        -0.4285,
        atol = ATOL,
    )
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
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
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
    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        5.0,
        atol = ATOL,
    )
    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        -2.0,
        atol = ATOL,
    )

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(1.0))
    MOI.optimize!(optimizer)

    # @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 5.7142, atol = ATOL)
    # @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 2.1428, atol = ATOL)
    # @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), -0.4285, atol = ATOL)

end
