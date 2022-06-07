@testset "QP - No parameters 1" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)

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
        quad_terms,
        MOI.ScalarAffineTerm.(q, x),
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

@testset "QP - No parameters 2" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)

    A = [0.0 1.0; 1.0 0.0]
    a = [0.0, 0.0]
    c = [2.0, 1.0]

    x = MOI.add_variables(optimizer, 2)

    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(1.0))
        MOI.add_constraint(optimizer, x_i, MOI.LessThan(5.0))
    end

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2]))

    constraint_function = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2])],
        MOI.ScalarAffineTerm.(a, x),
        0.0,
    )

    MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(9.0))

    obj_func =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 11.8, atol = ATOL)
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        5.0,
        atol = ATOL,
    )
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        1.8,
        atol = ATOL,
    )
end

@testset "QP - Parameter in affine constraint" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)

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
        quad_terms,
        MOI.ScalarAffineTerm.(q, x),
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
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)

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
        quad_terms,
        MOI.ScalarAffineTerm.(q, [x[1], x[2], y]),
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

@testset "QP - Quadratic constraint variable x variable + parameter in affine part" begin
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

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], x[2], x[2]))

    constraint_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, [x[1], y]),
        0.0,
    )

    cons_index =
        MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(25.0))

    obj_func =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(
        MOI.get(optimizer, MOI.ObjectiveValue()),
        9.0664,
        atol = ATOL,
    )
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        4.3665,
        atol = ATOL,
    )
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        1 / 3,
        atol = ATOL,
    )

    @test isapprox(
        MOI.get(optimizer, MOI.ConstraintDual(), cy),
        -MOI.get(optimizer, MOI.ConstraintDual(), cons_index),
        atol = ATOL,
    )

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(
        MOI.get(optimizer, MOI.ObjectiveValue()),
        8.6609,
        atol = ATOL,
    )
end

@testset "QP - Quadratic constraint variable x variable + parameter in affine part - variation to assess duals" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)

    A = [2.0 1.0; 1.0 2.0]
    a = [1.0, 2.0]
    c = [2.0, 1.0]

    x = MOI.add_variables(optimizer, 2)

    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], x[2], x[2]))

    constraint_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, [x[1], y]),
        0.0,
    )

    cons_index =
        MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(25.0))

    obj_func =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(
        MOI.get(optimizer, MOI.ObjectiveValue()),
        9.0664,
        atol = ATOL,
    )
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        4.3665,
        atol = ATOL,
    )
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        1 / 3,
        atol = ATOL,
    )

    @test isapprox(
        MOI.get(optimizer, MOI.ConstraintDual(), cy),
        -2 * MOI.get(optimizer, MOI.ConstraintDual(), cons_index),
        atol = ATOL,
    )

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(
        MOI.get(optimizer, MOI.ObjectiveValue()),
        8.2376,
        atol = ATOL,
    )
end

@testset "QP - Quadratic constraint parameter x variable" begin
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

    MOI.add_constraint(optimizer, x[1], MOI.LessThan(20.0))

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], y, y))

    constraint_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, x),
        0.0,
    )

    MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(30.0))

    obj_func =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 30.25, atol = ATOL)
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        0.5,
        atol = ATOL,
    )
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        29.25,
        atol = ATOL,
    )

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 22.0, atol = ATOL)
end

@testset "QP - Quadratic constraint variable x parameter" begin
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

    MOI.add_constraint(optimizer, x[1], MOI.LessThan(20.0))

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], y, y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], y, x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))

    constraint_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, x),
        0.0,
    )

    MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(30.0))

    obj_func =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 30.25, atol = ATOL)
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        0.5,
        atol = ATOL,
    )
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        29.25,
        atol = ATOL,
    )

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 22.0, atol = ATOL)
end

@testset "QP - Quadratic constraint parameter x parameter" begin
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

    MOI.add_constraint(optimizer, x[1], MOI.LessThan(20.0))

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
    z, cz = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]

    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], y, y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], y, z))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], z, z))

    constraint_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, x),
        0.0,
    )

    MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(30.0))

    obj_func =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 50.0, atol = ATOL)
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        20.0,
        atol = ATOL,
    )
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        10.0,
        atol = ATOL,
    )

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
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)

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
        quad_terms,
        MOI.ScalarAffineTerm.(q, [x[1], x[2], y]),
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

@testset "QP - Quadratic objective parameter x parameter" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)

    A = [0.0 1.0; 1.0 0.0]
    a = [1.0, 1.0]

    x = MOI.add_variables(optimizer, 2)

    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(1))
    z, cz = MOI.add_constrained_variable(optimizer, POI.Parameter(1))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], y, z))

    objective_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, x),
        0.0,
    )

    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 1.0, atol = ATOL)
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        0.0,
        atol = ATOL,
    )

    @test_throws ErrorException(
        "Cannot calculate the dual of a multiplicative parameter",
    ) MOI.get(optimizer, MOI.ConstraintDual(), cy)
    @test_throws ErrorException(
        "Cannot calculate the dual of a multiplicative parameter",
    ) MOI.get(optimizer, MOI.ConstraintDual(), cz)

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
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)

    A = [0.0 1.0; 1.0 0.0]
    a = [2.0, 1.0]

    x = MOI.add_variables(optimizer, 2)

    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end

    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(1))
    z, cz = MOI.add_constrained_variable(optimizer, POI.Parameter(1))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], x[2], x[2]))

    objective_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, [y, z]),
        0.0,
    )

    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 3.0, atol = ATOL)
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        0,
        atol = ATOL,
    )

    @test isapprox(
        MOI.get(optimizer, MOI.ConstraintDual(), cy),
        -2.0,
        atol = ATOL,
    )
    @test isapprox(
        MOI.get(optimizer, MOI.ConstraintDual(), cz),
        -1.0,
        atol = ATOL,
    )

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 5.0, atol = ATOL)

    @test isapprox(
        MOI.get(optimizer, MOI.ConstraintDual(), cy),
        -2.0,
        atol = ATOL,
    )
    @test isapprox(
        MOI.get(optimizer, MOI.ConstraintDual(), cz),
        -1.0,
        atol = ATOL,
    )

    MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(3.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 7.0, atol = ATOL)

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(5.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cz, POI.Parameter(5.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 15.0, atol = ATOL)
end

@testset "QP - Quadratic objective parameter in quadratic part" begin
    model = POI.Optimizer(Ipopt.Optimizer())
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    z = MOI.add_variable(model)
    p = first(MOI.add_constrained_variable.(model, POI.Parameter(1.0)))

    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))

    cons1 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 1.0], [x, y]), 0.0)
    ci1 = MOI.add_constraint(model, cons1, MOI.LessThan(4.0))
    cons2 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0], [x, y]), 0.0)
    ci2 = MOI.add_constraint(model, cons2, MOI.LessThan(4.0))

    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    obj_func = MOI.ScalarQuadraticFunction(
        [
            MOI.ScalarQuadraticTerm(1.0, x, x)
            MOI.ScalarQuadraticTerm(1.0, y, y)
        ],
        MathOptInterface.ScalarAffineTerm{Float64}[],
        0.0,
    )

    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        obj_func,
    )

    MOI.set(model, POI.QuadraticObjectiveCoef(), (x, y), 2p + 3)
    @test MOI.get(model, POI.QuadraticObjectiveCoef(), (x, y)) ≈
          MathOptInterface.ScalarAffineFunction{Int64}(
        MathOptInterface.ScalarAffineTerm{Int64}[MathOptInterface.ScalarAffineTerm{
            Int64,
        }(
            2,
            MathOptInterface.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1),
        )],
        3,
    )
    @test_throws ErrorException MOI.get(
        model,
        POI.QuadraticObjectiveCoef(),
        (x, z),
    )

    MOI.optimize!(model)

    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 32 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL

    MOI.set(model, POI.ParameterValue(), p, 2.0)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 128 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL

    model = POI.Optimizer(Ipopt.Optimizer())
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    p = first(MOI.add_constrained_variable.(model, POI.Parameter(1.0)))

    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))

    cons1 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 1.0], [x, y]), 0.0)
    ci1 = MOI.add_constraint(model, cons1, MOI.LessThan(4.0))
    cons2 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0], [x, y]), 0.0)
    ci2 = MOI.add_constraint(model, cons2, MOI.LessThan(4.0))

    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    obj_func = MOI.ScalarAffineFunction(
        [
            MOI.ScalarAffineTerm(1.0, x)
            MOI.ScalarAffineTerm(2.0, y)
        ],
        1.0,
    )

    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )

    MOI.set(model, POI.QuadraticObjectiveCoef(), (x, y), p)

    MOI.optimize!(model)

    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 61 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, POI.QuadraticObjectiveCoef(), (x, y)) ≈
          MathOptInterface.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1)

    MOI.set(model, POI.ParameterValue(), p, 2.0)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 77 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL

    model = POI.Optimizer(Ipopt.Optimizer())
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    p = first(MOI.add_constrained_variable.(model, POI.Parameter(1.0)))

    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))

    cons1 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 1.0], [x, y]), 0.0)
    ci1 = MOI.add_constraint(model, cons1, MOI.LessThan(4.0))
    cons2 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0], [x, y]), 0.0)
    ci2 = MOI.add_constraint(model, cons2, MOI.LessThan(4.0))

    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    obj_func = x

    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), obj_func)

    MOI.set(model, POI.QuadraticObjectiveCoef(), (x, y), p)

    MOI.optimize!(model)

    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 28 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL

    MOI.set(model, POI.ParameterValue(), p, 2.0)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 44 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL

    model = POI.Optimizer(Ipopt.Optimizer())
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    p = first(MOI.add_constrained_variable.(model, POI.Parameter(1.0)))

    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))

    cons1 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 1.0], [x, y]), 0.0)
    ci1 = MOI.add_constraint(model, cons1, MOI.LessThan(4.0))
    cons2 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0], [x, y]), 0.0)
    ci2 = MOI.add_constraint(model, cons2, MOI.LessThan(4.0))

    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.set(model, POI.QuadraticObjectiveCoef(), (x, y), p)

    MOI.optimize!(model)

    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 16 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL

    MOI.set(model, POI.ParameterValue(), p, 2.0)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 32 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL
end

@testset "JuMP direct model - QP - Quadratic objective parameter in quadratic part" begin
    optimizer = POI.Optimizer(Ipopt.Optimizer())
    model = direct_model(optimizer)
    MOI.set(model, MOI.Silent(), true)

    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, p in POI.Parameter(1.0))
    @constraint(model, 2x + y <= 4)
    @constraint(model, x + 2y <= 4)
    @objective(model, Max, (x^2 + y^2) / 2)
    MOI.set(
        backend(model),
        POI.QuadraticObjectiveCoef(),
        (index(x), index(y)),
        2index(p) + 3,
    )
    optimize!(model)

    @test MOI.get(
        backend(model),
        POI.QuadraticObjectiveCoef(),
        (index(x), index(y)),
    ) ≈ MathOptInterface.ScalarAffineFunction{Int64}(
        MathOptInterface.ScalarAffineTerm{Int64}[MathOptInterface.ScalarAffineTerm{
            Int64,
        }(
            2,
            MathOptInterface.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1),
        )],
        3,
    )
    @test objective_value(model) ≈ 32 / 3 atol = ATOL
    @test value(x) ≈ 4 / 3 atol = ATOL
    @test value(y) ≈ 4 / 3 atol = ATOL

    MOI.set(model, POI.ParameterValue(), p, 2.0)
    optimize!(model)

    @test objective_value(model) ≈ 128 / 9 atol = ATOL
    @test value(x) ≈ 4 / 3 atol = ATOL
    @test value(y) ≈ 4 / 3 atol = ATOL
end

@testset "JuMP direct model - Vector Constraints - RSOC - Parameter in quadratic part" begin
    """
        Problem RSOC

        min  x
        s.t. y ≥ 1/√2
            x² + (y-p)² ≤ 1

        in conic form:

        min  x
        s.t.  -1/√2 + y ∈ R₊
            1 - t ∈ {0}
            (t, x ,y-p) ∈ RSOC

        opt
            x* = 1/2*(max{1/√2,p}-p)^2
            y* = max{1/√2,p}
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
    MOI.set(model, MOI.Silent(), true)

    @variable(model, x)
    @variable(model, y)
    @variable(model, t)
    @variable(model, p in POI.Parameter(0))

    @constraint(model, [y - 1 / √2] in MOI.Nonnegatives(1))
    @constraint(model, [t - 1] in MOI.Zeros(1))
    @constraint(model, [t, x, y - p] in RotatedSecondOrderCone())
    @objective(model, Min, 1.0 * x)
    optimize!(model)

    @test objective_value(model) ≈ 1 / 4 atol = ATOL
    @test value(x) ≈ 1 / 4 atol = ATOL
    @test value(y) ≈ 1 / √2 atol = ATOL
    @test value(t) ≈ 1 atol = ATOL

    MOI.set(model, POI.ParameterValue(), p, 2)
    optimize!(model)

    @test objective_value(model) ≈ 0.0 atol = ATOL
    @test value(x) ≈ 0.0 atol = ATOL
    @test value(y) ≈ 2 atol = ATOL
end

@testset "ListOfConstraintTypesPresent" begin
    N = 10
    ipopt = Ipopt.Optimizer()
    model = POI.Optimizer(ipopt)
    x = MOI.add_variables(model, N / 2)
    y =
        first.(
            MOI.add_constrained_variable.(
                model,
                POI.Parameter.(ones(Int(N / 2))),
            ),
        )

    MOI.add_constraint(
        model,
        MOI.ScalarQuadraticFunction(
            MOI.ScalarQuadraticTerm.(1.0, x, y),
            MOI.ScalarAffineTerm{Float64}[],
            0.0,
        ),
        MOI.GreaterThan(1.0),
    )

    list_ctrs_types = MOI.get(model, MOI.ListOfConstraintTypesPresent())
    @test list_ctrs_types == [(
        MathOptInterface.ScalarQuadraticFunction{Float64},
        MathOptInterface.GreaterThan{Float64},
    )]
end
