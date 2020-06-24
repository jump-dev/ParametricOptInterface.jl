@testset "Quadratic objective - parameter in the affine part" begin

    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), Ipopt.Optimizer())

    optimizer = POI.ParametricOptimizer(opt_in)

    Q = [3.0 2.0; 2.0 1.0]
    q = [1.0, 6.0]
    G = [2.0 3.0; 1.0 1.0]
    h = [4.0; 3.0]

    x = MOI.add_variables(optimizer, 2)
    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(
                quad_terms, 
                MOI.ScalarQuadraticTerm(Q[i,j],x[i],x[j])
            )
        end
    end
    aff_terms = MOI.ScalarAffineTerm.(q, [x[1], y])

    objF = MOI.ScalarQuadraticFunction(
                            aff_terms,
                            quad_terms,
                            2.0
                        )
                           
    for i in 1:2
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], x), 0.0),
            MOI.GreaterThan(h[i])
        )
    end

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(0.0))
    end

    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objF)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 6.5, atol = ATOL)
    
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 0.0, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 3.0, atol = ATOL)

    MOI.get(optimizer, MOI.VariablePrimal(), y)
    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(5))
    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 36.5, atol = ATOL)

end



@testset "Quadratic constraints - parameters in the quadratic part" begin

    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), Ipopt.Optimizer())

    optimizer = POI.ParametricOptimizer(opt_in)

    Q = [3.0 2.0; 2.0 1.0]
    q = [1.0, 6.0]
    G = [2.0 3.0; 1.0 1.0]
    h = [4.0; 3.0]
    R = [4.0, 2.0, 1.0, 0.5]

    x = MOI.add_variables(optimizer, 2)
    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(2))
    w, cw = MOI.add_constrained_variable(optimizer, POI.Parameter(3))

    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]

    push!(quad_terms, MOI.ScalarQuadraticTerm(Q[1,1],y,y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(Q[1,2],w,y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(Q[2,2],w,w))
    

    aff_terms = MOI.ScalarAffineTerm.(q, x)

    quad_cons = MOI.ScalarQuadraticFunction(
                            aff_terms,
                            quad_terms,
                            2.0
                        )
                           
    for i in 1:2
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], x), 0.0),
            MOI.GreaterThan(h[i])
        )
    end

    for x_i in x
        MOI.add_constraint(optimizer, MOI.SingleVariable(x_i), MOI.GreaterThan(3.0))
    end

    objF = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(R, [x[1], x[2], y, w]), 0.0)

    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objF)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 21.5, atol = ATOL)
    
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 3.0, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 3.0, atol = ATOL)

    MOI.get(optimizer, MOI.VariablePrimal(), y)
    @test MOI.get(optimizer, MOI.VariablePrimal(), w) == 3

    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(5))
    MOI.set(optimizer, MOI.ConstraintSet(), cw, POI.Parameter(10))
    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.VariablePrimal(), w) == 10

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 28, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 3.0, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 3.0, atol = ATOL)

end


@testset "Affine constraint with parameters in QP" begin

    opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), Ipopt.Optimizer())

    optimizer = POI.ParametricOptimizer(opt_in)

    Q = [4.0 1.0; 1.0 2.0]
    q = [1.0; 1.0]
    G = [1.0 1.0; 1.0 0.0; 0.0 1.0; -1.0 -1.0; -1.0 0.0; 0.0 -1.0]
    h = [1.0; 0.7; 0.7; -1.0; 0.0; 0.0];

    x = MOI.add_variables(optimizer, 2)
    y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))

    # define objective
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(
                quad_terms, 
                MOI.ScalarQuadraticTerm(Q[i,j],x[i],x[j])
            )
        end
    end

    objective_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(q, x),
                            quad_terms,
                            0.0
                        )
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)


    # add constraints
    for i in 1:2
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], [x[1], y]), 0.0),
            MOI.LessThan(h[i])
        )
    end

    for i in 3:4
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], [x[2], y]), 0.0),
            MOI.LessThan(h[i])
        )
    end


    for i in 5:6
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], x), 0.0),
            MOI.LessThan(h[i])
        )
    end


    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 2.0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 1.0, atol = ATOL)


    MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(-1))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 6.0, atol = ATOL)
    @test isapprox(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 2.0, atol = ATOL)

end
