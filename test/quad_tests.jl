@testset "Quadratic tests" begin
    @testset "No parameters" begin
ipopt = Ipopt.Optimizer()
MOI.set(ipopt, MOI.RawParameter("print_level"), 0)
opt_in = MOIU.CachingOptimizer(MOIU.Model{Float64}(), ipopt)
optimizer = POI.ParametricOptimizer(opt_in)

Q = [4.0 1.0; 1.0 2.0]
q = [1.0; 1.0]
G = [1.0 1.0; 1.0 0.0; 0.0 1.0; -1.0 -1.0; -1.0 0.0; 0.0 -1.0]
h = [1.0; 0.7; 0.7; -1.0; 0.0; 0.0];

x = MOI.add_variables(optimizer, 2)

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
for i in 1:6
    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], x), 0.0),
        MOI.LessThan(h[i])
    )
end

    MOI.optimize!(optimizer)

    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 1.88, atol = ATOL)
    
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 0.3, atol = ATOL)
    @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 0.7, atol = ATOL)

    end

    @testset "Name 2" begin
        @test 1 == 1

    end

    @testset "Name 3" begin
        @test 1 == 1

    end

end