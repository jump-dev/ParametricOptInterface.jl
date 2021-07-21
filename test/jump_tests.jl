@testset "JuMP direct model - Linear Constraints - Affine parameters" begin
    optimizer = POI.ParametricOptimizer(GLPK.Optimizer())

    model = direct_model(optimizer)

    @variable(model, x[i=1:2] >= 0)

    @variable(model, y in POI.Parameter(0))
    @variable(model, w in POI.Parameter(0))
    @variable(model, z in POI.Parameter(0))

    @constraint(model, 2*x[1] + x[2] + y <= 4)
    @constraint(model, 1*x[1] + 2*x[2] + z <= 4)

    @objective(model, Max, 4*x[1] + 3*x[2] + w)

    optimize!(model)

    @test isapprox.(value(x[1]), 4.0/3.0, atol = ATOL)
    @test isapprox.(value(x[2]), 4.0/3.0, atol = ATOL)
    @test isapprox.(value(y), 0, atol = ATOL)

    # ===== Set parameter value =====
    MOI.set(model, POI.ParameterValue(), y, 2.0)
    optimize!(model)

    @test isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test isapprox.(value(y), 2.0, atol = ATOL)
end

@testset "JuMP direct model - Linear Constraints - Parameter x variable" begin
    optimizer = POI.ParametricOptimizer(GLPK.Optimizer())

    model = direct_model(optimizer)

    @variable(model, x[i=1:2] >= 0)

    @variable(model, y in POI.Parameter(0))
    @variable(model, w in POI.Parameter(0))
    @variable(model, z in POI.Parameter(0))

    @constraint(model, 2*x[1] + x[2] + y <= 4)
    @constraint(model, (1+y)*x[1] + 2*x[2] + z <= 4)

    @objective(model, Max, 4*x[1] + 3*x[2] + w)

    optimize!(model)

    @test isapprox.(value(x[1]), 4.0/3.0, atol = ATOL)
    @test isapprox.(value(x[2]), 4.0/3.0, atol = ATOL)
    @test isapprox.(value(y), 0, atol = ATOL)

    # ===== Set parameter value =====
    MOI.set(model, POI.ParameterValue(), y, 2.0)
    optimize!(model)

    @test isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test isapprox.(value(y), 2.0, atol = ATOL)
end

@testset "JuMP - Linear Constraints - Affine parameters" begin
    model = Model(() -> POI.ParametricOptimizer(GLPK.Optimizer()))

    @variable(model, x[i=1:2] >= 0)

    @variable(model, y in POI.Parameter(0))
    @variable(model, w in POI.Parameter(0))
    @variable(model, z in POI.Parameter(0))

    @constraint(model, 2*x[1] + x[2] + y <= 4)
    @constraint(model, 1*x[1] + 2*x[2] + z <= 4)

    @objective(model, Max, 4*x[1] + 3*x[2] + w)

    optimize!(model)

    @test isapprox.(value(x[1]), 4.0/3.0, atol = ATOL)
    @test isapprox.(value(x[2]), 4.0/3.0, atol = ATOL)
    @test isapprox.(value(y), 0, atol = ATOL)

    # ===== Set parameter value =====
    MOI.set(model, POI.ParameterValue(), y, 2.0)
    optimize!(model)

    @test isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test isapprox.(value(y), 2.0, atol = ATOL)
end

@testset "JuMP - Linear Constraints - Parameter x variable" begin
    model = Model(() -> POI.ParametricOptimizer(GLPK.Optimizer()))

    @variable(model, x[i=1:2] >= 0)

    @variable(model, y in POI.Parameter(0))
    @variable(model, w in POI.Parameter(0))
    @variable(model, z in POI.Parameter(0))

    @constraint(model, 2*x[1] + x[2] + y <= 4)
    @constraint(model, (1+y)*x[1] + 2*x[2] + z <= 4)

    @objective(model, Max, 4*x[1] + 3*x[2] + w)

    @test_broken optimize!(model)

    @test_broken isapprox.(value(x[1]), 4.0/3.0, atol = ATOL)
    @test_broken isapprox.(value(x[2]), 4.0/3.0, atol = ATOL)
    @test_broken isapprox.(value(y), 0, atol = ATOL)

    # ===== Set parameter value =====
    MOI.set(model, POI.ParameterValue(), y, 2.0)
    @test_broken optimize!(model)

    @test_broken isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test_broken isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test_broken isapprox.(value(y), 2.0, atol = ATOL)
end

@testset "JuMP direct model - Variable in POI.Parameters" begin
    optimizer = POI.ParametricOptimizer(GLPK.Optimizer())

    model = direct_model(optimizer)

    @variable(model, x[i=1:2] >= 0)

    @variable(model, y[i=1:3] in POI.Parameters(zeros(3)))

    @constraint(model, 2*x[1] + x[2] + y[1] <= 4)
    @constraint(model, 1*x[1] + 2*x[2] + y[3] <= 4)

    @objective(model, Max, 4*x[1] + 3*x[2] + y[2])

    optimize!(model)

    @test isapprox.(value(x[1]), 4.0/3.0, atol = ATOL)
    @test isapprox.(value(x[2]), 4.0/3.0, atol = ATOL)
    @test isapprox.(value(y[1]), 0, atol = ATOL)

    # ===== Set parameter value =====
    MOI.set(model, POI.ParameterValue(), y[1], 2.0)
    optimize!(model)

    @test isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test isapprox.(value(y[1]), 2.0, atol = ATOL)
end

@testset "JuMP model - Variable in POI.Parameters" begin
    model = Model(() -> POI.ParametricOptimizer(GLPK.Optimizer()))

    @variable(model, x[i=1:2] >= 0)

    @variable(model, y[i=1:3] in POI.Parameters(zeros(3)))

    @constraint(model, 2*x[1] + x[2] + y[1] <= 4)
    @constraint(model, 1*x[1] + 2*x[2] + y[3] <= 4)

    @objective(model, Max, 4*x[1] + 3*x[2] + y[2])

    @test_broken optimize!(model)

    @test_broken isapprox.(value(x[1]), 4.0/3.0, atol = ATOL)
    @test_broken isapprox.(value(x[2]), 4.0/3.0, atol = ATOL)
    @test_broken isapprox.(value(y[1]), 0, atol = ATOL)

    # ===== Set parameter value =====
    @test_broken MOI.set(model, POI.ParameterValue(), y[1], 2.0)
    @test_broken optimize!(model)

    @test_broken isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test_broken isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test_broken isapprox.(value(y[1]), 2.0, atol = ATOL)
end