@testset "JuMP direct model - Linear Constraints - Affine parameters" begin
    optimizer = POI.Optimizer(GLPK.Optimizer())

    model = direct_model(optimizer)

    @variable(model, x[i = 1:2] >= 0)

    @variable(model, y in POI.Parameter(0))
    @variable(model, w in POI.Parameter(0))
    @variable(model, z in POI.Parameter(0))

    @constraint(model, 2 * x[1] + x[2] + y <= 4)
    @constraint(model, 1 * x[1] + 2 * x[2] + z <= 4)

    @objective(model, Max, 4 * x[1] + 3 * x[2] + w)

    optimize!(model)

    @test isapprox.(value(x[1]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(x[2]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(y), 0, atol = ATOL)

    # ===== Set parameter value =====
    MOI.set(model, POI.ParameterValue(), y, 2.0)
    optimize!(model)

    @test isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test isapprox.(value(y), 2.0, atol = ATOL)
end

@testset "JuMP direct model - Linear Constraints - Parameter x variable" begin
    optimizer = POI.Optimizer(GLPK.Optimizer())

    model = direct_model(optimizer)

    @variable(model, x[i = 1:2] >= 0)

    @variable(model, y in POI.Parameter(0))
    @variable(model, w in POI.Parameter(0))
    @variable(model, z in POI.Parameter(0))

    @constraint(model, 2 * x[1] + x[2] + y <= 4)
    @constraint(model, (1 + y) * x[1] + 2 * x[2] + z <= 4)

    @objective(model, Max, 4 * x[1] + 3 * x[2] + w)

    optimize!(model)

    @test isapprox.(value(x[1]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(x[2]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(y), 0, atol = ATOL)

    # ===== Set parameter value =====
    MOI.set(model, POI.ParameterValue(), y, 2.0)
    optimize!(model)

    @test isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test isapprox.(value(y), 2.0, atol = ATOL)
end

@testset "JuMP - Linear Constraints - Affine parameters" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))

    @variable(model, x[i = 1:2] >= 0)

    @variable(model, y in POI.Parameter(0))
    @variable(model, w in POI.Parameter(0))
    @variable(model, z in POI.Parameter(0))

    @constraint(model, 2 * x[1] + x[2] + y <= 4)
    @constraint(model, 1 * x[1] + 2 * x[2] + z <= 4)

    @objective(model, Max, 4 * x[1] + 3 * x[2] + w)

    optimize!(model)

    @test isapprox.(value(x[1]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(x[2]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(y), 0, atol = ATOL)

    # ===== Set parameter value =====
    MOI.set(model, POI.ParameterValue(), y, 2.0)
    optimize!(model)

    @test isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test isapprox.(value(y), 2.0, atol = ATOL)
end

@testset "JuMP - Linear Constraints - Parameter x variable" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))

    @variable(model, x[i = 1:2] >= 0)

    @variable(model, y in POI.Parameter(0))
    @variable(model, w in POI.Parameter(0))
    @variable(model, z in POI.Parameter(0))

    @constraint(model, 2 * x[1] + x[2] + y <= 4)
    @constraint(model, (1 + y) * x[1] + 2 * x[2] + z <= 4)

    @objective(model, Max, 4 * x[1] + 3 * x[2] + w)

    optimize!(model)

    @test isapprox.(value(x[1]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(x[2]), 4.0 / 3.0, atol = ATOL)
    @test isapprox.(value(y), 0, atol = ATOL)

    # ===== Set parameter value =====
    MOI.set(model, POI.ParameterValue(), y, 2.0)
    optimize!(model)

    @test isapprox.(value(x[1]), 0.0, atol = ATOL)
    @test isapprox.(value(x[2]), 2.0, atol = ATOL)
    @test isapprox.(value(y), 2.0, atol = ATOL)
end

@testset "JuMP prints" begin
    model = direct_model(
        ParametricOptInterface.Optimizer(
            MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
                MOI.Utilities.AUTOMATIC,
            ),
        ),
    );

    vx = @variable(model, x[i = 1:5])
    vp = @variable(model, p[i = 1:5] in ParametricOptInterface.Parameter.(-1))
    c1 = @constraint(model, con, sum(x) + sum(p) >= 1)
    c2 = @constraint(model, conq, sum(x.*p) >= 1)
    c3 = @constraint(model, conqa, sum(x.*p) + x[1] * x[1] >= 1)

    o1 = @objective(model, Min, sum(x) + sum(p) + 1)
    o2 = @objective(model, Min, sum(x.*p) + 1)
    o3 = @objective(model, Min, sum(x.*p) + x[1] * x[1] + 3)

    println("Printing variables:")
    show(vx)
    println("\nPrinting parameters:")
    show(vp)
    println("\nPrinting saf:")
    show(c1)
    println("\nPrinting sqf1:")
    show(c2)
    println("\nPrinting sqf2:")
    show(c3)
    println("\nPrinting safobj:")
    show(o1)
    println("\nPrinting sqf1obj:")
    show(o2)
    println("\nPrinting sqf2obj:")
    show(o3)
    print("\n")
end