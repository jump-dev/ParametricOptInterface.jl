@testset "SDP - scalar parameter" begin
    cached = MOI.Bridges.full_bridge_optimizer(
        MOIU.CachingOptimizer(
            MOIU.UniversalFallback(MOIU.Model{Float64}()),
            SCS.Optimizer(),
        ),
        Float64,
    )
    optimizer = POI.Optimizer(cached)

    m = direct_model(optimizer)
    set_silent(m)

    @variable(m, p in POI.Parameter(0))
    @variable(m, x[1:2, 1:2], Symmetric)
    @objective(m, Min, tr(x))
    @constraint(m, con, Symmetric(x .- [1+p 0; 0 1+p]) in PSDCone())

    optimize!(m)
    @test all(isapprox.(value.(x), [1 0; 0 1], atol = ATOL))

    MOI.set(m, POI.ParameterValue(), p, 1)
    optimize!(m)

    @test all(isapprox.(value.(x), [2 0; 0 2], atol = ATOL))
end

@testset "SDP - Matrix parameter" begin
    cached = MOI.Bridges.full_bridge_optimizer(
        MOIU.CachingOptimizer(
            MOIU.UniversalFallback(MOIU.Model{Float64}()),
            SCS.Optimizer(),
        ),
        Float64,
    )
    optimizer = POI.Optimizer(cached)

    m = direct_model(optimizer)
    set_silent(m)

    P1 = [1 2; 2 3]
    @variable(m, p[1:2, 1:2] in POI.Parameter.(P1))
    @variable(m, x[1:2, 1:2], Symmetric)
    @objective(m, Min, tr(x))
    @constraint(m, con, Symmetric(x - p) in PSDCone())

    optimize!(m)

    @test all(isapprox.(value.(x), P1, atol = ATOL))

    P2 = [1 2; 2 1]
    MOI.set.(m, POI.ParameterValue(), p, P2)
    optimize!(m)

    @test all(isapprox.(value.(x), P2, atol = ATOL))
end
