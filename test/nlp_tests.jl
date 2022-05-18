@testset "NLP with parameters" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    cached = () -> MOI.Bridges.full_bridge_optimizer(
        MOIU.CachingOptimizer(
            MOIU.UniversalFallback(MOIU.Model{Float64}()),
            ipopt,
        ),
        Float64,
    )
    POI_cached_optimizer() = ParametricOptInterface.Optimizer(cached())

    m = Model(() -> POI_cached_optimizer())

    @variable(m, x)
    @variable(m, z in ParametricOptInterface.Parameter(10))
    # MOI.set(m, ParametricOptInterface.ConstraintsInterpretation(), ParametricOptInterface.ONLY_BOUNDS)
    @constraint(m, x >= z)
    @NLobjective(m, Min, x^2)

    optimize!(m)
    objective_value(m)
    @test value(x) ≈ 10
    MOI.get(m, ParametricOptInterface.ParameterDual(), z)
    MOI.set(m, ParametricOptInterface.ParameterValue(), z, 2.0)
    optimize!(m)
    objective_value(m)
    @test value(x) ≈ 2
end