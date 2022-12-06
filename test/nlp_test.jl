# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

@testset "NLP with parameters" begin
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    cached =
        () -> MOI.Bridges.full_bridge_optimizer(
            MOIU.CachingOptimizer(
                MOIU.UniversalFallback(MOIU.Model{Float64}()),
                ipopt,
            ),
            Float64,
        )
    POI_cached_optimizer() = ParametricOptInterface.Optimizer(cached())

    model = Model(() -> POI_cached_optimizer())

    @variable(model, x)
    @variable(model, y)
    @variable(model, z in ParametricOptInterface.Parameter(10))
    @constraint(model, x + y >= z)
    @NLobjective(model, Min, x^2 + y^2)

    optimize!(model)
    objective_value(model)
    @test value(x) ≈ 5
    MOI.get(model, ParametricOptInterface.ParameterDual(), z)
    MOI.set(model, ParametricOptInterface.ParameterValue(), z, 2.0)
    optimize!(model)
    @test objective_value(model) ≈ 2 atol = 1e-3
    @test value(x) ≈ 1

    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    model = Model(() -> ParametricOptInterface.Optimizer(ipopt))

    @variable(model, x)
    @variable(model, z in ParametricOptInterface.Parameter(10))
    MOI.set(
        model,
        ParametricOptInterface.ConstraintsInterpretation(),
        ParametricOptInterface.ONLY_BOUNDS,
    )
    @constraint(model, x >= z)
    @NLobjective(model, Min, x^2)

    optimize!(model)
    objective_value(model)
    @test value(x) ≈ 10
    MOI.get(model, ParametricOptInterface.ParameterDual(), z)
    MOI.set(model, ParametricOptInterface.ParameterValue(), z, 2.0)
    optimize!(model)
    @test objective_value(model) ≈ 4 atol = 1e-3
    @test value(x) ≈ 2
end
