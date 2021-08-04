@testset "JuMP NLP models" begin
    @test MOI.supports(POI.ParametricOptimizer(Ipopt.Optimizer()), MOI.NLPBlock())

    model = Model(() -> POI.ParametricOptimizer(Ipopt.Optimizer()))

    @variable(model, x[i=1:2] >= 0)
    @variable(model, y in POI.Parameter(10))
    @constraint(model, sum(x) + y >= 0)
    @NLobjective(model, Min, x[1]^2 + x[2]^2)
    optimize!(model)
end
