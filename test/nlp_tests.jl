@testset "JuMP NLP models" begin
    @test MOI.supports(POI.ParametricOptimizer(Ipopt.Optimizer()), MOI.NLPBlock())

    model = Model(() -> POI.ParametricOptimizer(Ipopt.Optimizer()))

    y = @variable(model, y in POI.Parameter(10))
    @variable(model, x[i=1:2] >= 0)
    @constraint(model, x[1] >= y)
    @constraint(model, sum(x) + 2*y >= 0)
    @NLobjective(model, Min, x[1]^2 + x[2]^2)
    optimize!(model)
    @test isapprox(JuMP.value(x[1]), JuMP.value(y))

    MOI.set(model, POI.ParameterValue(), y, 20.0)
    optimize!(model)
    @test isapprox(JuMP.value(x[1]), 20.0)
    @test isapprox(JuMP.value(y), 20.0)

    model = Model(() -> POI.ParametricOptimizer(Ipopt.Optimizer()))

    @variable(model, y in POI.Parameter(10))
    @variable(model, x[i=1:2] >= 0)
    @constraint(model, x[1] >= y)
    @NLconstraint(model, sum(x[i]^2 for i in 1:2) >= 0)
    @NLobjective(model, Min, x[1]^2 + x[2]^2)
    optimize!(model)
    @test isapprox(JuMP.value(x[1]), JuMP.value(y))
end

@testset "JuMP direct mode NLP models" begin
    # Requires https://github.com/jump-dev/Ipopt.jl/issues/280
    #@test MOI.supports(POI.ParametricOptimizer(Ipopt.Optimizer()), MOI.NLPBlock())
    #optimizer = POI.ParametricOptimizer(Ipopt.Optimizer())
    #model = direct_model(optimizer)
    #@variable(model, x[i=1:2] >= 0)
    #@variable(model, y in POI.Parameter(10))
    #@constraint(model, sum(x) + y >= 0)
    #@NLobjective(model, Min, x[1]^2 + x[2]^2)
    #optimize!(model)
end
