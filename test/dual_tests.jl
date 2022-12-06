# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

@testset "Duals: Test Basic" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))

    @variable(model, x[1:2] in POI.Parameter.(ones(2) .* 4.0))
    @variable(model, y[1:6])

    @constraint(model, ctr1, 3 * y[1] >= 2 - 7 * x[1])

    @objective(model, Min, 5 * y[1])

    JuMP.optimize!(model)

    @test 5 / 3 ≈ JuMP.dual(ctr1) atol = 1e-3
    @test [-35 / 3, 0.0] ≈ MOI.get.(model, POI.ParameterDual(), x) atol = 1e-3
    @test [-26 / 3, 0.0, 0.0, 0.0, 0.0, 0.0] ≈ JuMP.value.(y) atol = 1e-3
    @test -130 / 3 ≈ JuMP.objective_value(model) atol = 1e-3
end

@testset "Duals: Test Multiple Parameters" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))

    @variable(model, x[1:6] in POI.Parameter.(ones(6) .* 4.0))
    @variable(model, y[1:6])

    @constraint(model, ctr1, 3 * y[1] >= 2 - 7 * x[3])
    @constraint(model, ctr2, 3 * y[1] >= 2 - 7 * x[3])
    @constraint(model, ctr3, 3 * y[1] >= 2 - 7 * x[3])
    @constraint(model, ctr4, 3 * y[1] >= 2 - 7 * x[3])
    @constraint(model, ctr5, 3 * y[1] >= 2 - 7 * x[3])
    @constraint(model, ctr6, 3 * y[1] >= 2 - 7 * x[3])
    @constraint(model, ctr7, sum(3 * y[i] + x[i] for i in 2:4) >= 2 - 7 * x[3])
    @constraint(
        model,
        ctr8,
        sum(3 * y[i] + 7.0 * x[i] - x[i] for i in 2:4) >= 2 - 7 * x[3]
    )

    @objective(model, Min, 5 * y[1])

    JuMP.optimize!(model)

    @test 5 / 3 ≈
          JuMP.dual(ctr1) +
          JuMP.dual(ctr2) +
          JuMP.dual(ctr3) +
          JuMP.dual(ctr4) +
          JuMP.dual(ctr5) +
          JuMP.dual(ctr6) atol = 1e-3
    @test 0.0 ≈ JuMP.dual(ctr7) atol = 1e-3
    @test 0.0 ≈ JuMP.dual(ctr8) atol = 1e-3
    @test [0.0, 0.0, -35 / 3, 0.0, 0.0, 0.0] ≈
          MOI.get.(model, POI.ParameterDual(), x) atol = 1e-3
    @test [-26 / 3, 0.0, 0.0, 0.0, 0.0, 0.0] ≈ JuMP.value.(y) atol = 1e-3
    @test -130 / 3 ≈ JuMP.objective_value(model) atol = 1e-3
end

@testset "Duals: Test LessThan" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α in POI.Parameter(-1.0))
    @variable(model, x)
    cref = @constraint(model, x ≤ α)
    @objective(model, Max, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -1.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0

    MOI.set(model, POI.ParameterValue(), α, 2.0)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 2.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
end

@testset "Duals: Test EqualTo" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α in POI.Parameter(-1.0))
    @variable(model, x)
    cref = @constraint(model, x == α)
    @objective(model, Max, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -1.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0

    MOI.set(model, POI.ParameterValue(), α, 2.0)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 2.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
end

@testset "Duals: Test GreaterThan" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α in POI.Parameter(1.0))
    MOI.set(model, POI.ParameterValue(), α, -1.0)
    @variable(model, x)
    cref = @constraint(model, x >= α)
    @objective(model, Min, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -1.0
    @test JuMP.dual(cref) == 1.0
    @test MOI.get(model, POI.ParameterDual(), α) == 1.0

    MOI.set(model, POI.ParameterValue(), α, 2.0)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 2.0
    @test JuMP.dual(cref) == 1.0
    @test MOI.get(model, POI.ParameterDual(), α) == 1.0
end

@testset "Duals: Test Multiple Parameters 2" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α[1:10] in POI.Parameter.(ones(10)))
    @variable(model, x)
    cref = @constraint(model, x == sum(2 * α[i] for i in 1:10))
    @objective(model, Min, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 20.0
    @test JuMP.dual(cref) == 1.0
    @test MOI.get(model, POI.ParameterDual(), α[3]) == 2.0
end

@testset "Duals: Test Mixing Params and vars 1" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α[1:5] in POI.Parameter.(ones(5)))
    @variable(model, x)
    cref = @constraint(model, sum(x for i in 1:5) == sum(2 * α[i] for i in 1:5))
    @objective(model, Min, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 2.0
    @test JuMP.dual(cref) == 1 / 5
    @test MOI.get(model, POI.ParameterDual(), α[3]) == 2 / 5
end

@testset "Duals: Test mixing Params and vars 2" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α[1:5] in POI.Parameter.(ones(5)))
    @variable(model, x)
    cref = @constraint(model, 0.0 == sum(-x + 2 * α[i] for i in 1:5))
    @objective(model, Min, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 2.0
    @test JuMP.dual(cref) == 1 / 5
    @test MOI.get(model, POI.ParameterDual(), α[3]) == 2 / 5
end

@testset "Duals: Test mixing Params and vars 3" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α[1:5] in POI.Parameter.(ones(5)))
    @variable(model, x)
    cref = @constraint(model, 0.0 == sum(-x + 2.0 + 2 * α[i] for i in 1:5))
    @objective(model, Min, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == 4.0
    @test JuMP.dual(cref) == 1 / 5
    @test MOI.get(model, POI.ParameterDual(), α[3]) == 2 / 5
end

@testset "Duals: Test add after solve" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α in POI.Parameter(1.0))
    MOI.set(model, POI.ParameterValue(), α, -1.0)
    @variable(model, x)
    cref = @constraint(model, x <= α)
    @objective(model, Max, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -1.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0

    @variable(model, b in POI.Parameter(-2.0))
    cref = @constraint(model, x <= b)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -2.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == 0.0
    @test MOI.get(model, POI.ParameterDual(), b) == -1.0
end

@testset "Duals: Test add ctr alaternative" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α in POI.Parameter(-1.0))
    @variable(model, x)
    exp = x - α
    cref = @constraint(model, exp ≤ 0)
    @objective(model, Max, x)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -1.0
    @test JuMP.dual(cref) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0
end

@testset "Duals: Test deletion constraint" begin
    model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
    @variable(model, α in POI.Parameter(-1.0))
    @variable(model, x)
    cref1 = @constraint(model, x ≤ α / 2)
    cref2 = @constraint(model, x ≤ α)
    cref3 = @constraint(model, x ≤ 2α)
    @objective(model, Max, x)
    JuMP.delete(model, cref3)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -1.0
    @test JuMP.dual(cref1) == 0.0
    @test JuMP.dual(cref2) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -1.0

    JuMP.delete(model, cref2)
    JuMP.optimize!(model)
    @test JuMP.value(x) == -0.5
    @test JuMP.dual(cref1) == -1.0
    @test MOI.get(model, POI.ParameterDual(), α) == -0.5
end
