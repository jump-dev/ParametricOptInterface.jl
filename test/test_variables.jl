module TestVariables

using Test
import MathOptInterface as MOI
import ParametricOptInterface as POI

function test_add_single_variable()
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    MOI.add_variable(model)
    @test MOI.get(model, MOI.NumberOfVariables()) == 1
end

function test_add_variables_without_parameters()
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    MOI.add_variables(model, 10)
    @test MOI.get(model, MOI.NumberOfVariables()) == 10
    @test MOI.get(model, POI.NumberOfParameters()) == 0
end

function test_add_single_variable_with_parameter()
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    MOI.add_constrained_variable(model, MOI.Parameter(1.0))
    @test MOI.get(model, MOI.NumberOfVariables()) == 0
    @test MOI.get(model, POI.NumberOfParameters()) == 1
end

function test_add_variables_with_parameters()
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    MOI.add_constrained_variable.(model, MOI.Parameter.(ones(10)))
    @test MOI.get(model, MOI.NumberOfVariables()) == 0
    @test MOI.get(model, POI.NumberOfParameters()) == 10
end

function test_add_constrained_variable_without_parameters()
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    MOI.add_constrained_variable(model, MOI.GreaterThan{Float64}(1.0))
    @test MOI.get(model, MOI.NumberOfVariables()) == 1
    @test MOI.get(model, POI.NumberOfParameters()) == 0
end

function test_add_constrained_variables_without_parameters()
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    MOI.add_constrained_variables(model, MOI.GreaterThan{Float64}.(ones(10)))
    @test MOI.get(model, MOI.NumberOfVariables()) == 10
    @test MOI.get(model, POI.NumberOfParameters()) == 0
end

function test_delete_variables()
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    variable_indexes = MOI.add_variables(model, 10)
    @test MOI.get(model, MOI.NumberOfVariables()) == 10
    MOI.delete(model, variable_indexes)
    @test MOI.get(model, MOI.NumberOfVariables()) == 0
    @test MOI.get(model, POI.NumberOfParameters()) == 0
end

function test_delete_parameters()
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    parameter_index, _ = MOI.add_constrained_variable(model, MOI.Parameter(1.0))
    @test MOI.get(model, POI.NumberOfParameters()) == 1
    MOI.delete(model, parameter_index)
    @test MOI.get(model, MOI.NumberOfVariables()) == 0
    @test MOI.get(model, POI.NumberOfParameters()) == 0
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

end

TestVariables.runtests()