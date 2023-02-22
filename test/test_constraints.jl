module TestConstraints

using Test
import MathOptInterface as MOI
import ParametricOptInterface as POI

function test_add_variable_index_in_greater_than_constraint()
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan{Float64}(1.0))
    F_in_S = MOI.get(POI.inner_optimizer(model), MOI.ListOfConstraintTypesPresent())
    @test F_in_S == [(MOI.VariableIndex, MOI.GreaterThan{Float64})]
end

function test_error_add_parameter_in_greater_than_constraint()
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    p, _ = MOI.add_constrained_variable(model, MOI.Parameter(1.0))
    @test_throws ErrorException MOI.add_constraint(model, p, MOI.GreaterThan{Float64}(1.0))
end

function test_add_scalar_affine_function_without_parameters()
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, 10)
    func1 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(10), x), 1.0)
    MOI.add_constraint(model, func1, MOI.LessThan{Float64}(3.0))
    F_in_S = MOI.get(POI.inner_optimizer(model), MOI.ListOfConstraintTypesPresent())
    @test F_in_S == [(MOI.ScalarAffineFunction, MOI.LessThan{Float64})]
end

function test_add_scalar_affine_function_with_parameters()
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, 5)
    p1, _ = MOI.add_constrained_variable(model, MOI.Parameter(1.0))
    p2, _ = MOI.add_constrained_variable(model, MOI.Parameter(1.0))
    func1 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([ones(5); 0.5; 2.0], [x; p1; p2]), 0.0)
    MOI.add_constraint(model, func1, MOI.LessThan{Float64}(3.0))
    F_in_S = MOI.get(POI.inner_optimizer(model), MOI.ListOfConstraintTypesPresent())
    @test F_in_S == [(MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64})]
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

TestConstraints.runtests()