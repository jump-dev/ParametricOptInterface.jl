optimizer_glpk = () -> POI.Optimizer(GLPK.Optimizer())
optimizer_ipopt = () -> POI.Optimizer(Ipopt.Optimizer())
const OPTIMIZER_GLPK =
    MOI.Bridges.full_bridge_optimizer(optimizer_glpk(), Float64)
const OPTIMIZER_CACHED_IPOPT = MOI.Bridges.full_bridge_optimizer(
    MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        optimizer_ipopt(),
    ),
    Float64,
)
const CONFIG = MOIT.Config()

@testset "GLPK tests" begin
    # TODO see why tests error or fail
    MOIT.runtests(
        OPTIMIZER_GLPK,
        CONFIG;
        exclude = [
            "test_attribute_SolverVersion",
            "test_linear_Interval_inactive",
            "test_linear_add_constraints",
            "test_linear_inactive_bounds",
            "test_linear_integration_Interval",
            "test_linear_integration_delete_variables",
            "test_model_ListOfConstraintAttributesSet",
            "test_model_ModelFilter_ListOfConstraintIndices",
            "test_model_ModelFilter_ListOfConstraintTypesPresent", # Needs ListOfConstraintAttributesSet to work
            "test_constraint_ZeroOne_bounds_3",
            "test_linear_integration_2",
        ],
    )
end

@testset "ListOfConstraintTypesPresent" begin
    N = 10
    model = OPTIMIZER_GLPK
    x = MOI.add_variables(model, N/2)
    y = first.(MOI.add_constrained_variable.(model, POI.Parameter.(ones(Int(N/2)))))

    MOI.add_constraint(
                    model,
                    MOI.ScalarQuadraticFunction(
                        MOI.ScalarQuadraticTerm.(1.0, x, y),
                        MOI.ScalarAffineTerm{Float64}[],
                        0.0,
                    ),
                    MOI.GreaterThan(1.0),
                )
                
    list_ctrs_types = MOI.get(model, MOI.ListOfConstraintTypesPresent())
    @test list_ctrs_types == [((MathOptInterface.ScalarQuadraticFunction{Float64}, MathOptInterface.GreaterThan{Float64}))]
end

# @testset "Ipopt tests" begin
#     MOIT.runtests(
#         OPTIMIZER_CACHED_IPOPT,
#         CONFIG; 
#         exclude = [
#             "test_attribute_RawStatusString",
#             "test_attribute_SolveTimeSec",
#             "test_attribute_SolverVersion",
#             "test_conic_NormInfinityCone_VectorOfVariables",
#             "test_conic_NormOneCone",
#             "test_conic_linear",
#             "test_infeasible",
#             "test_variable_solve_with",
#             "test_solve_optimize_twice",
#             "test_unbounded",
#             "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE"
#         ]
#     )
# end
