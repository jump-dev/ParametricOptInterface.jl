optimizer_glpk = () -> POI.Optimizer(GLPK.Optimizer())
optimizer_ipopt = () -> POI.Optimizer(Ipopt.Optimizer())
const OPTIMIZER_GLPK =
    MOI.Bridges.full_bridge_optimizer(optimizer_glpk(), Float64)
const OPTIMIZER_CACHED_IPOPT = MOI.Utilities.CachingOptimizer(
    MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
    MOI.Bridges.full_bridge_optimizer(optimizer_ipopt(), Float64),
)
const CONFIG = MOIT.Config()

@testset "GLPK tests" begin
    # TODO see why tests error or fail
    MOIT.runtests(
        OPTIMIZER_GLPK,
        CONFIG;
        exclude = [
            # GLPK returns INVALID_MODEL instead of INFEASIBLE
            "test_constraint_ZeroOne_bounds_3",
            # Upstream issue: https://github.com/jump-dev/MathOptInterface.jl/issues/1431
            "test_model_LowerBoundAlreadySet",
            "test_model_UpperBoundAlreadySet",
            # Needs a proper ListOfConstraintAttributesSet to work
            "test_model_ListOfConstraintAttributesSet",
            "test_model_ModelFilter_ListOfConstraintIndices",
            "test_model_ModelFilter_ListOfConstraintTypesPresent", 
        ],
    )
end

@testset "Ipopt tests" begin
    MOI.set(OPTIMIZER_CACHED_IPOPT, MOI.Silent(), true)
    # Without fixed_variable_treatment set, duals are not computed for variables
    # that have lower_bound == upper_bound.
    MOI.set(
        OPTIMIZER_CACHED_IPOPT,
        MOI.RawOptimizerAttribute("fixed_variable_treatment"),
        "make_constraint",
    )
    MOIT.runtests(
        OPTIMIZER_CACHED_IPOPT,
        MOIT.Config(
            atol = 1e-4,
            rtol = 1e-4,
            optimal_status = MOI.LOCALLY_SOLVED,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.DualObjectiveValue,
                MOI.ObjectiveBound,
            ],
        );
        exclude = String[
            # Tests purposefully excluded:
            #  - Upstream: ZeroBridge does not support ConstraintDual
            "test_conic_linear_VectorOfVariables_2",
            #  - Excluded because this test is optional
            "test_model_ScalarFunctionConstantNotZero",
            #  - Excluded because Ipopt returns NORM_LIMIT instead of
            #    DUAL_INFEASIBLE
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            #  - Excluded because Ipopt returns INVALID_MODEL instead of
            #    LOCALLY_SOLVED
            "test_linear_VectorAffineFunction_empty_row",
            #  - Excluded because Ipopt returns LOCALLY_INFEASIBLE instead of
            #    INFEASIBLE
            "INFEASIBLE",
            "test_conic_linear_INFEASIBLE",
            "test_conic_linear_INFEASIBLE_2",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_",
            #  - Excluded due to upstream issue
            "test_model_LowerBoundAlreadySet",
            "test_model_UpperBoundAlreadySet",
            #  - CachingOptimizer does not throw if optimizer not attached
            "test_model_copy_to_UnsupportedAttribute",
            "test_model_copy_to_UnsupportedConstraint",
        ],
    )
end
