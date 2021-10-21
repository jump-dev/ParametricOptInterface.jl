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
const CONFIG = MOIT.TestConfig()

@testset "GLPK tests" begin
    MOIT.basic_constraint_tests(
        OPTIMIZER_GLPK,
        CONFIG,
        exclude = [
            # GLPK does not support QuadraticFunctions
            (
                MathOptInterface.VectorQuadraticFunction{Float64},
                MathOptInterface.NormOneCone,
            ),
            (
                MathOptInterface.VectorQuadraticFunction{Float64},
                MathOptInterface.NormInfinityCone,
            ),
            (
                MathOptInterface.VectorQuadraticFunction{Float64},
                MathOptInterface.Zeros,
            ),
            (
                MathOptInterface.VectorQuadraticFunction{Float64},
                MathOptInterface.Nonpositives,
            ),
            (
                MathOptInterface.VectorQuadraticFunction{Float64},
                MathOptInterface.Nonnegatives,
            ),
            (
                MathOptInterface.ScalarQuadraticFunction{Float64},
                MathOptInterface.GreaterThan{Float64},
            ),
            (
                MathOptInterface.ScalarQuadraticFunction{Float64},
                MathOptInterface.LessThan{Float64},
            ),
            (
                MathOptInterface.ScalarQuadraticFunction{Float64},
                MathOptInterface.EqualTo{Float64},
            ),
            (
                MathOptInterface.ScalarQuadraticFunction{Float64},
                MathOptInterface.Interval{Float64},
            ),
        ],
    )

    MOIT.unittest(
        OPTIMIZER_GLPK,
        CONFIG,
        [
            # FIXME `NumberOfThreads` not supported by GLPK
            "number_threads",
            # These are excluded because GLPK does not support quadratics.
            "solve_qcp_edge_cases",
            "solve_qp_edge_cases",
            "solve_qp_zero_offdiag",
            "delete_soc_variables",

            # Tested below because the termination status is different.
            "solve_zero_one_with_bounds_3",
        ],
    )

    MOIT.modificationtest(OPTIMIZER_GLPK, CONFIG)

    @testset "ModelLike tests" begin
        @test MOI.get(OPTIMIZER_GLPK, MOI.SolverName()) ==
              "Parametric Optimizer with GLPK attached"
        @testset "default_objective_test" begin
            MOIT.default_objective_test(OPTIMIZER_GLPK)
        end
        @testset "default_status_test" begin
            MOIT.default_status_test(OPTIMIZER_GLPK)
        end
        @testset "nametest" begin
            MOIT.nametest(OPTIMIZER_GLPK)
        end
        @testset "validtest" begin
            MOIT.validtest(OPTIMIZER_GLPK)
        end
        @testset "emptytest" begin
            MOIT.emptytest(OPTIMIZER_GLPK)
        end
        @testset "orderedindicestest" begin
            MOIT.orderedindicestest(OPTIMIZER_GLPK)
        end
        @testset "copytest" begin
            MOIT.copytest(
                OPTIMIZER_GLPK,
                MOI.Bridges.full_bridge_optimizer(GLPK.Optimizer(), Float64),
            )
        end
        @testset "scalar_function_constant_not_zero" begin
            MOIT.scalar_function_constant_not_zero(OPTIMIZER_GLPK)
        end
    end
end

@testset "Ipopt tests" begin
    MOIT.basic_constraint_tests(OPTIMIZER_CACHED_IPOPT, CONFIG)
    MOIT.unittest(
        OPTIMIZER_CACHED_IPOPT,
        CONFIG,
        [
            # FIXME `NumberOfThreads` not supported by Ipopt
            "number_threads",
            # Tests excluded because the termination status is different.
            # Ipopt returns LOCALLY_SOLVED instead of OPTIMAL
            "solve_time",
            "raw_status_string",
            "solve_affine_greaterthan",
            "solve_affine_interval",
            "solve_affine_lessthan",
            "solve_affine_equalto",
            "solve_farkas_interval_lower",
            "solve_farkas_greaterthan",
            "solve_farkas_variable_lessthan_max",
            "solve_farkas_interval_upper",
            "solve_farkas_variable_lessthan",
            "solve_farkas_lessthan",
            "solve_farkas_equalto_lower",
            "solve_farkas_equalto_upper",
            "solve_with_lowerbound",
            "solve_with_upperbound",
            "solve_duplicate_terms_scalar_affine",
            "solve_duplicate_terms_vector_affine",
            "solve_blank_obj",
            "solve_constant_obj",
            "solve_singlevariable_obj",
            "solve_duplicate_terms_obj",
            "solve_unbounded_model",
            "solve_single_variable_dual_min",
            "solve_single_variable_dual_max",
            "solve_qp_zero_offdiag",
            "solve_zero_one_with_bounds_1",
            "solve_zero_one_with_bounds_2",
            "solve_zero_one_with_bounds_3",
            "solve_twice",
            "solve_result_index",
            "solve_integer_edge_cases",
            "solve_objbound_edge_cases",
            "solve_qp_edge_cases",
            "solve_objbound_edge_cases",
            "solve_affine_deletion_edge_cases",
            "solve_qcp_edge_cases",
        ],
    )
end
