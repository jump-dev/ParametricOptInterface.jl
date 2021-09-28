optimizer = () -> POI.Optimizer(GLPK.Optimizer())
const OPTIMIZER = MOI.Bridges.full_bridge_optimizer(optimizer(), Float64)
const CONFIG = MOIT.TestConfig()

@testset "Unit Tests" begin
    MOIT.basic_constraint_tests(OPTIMIZER, CONFIG)
    MOIT.unittest(
        OPTIMIZER,
        CONFIG,
        [
            # FIXME `NumberOfThreads` not supported
            "number_threads",
            # These are excluded because GLPK does not support quadratics.
            "solve_qcp_edge_cases",
            "solve_qp_edge_cases",
            "delete_soc_variables",

            # Tested below because the termination status is different.
            "solve_zero_one_with_bounds_3",

            # TODO(odow): not implemented.
            "number_threads",
        ],
    )
end