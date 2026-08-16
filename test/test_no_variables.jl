# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestNoVariablesTests

using Test
using JuMP

import HiGHS
import MathOptInterface as MOI
import ParametricOptInterface as POI

# A set for which `MOI.Utilities.distance_to_set` is not implemented, so the
# feasibility of a model with no decision variables cannot be checked.
struct UncheckableSet <: MOI.AbstractVectorSet
    dimension::Int
end

# A set whose `MOI.Utilities.distance_to_set` fails for a reason other than
# being unimplemented, which POI must not relabel as unimplemented.
struct ThrowingSet <: MOI.AbstractVectorSet
    dimension::Int
end

function MOI.Utilities.distance_to_set(
    ::MOI.Utilities.ProjectionUpperBoundDistance,
    ::AbstractVector,
    ::ThrowingSet,
)
    return throw(DimensionMismatch("some unrelated failure"))
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function _new_optimizer(; kwargs...)
    optimizer = POI.Optimizer(HiGHS.Optimizer(); kwargs...)
    MOI.set(optimizer, MOI.Silent(), true)
    return optimizer
end

function test_affine_objective()
    optimizer = _new_optimizer()
    p, cp = MOI.add_constrained_variable(optimizer, MOI.Parameter(2.0))
    q, cq = MOI.add_constrained_variable(optimizer, MOI.Parameter(3.0))
    f = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([4.0, 5.0], [p, q]), 1.0)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(optimizer, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(optimizer, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(optimizer, MOI.ResultCount()) == 1
    @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 4 * 2 + 5 * 3 + 1
    @test MOI.get(optimizer, MOI.DualObjectiveValue()) ≈ 4 * 2 + 5 * 3 + 1
    @test MOI.get(optimizer, MOI.VariablePrimal(), p) ≈ 2.0
    @test MOI.get(optimizer, MOI.VariablePrimal(), q) ≈ 3.0
    @test MOI.get(optimizer, MOI.ConstraintDual(), cp) ≈ 4.0
    @test MOI.get(optimizer, MOI.ConstraintDual(), cq) ≈ 5.0
    # updating a parameter and re-solving
    MOI.set(optimizer, MOI.ConstraintSet(), cp, MOI.Parameter(7.0))
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 4 * 7 + 5 * 3 + 1
    @test MOI.get(optimizer, MOI.ConstraintDual(), cp) ≈ 4.0
    return
end

function test_max_sense_objective()
    optimizer = _new_optimizer()
    p, cp = MOI.add_constrained_variable(optimizer, MOI.Parameter(2.0))
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(4.0, p)], 0.0)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(optimizer, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 8.0
    @test MOI.get(optimizer, MOI.ConstraintDual(), cp) ≈ -4.0
    return
end

function test_quadratic_objective()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)
    @variable(model, p in Parameter(3.0))
    @variable(model, q in Parameter(5.0))
    @objective(model, Min, p^2 + 2p * q + 4q + 7)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test MOI.get(unsafe_backend(model), POI.ParametricObjectiveType()) ==
          POI.ParametricQuadraticFunction{Float64}
    @test objective_value(model) ≈ 9 + 2 * 15 + 4 * 5 + 7
    # ∂/∂p = 2p + 2q and ∂/∂q = 2p + 4
    @test dual(ParameterRef(p)) ≈ 2 * 3 + 2 * 5
    @test dual(ParameterRef(q)) ≈ 2 * 3 + 4
    # updating a parameter and re-solving
    set_parameter_value(p, 1.0)
    optimize!(model)
    @test objective_value(model) ≈ 1 + 2 * 5 + 4 * 5 + 7
    @test dual(ParameterRef(p)) ≈ 2 * 1 + 2 * 5
    @test dual(ParameterRef(q)) ≈ 2 * 1 + 4
    return
end

function test_quadratic_objective_max_sense()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)
    @variable(model, p in Parameter(3.0))
    @objective(model, Max, p^2)
    optimize!(model)
    @test objective_value(model) ≈ 9.0
    @test dual(ParameterRef(p)) ≈ -2 * 3
    return
end

function test_cubic_objective()
    # a cubic objective reaches POI as a `ScalarNonlinearFunction`, and `p^3`
    # needs no decision variables to be a valid cubic
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)
    @variable(model, p in Parameter(2.0))
    @objective(model, Min, p^3)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test MOI.get(unsafe_backend(model), POI.ParametricObjectiveType()) ==
          POI.ParametricCubicFunction{Float64}
    @test objective_value(model) ≈ 8.0
    # ∂(p³)/∂p = 3p²
    @test dual(ParameterRef(p)) ≈ 3 * 4
    set_parameter_value(p, 3.0)
    optimize!(model)
    @test objective_value(model) ≈ 27.0
    @test dual(ParameterRef(p)) ≈ 3 * 9
    return
end

function test_cubic_objective_distinct_parameters()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)
    @variable(model, p in Parameter(2.0))
    @variable(model, q in Parameter(3.0))
    @variable(model, r in Parameter(4.0))
    @objective(model, Min, p * q * r)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 24.0
    # ∂(pqr)/∂p = qr, and so on
    @test dual(ParameterRef(p)) ≈ 3 * 4
    @test dual(ParameterRef(q)) ≈ 2 * 4
    @test dual(ParameterRef(r)) ≈ 2 * 3
    return
end

function test_cubic_objective_mixed_degrees()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)
    @variable(model, p in Parameter(2.0))
    @variable(model, q in Parameter(3.0))
    # cubic, quadratic and affine parameter terms together
    @objective(model, Min, 2 * p^2 * q + p)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 2 * 4 * 3 + 2
    # ∂/∂p = 4pq + 1 and ∂/∂q = 2p²
    @test dual(ParameterRef(p)) ≈ 4 * 2 * 3 + 1
    @test dual(ParameterRef(q)) ≈ 2 * 4
    return
end

function test_cubic_objective_max_sense()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)
    @variable(model, p in Parameter(2.0))
    @objective(model, Max, p^3)
    optimize!(model)
    @test objective_value(model) ≈ 8.0
    @test dual(ParameterRef(p)) ≈ -3 * 4
    return
end

function test_no_objective()
    optimizer = _new_optimizer()
    p, cp = MOI.add_constrained_variable(optimizer, MOI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 0.0
    @test MOI.get(optimizer, MOI.ConstraintDual(), cp) ≈ 0.0
    return
end

function test_constant_objective()
    optimizer = _new_optimizer()
    p, cp = MOI.add_constrained_variable(optimizer, MOI.Parameter(2.0))
    f = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], 3.0)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(optimizer, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 3.0
    @test MOI.get(optimizer, MOI.ConstraintDual(), cp) ≈ 0.0
    return
end

function test_feasible_scalar_constraint_has_zero_dual()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)
    @variable(model, p in Parameter(2.0))
    @constraint(model, c, 3p <= 12)
    @objective(model, Min, 4p)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 8.0
    @test value(c) ≈ 6.0
    @test dual(c) ≈ 0.0
    # the constraint dual is zero, so only the objective contributes
    @test dual(ParameterRef(p)) ≈ 4.0
    return
end

function test_feasible_vector_constraint_has_zero_dual()
    # HiGHS needs the bridges to handle `Zeros`
    model =
        direct_model(POI.Optimizer(HiGHS.Optimizer; with_bridge_type = Float64))
    set_silent(model)
    @variable(model, p[1:2] in Parameter.([1.0, 2.0]))
    @constraint(model, c, [p[1] - 1, p[2] - 2] in MOI.Zeros(2))
    @objective(model, Min, p[1] + 3p[2])
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 7.0
    @test dual(c) ≈ [0.0, 0.0]
    @test dual(ParameterRef(p[1])) ≈ 1.0
    @test dual(ParameterRef(p[2])) ≈ 3.0
    return
end

function test_feasible_quadratic_constraint_has_zero_dual()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)
    @variable(model, p in Parameter(2.0))
    @variable(model, q in Parameter(3.0))
    # a `pp` constraint, so the dual loop reads a scalar quadratic constraint
    @constraint(model, c, p * q <= 12)
    @objective(model, Min, 4p + 5q)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 4 * 2 + 5 * 3
    @test dual(c) ≈ 0.0
    # `c` contributes nothing, so the duals are the objective coefficients even
    # though `∂(p*q)/∂p = q` is nonzero
    @test dual(ParameterRef(p)) ≈ 4.0
    @test dual(ParameterRef(q)) ≈ 5.0
    return
end

function test_feasible_vector_quadratic_constraint_has_zero_dual()
    # HiGHS needs the bridges to handle `Nonnegatives`
    model =
        direct_model(POI.Optimizer(HiGHS.Optimizer; with_bridge_type = Float64))
    set_silent(model)
    @variable(model, p in Parameter(2.0))
    @variable(model, q in Parameter(3.0))
    # a vector `pp` constraint, so the dual loop reads a vector quadratic
    # constraint and needs a zero vector of the right dimension
    @constraint(model, c, [12 - p * q, 6 - p * p] in MOI.Nonnegatives(2))
    @objective(model, Min, 4p + 5q)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 4 * 2 + 5 * 3
    @test dual(c) ≈ [0.0, 0.0]
    @test dual(ParameterRef(p)) ≈ 4.0
    @test dual(ParameterRef(q)) ≈ 5.0
    return
end

function test_solution_is_cached()
    optimizer = _new_optimizer()
    p, _ = MOI.add_constrained_variable(optimizer, MOI.Parameter(2.0))
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(4.0, p)], 0.0)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(optimizer, MOI.ObjectiveFunction{typeof(f)}(), f)
    @test optimizer.no_variables_solution === nothing
    MOI.optimize!(optimizer)
    solution = optimizer.no_variables_solution
    @test solution isa POI._NoVariablesSolution{Float64}
    @test solution.termination_status == MOI.OPTIMAL
    @test solution.objective_value ≈ 8.0
    # a solve that the inner optimizer handles must clear the cache
    x = MOI.add_variable(optimizer)
    MOI.add_constraint(optimizer, x, MOI.GreaterThan(1.0))
    MOI.optimize!(optimizer)
    @test optimizer.no_variables_solution === nothing
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
    return
end

function test_violated_constraint_is_infeasible()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)
    @variable(model, p in Parameter(20.0))
    @constraint(model, c, p <= 10)
    @objective(model, Min, 4p)
    optimize!(model)
    @test termination_status(model) == MOI.INFEASIBLE
    @test primal_status(model) == MOI.INFEASIBLE_POINT
    @test dual_status(model) == MOI.NO_SOLUTION
    # the parameter values are still available
    @test value(p) ≈ 20.0
    @test objective_value(model) ≈ 80.0
    # the dual of a parameter is the derivative of the objective function, which
    # is well defined even though the parameter values violate `c`
    @test dual(ParameterRef(p)) ≈ 4.0
    @test dual(c) ≈ 0.0
    return
end

function test_solution_attributes()
    optimizer = _new_optimizer()
    p, _ = MOI.add_constrained_variable(optimizer, MOI.Parameter(2.0))
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(4.0, p)], 0.0)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(optimizer, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(optimizer)
    # the solution is exact, so there is no gap and no bound to close
    @test MOI.get(optimizer, MOI.ObjectiveBound()) ≈ 8.0
    @test MOI.get(optimizer, MOI.RelativeGap()) ≈ 0.0
    @test MOI.get(optimizer, MOI.SolveTimeSec()) == 0.0
    @test occursin(
        "ParametricOptInterface",
        MOI.get(optimizer, MOI.RawStatusString()),
    )
    return
end

function test_feasibility_within_tolerance()
    # a violation strictly between zero and the default tolerance, so the test
    # fails if the default becomes exact
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)
    @variable(model, p in Parameter(1e-9))
    @constraint(model, c, p == 0)
    @objective(model, Min, 2p)
    optimize!(model)
    @test 0 <
          1e-9 <
          MOI.get(unsafe_backend(model), POI.NoVariablesFeasibilityTolerance())
    @test termination_status(model) == MOI.OPTIMAL
    # just beyond the default tolerance
    set_parameter_value(p, 1e-3)
    optimize!(model)
    @test termination_status(model) == MOI.INFEASIBLE
    return
end

function test_feasibility_tolerance_is_respected()
    for (tolerance, status) in ((1e-3, MOI.OPTIMAL), (1e-9, MOI.INFEASIBLE))
        optimizer = _new_optimizer()
        MOI.set(optimizer, POI.NoVariablesFeasibilityTolerance(), tolerance)
        p, _ = MOI.add_constrained_variable(optimizer, MOI.Parameter(1e-6))
        f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, p)], 0.0)
        MOI.add_constraint(optimizer, f, MOI.EqualTo(0.0))
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == status
    end
    return
end

function test_attribute_defaults_and_round_trip()
    optimizer = _new_optimizer()
    @test MOI.supports(optimizer, POI.SolveWithoutVariables())
    @test MOI.supports(optimizer, POI.NoVariablesFeasibilityTolerance())
    @test MOI.get(optimizer, POI.SolveWithoutVariables()) == true
    @test MOI.get(optimizer, POI.NoVariablesFeasibilityTolerance()) == 1e-6
    MOI.set(optimizer, POI.SolveWithoutVariables(), false)
    MOI.set(optimizer, POI.NoVariablesFeasibilityTolerance(), 1e-5)
    @test MOI.get(optimizer, POI.SolveWithoutVariables()) == false
    @test MOI.get(optimizer, POI.NoVariablesFeasibilityTolerance()) == 1e-5
    # `MOI.empty!` resets the solution but keeps the configuration
    MOI.empty!(optimizer)
    @test MOI.get(optimizer, POI.SolveWithoutVariables()) == false
    @test MOI.get(optimizer, POI.NoVariablesFeasibilityTolerance()) == 1e-5
    # exact arithmetic needs no slack
    exact = POI.Optimizer{Int}(MOI.Utilities.Model{Int}())
    @test MOI.get(exact, POI.NoVariablesFeasibilityTolerance()) == 0
    return
end

function test_attributes_through_jump()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)
    set_attribute(model, POI.SolveWithoutVariables(), false)
    @test get_attribute(model, POI.SolveWithoutVariables()) == false
    set_attribute(model, POI.NoVariablesFeasibilityTolerance(), 1e-5)
    @test get_attribute(model, POI.NoVariablesFeasibilityTolerance()) == 1e-5
    return
end

function test_solve_without_variables_false()
    optimizer = _new_optimizer()
    MOI.set(optimizer, POI.SolveWithoutVariables(), false)
    p, _ = MOI.add_constrained_variable(optimizer, MOI.Parameter(2.0))
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(4.0, p)], 0.0)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(optimizer, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(optimizer)
    # HiGHS rejects a model with no variables
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.INVALID_MODEL
    @test optimizer.no_variables_solution === nothing
    return
end

function test_evaluate_duals_false()
    optimizer = _new_optimizer(; evaluate_duals = false)
    p, cp = MOI.add_constrained_variable(optimizer, MOI.Parameter(2.0))
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(4.0, p)], 0.0)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(optimizer, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 8.0
    @test_throws(
        MOI.GetAttributeNotAllowed,
        MOI.get(optimizer, MOI.ConstraintDual(), cp),
    )
    return
end

function test_adding_a_variable_after_solving()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)
    @variable(model, p in Parameter(2.0))
    @constraint(model, c1, 3p <= 12)
    @objective(model, Min, 4p)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 8.0
    # the model now gains a decision variable, so the inner optimizer solves it
    # and must still hold data consistent with the parameter values
    @variable(model, x >= 0)
    @constraint(model, c2, x >= 2p)
    @objective(model, Min, 4p + x)
    set_parameter_value(p, 3.0)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test value(x) ≈ 6.0
    @test objective_value(model) ≈ 4 * 3 + 6
    # 4 from the objective and 2 from the dual of `c2`
    @test dual(ParameterRef(p)) ≈ 4 + 2
    return
end

function test_deleting_the_last_variable()
    model = direct_model(POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)
    @variable(model, p in Parameter(2.0))
    @variable(model, x >= 0)
    @objective(model, Min, 4p + x)
    optimize!(model)
    @test objective_value(model) ≈ 8.0
    delete(model, x)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 8.0
    @test dual(ParameterRef(p)) ≈ 4.0
    return
end

function test_result_index_bounds()
    # `MOI.ConstraintDual` must respect `attr.result_index` both when POI
    # answers from its own solution and when the inner optimizer answers
    optimizer = _new_optimizer()
    p, cp = MOI.add_constrained_variable(optimizer, MOI.Parameter(2.0))
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, p)], 0.0)
    c = MOI.add_constraint(optimizer, f, MOI.LessThan(10.0))
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(optimizer, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.ResultCount()) == 1
    @test MOI.get(optimizer, MOI.ConstraintDual(1), c) ≈ 0.0
    @test_throws(
        MOI.ResultIndexBoundsError,
        MOI.get(optimizer, MOI.ConstraintDual(2), c),
    )
    @test_throws(
        MOI.ResultIndexBoundsError,
        MOI.get(optimizer, MOI.ObjectiveValue(2)),
    )
    @test MOI.get(optimizer, MOI.PrimalStatus(2)) == MOI.NO_SOLUTION
    @test MOI.get(optimizer, MOI.DualStatus(2)) == MOI.NO_SOLUTION
    return
end

function test_result_index_bounds_vector()
    inner = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    optimizer = POI.Optimizer(MOI.Utilities.MockOptimizer(inner))
    p, _ = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    f = MOI.VectorAffineFunction(
        [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, p))],
        [0.0, 0.0],
    )
    c = MOI.add_constraint(optimizer, f, MOI.Zeros(2))
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.ConstraintDual(1), c) ≈ [0.0, 0.0]
    @test_throws(
        MOI.ResultIndexBoundsError,
        MOI.get(optimizer, MOI.ConstraintDual(2), c),
    )
    return
end

function test_empty_model()
    optimizer = _new_optimizer()
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 0.0
    MOI.empty!(optimizer)
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
    return
end

function test_unsupported_set_for_feasibility_check()
    # `UniversalFallback` stores the constraint even though `Model` does not
    # support the set, so the feasibility check is what fails.
    inner = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    optimizer = POI.Optimizer(MOI.Utilities.MockOptimizer(inner))
    p, _ = MOI.add_constrained_variable(optimizer, MOI.Parameter(2.0))
    f = MOI.VectorAffineFunction(
        [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, p))],
        [0.0],
    )
    MOI.add_constraint(optimizer, f, UncheckableSet(1))
    err = try
        MOI.optimize!(optimizer)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("distance_to_set", err.msg)
    @test occursin("SolveWithoutVariables", err.msg)
    return
end

function test_unrelated_error_in_feasibility_check_is_rethrown()
    # a failure that is not "the method is missing" must reach the user as
    # itself, not relabelled as an unimplemented `distance_to_set`
    inner = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    optimizer = POI.Optimizer(MOI.Utilities.MockOptimizer(inner))
    p, _ = MOI.add_constrained_variable(optimizer, MOI.Parameter(2.0))
    f = MOI.VectorAffineFunction(
        [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, p))],
        [0.0],
    )
    MOI.add_constraint(optimizer, f, ThrowingSet(1))
    @test_throws DimensionMismatch MOI.optimize!(optimizer)
    return
end

end # module

TestNoVariablesTests.runtests()
