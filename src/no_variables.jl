# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    _default_no_variables_feasibility_tolerance(::Type{T})

The default value of the [`NoVariablesFeasibilityTolerance`](@ref) attribute.
"""
_default_no_variables_feasibility_tolerance(::Type{T}) where {T} = zero(T)

function _default_no_variables_feasibility_tolerance(
    ::Type{T},
) where {T<:AbstractFloat}
    return T(1e-6)
end

"""
    _has_no_variables(model::Optimizer)

Return `true` if `model` contains no decision variables. Parameters are not
decision variables, so a model that contains parameters only returns `true`.
"""
function _has_no_variables(model::Optimizer)
    return iszero(MOI.get(model, NumberOfPureVariables()))
end

"""
    _optimize_without_variables!(model::Optimizer)

Solve a model that has no decision variables and cache the result in
`model.no_variables_solution`, so that from then on the model has a solution
that the solution attributes are answered from.

The inner optimizer is never asked to `optimize!`, but `MOI.optimize!` still
flushes pending parameter updates to it beforehand so that its state stays
consistent with POI's caches. Parameter updates are incremental, so skipping
them would corrupt a later solve of a model that has gained decision variables.
"""
function _optimize_without_variables!(model::Optimizer{T}) where {T}
    objective_value = _objective_value_no_variables(model)
    if _is_feasible_no_variables(model)
        model.no_variables_solution = _NoVariablesSolution{T}(
            MOI.OPTIMAL,
            MOI.FEASIBLE_POINT,
            # Every constraint dual is zero, which is a feasible dual solution.
            MOI.FEASIBLE_POINT,
            objective_value,
            "Solved by ParametricOptInterface: the model has no decision variables, so the parameter values are the solution.",
        )
    else
        model.no_variables_solution = _NoVariablesSolution{T}(
            MOI.INFEASIBLE,
            # The parameter values are a point, it just violates a constraint.
            MOI.INFEASIBLE_POINT,
            # There is no certificate of infeasibility to report.
            MOI.NO_SOLUTION,
            objective_value,
            "Solved by ParametricOptInterface: the model has no decision variables and the parameter values violate at least one constraint.",
        )
    end
    return
end

"""
    _no_variables_value(model::Optimizer, attr, solution::_NoVariablesSolution)

Return the value of the solution attribute `attr` of a model that was solved
without decision variables. Dispatched on `attr` from the single `MOI.get`
method for [`_NoVariablesAttribute`](@ref) in `src/MOI_wrapper.jl`.
"""
function _no_variables_value(
    ::Optimizer,
    ::MOI.TerminationStatus,
    solution::_NoVariablesSolution,
)
    return solution.termination_status
end

function _no_variables_value(
    ::Optimizer,
    attr::MOI.PrimalStatus,
    solution::_NoVariablesSolution,
)
    return attr.result_index == 1 ? solution.primal_status : MOI.NO_SOLUTION
end

function _no_variables_value(
    ::Optimizer,
    attr::MOI.DualStatus,
    solution::_NoVariablesSolution,
)
    return attr.result_index == 1 ? solution.dual_status : MOI.NO_SOLUTION
end

function _no_variables_value(
    ::Optimizer,
    ::MOI.ResultCount,
    ::_NoVariablesSolution,
)
    return 1
end

function _no_variables_value(
    ::Optimizer,
    ::MOI.RawStatusString,
    solution::_NoVariablesSolution,
)
    return solution.raw_status_string
end

function _no_variables_value(
    ::Optimizer,
    ::MOI.SolveTimeSec,
    ::_NoVariablesSolution,
)
    return 0.0
end

function _no_variables_value(
    ::Optimizer{T},
    ::MOI.RelativeGap,
    ::_NoVariablesSolution{T},
) where {T}
    # the solution is exact, so there is no gap
    return zero(T)
end

function _no_variables_value(
    ::Optimizer,
    ::MOI.ObjectiveBound,
    solution::_NoVariablesSolution,
)
    return solution.objective_value
end

function _no_variables_value(
    model::Optimizer,
    attr::Union{MOI.ObjectiveValue,MOI.DualObjectiveValue},
    solution::_NoVariablesSolution,
)
    MOI.check_result_index_bounds(model, attr)
    # There are no variables, so the primal and dual objective values coincide
    # with the objective function evaluated at the parameter values.
    return solution.objective_value
end

"""
    _constraint_dual(model::Optimizer, attr::MOI.ConstraintDual, ci)
    _constraint_dual(model::Optimizer, attr::MOI.ConstraintDual, ci, dimension)

Return the dual of the inner constraint `ci`, which is zero (or a vector of
`dimension` zeros) if the model was solved without decision variables.
"""
function _constraint_dual(
    model::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex,
) where {T}
    if model.no_variables_solution !== nothing
        return zero(T)
    end
    return MOI.get(model.optimizer, attr, ci)
end

function _constraint_dual(
    model::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex,
    dimension::Int,
)::Vector{T} where {T}
    if model.no_variables_solution !== nothing
        return zeros(T, dimension)
    end
    return MOI.get(model.optimizer, attr, ci)
end

"""
    _objective_value_no_variables(model::Optimizer{T})::T

Evaluate the objective function of a model with no decision variables at the
current parameter values.
"""
function _objective_value_no_variables(model::Optimizer{T})::T where {T}
    if model.cubic_objective_cache !== nothing
        return _parametric_constant(model, model.cubic_objective_cache)
    elseif model.quadratic_objective_cache !== nothing
        return _parametric_constant(model, model.quadratic_objective_cache)
    elseif model.affine_objective_cache !== nothing
        return _parametric_constant(model, model.affine_objective_cache)
    end
    # The objective contains neither variables nor parameters, so it is a
    # constant.
    F = MOI.get(model.original_objective_cache, MOI.ObjectiveFunctionType())
    f = MOI.get(model.original_objective_cache, MOI.ObjectiveFunction{F}())
    return MOI.Utilities.eval_variables(p -> model.parameters[p_idx(p)]::T, f)
end

"""
    _is_feasible_no_variables(model::Optimizer)

Return `true` if every constraint of a model with no decision variables is
satisfied by the current parameter values, up to
`model.no_variables_feasibility_tolerance`.

Since the constraints contain no decision variables, each one is a statement
about the parameter values alone and can be checked by evaluating it and
measuring the distance to its set.
"""
function _is_feasible_no_variables(model::Optimizer{T}) where {T}
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        if F === MOI.VariableIndex && S <: MOI.Parameter
            # A parameter is always equal to its own value.
            continue
        end
        for ci in MOI.get(model, MOI.ListOfConstraintIndices{F,S}())
            if !_is_satisfied_no_variables(model, ci)
                return false
            end
        end
    end
    return true
end

function _is_satisfied_no_variables(
    model::Optimizer{T},
    ci::MOI.ConstraintIndex,
) where {T}
    f = MOI.get(model, MOI.ConstraintFunction(), ci)
    set = MOI.get(model, MOI.ConstraintSet(), ci)
    value = MOI.Utilities.eval_variables(p -> model.parameters[p_idx(p)]::T, f)
    distance = try
        MOI.Utilities.distance_to_set(value, set)
    catch err
        # `distance_to_set` has no method for this set. Rather than silently
        # declaring the constraint satisfied, point the user at the reason POI
        # needs it.
        if err isa ErrorException
            error(
                "Cannot determine whether the constraint $ci is satisfied " *
                "because `MOI.Utilities.distance_to_set` is not implemented " *
                "for the set type $(typeof(set)). This check is required to " *
                "solve a model that has parameters but no decision " *
                "variables. Set the `SolveWithoutVariables` attribute to " *
                "`false` to defer to the inner optimizer instead, or " *
                "implement `MOI.Utilities.distance_to_set` for " *
                "$(typeof(set)).\nOriginal error: " *
                sprint(showerror, err),
            )
        end
        rethrow(err)
    end
    return distance <= model.no_variables_feasibility_tolerance
end
