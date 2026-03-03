# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function MOI.set(
    model::Optimizer{T},
    ::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
    f::MOI.ScalarNonlinearFunction,
) where {T}
    # 1. Attempt to parse as cubic
    parsed = _parse_cubic_expression(f, T)
    if parsed === nothing
        error(
            "ScalarNonlinearFunction must be a valid cubic polynomial with " *
            "parameters multiplying at most quadratic variable terms. " *
            "Non-polynomial operations or degree > 3 are not supported.",
        )
    end

    # 2. Create ParametricCubicFunction
    cubic_func = ParametricCubicFunction(parsed)

    # 3. Compute current function for inner optimizer
    current = _current_function(cubic_func, model)

    # 4. Set current function on inner optimizer
    try
        MOI.set(
            model.optimizer,
            MOI.ObjectiveFunction{typeof(current)}(),
            current,
        )
    catch e
        # rethrow the original error with the additional info of the objective function that caused it
        error(
            "Failed to set cubic objective function, f = $f, on inner " *
            "optimizer. " *
            "This may be due to unsupported features in the cubic " *
            "expression. " *
            "Original error: $(e.msg)",
        )
    end

    # 5. Clear old caches
    _empty_objective_function_caches!(model)

    # 6. Cache multiplicative parameters
    _cache_multiplicative_params!(model, cubic_func)

    # 7. Store new cache
    model.cubic_objective_cache = cubic_func

    # 8. Store original for retrieval if option is enabled
    if model.save_original_objective_and_constraints
        MOI.set(
            model.original_objective_cache,
            MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}(),
            f,
        )
    end

    return nothing
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
)
    if model.cubic_objective_cache === nothing
        error("No ScalarNonlinearFunction objective is set")
    end
    if !model.save_original_objective_and_constraints
        error(
            "Cannot retrieve original objective: save_original_objective_and_constraints is false",
        )
    end
    return MOI.get(model.original_objective_cache, attr)
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
)
    return true
end

"""
    _update_cubic_objective!(model::Optimizer{T}) where {T}

Update the cubic objective after parameters have changed.
Uses incremental modifications (ScalarQuadraticCoefficientChange, ScalarCoefficientChange,
ScalarConstantChange) for efficiency when the solver supports them.
Falls back to rebuilding the full objective if incremental modifications are not supported.
"""
function _update_cubic_objective!(model::Optimizer{T}) where {T}
    if model.cubic_objective_cache === nothing
        return
    end
    pf = model.cubic_objective_cache

    # Check if any changes are needed by computing deltas
    delta_constant = _delta_parametric_constant(model, pf)
    delta_affine = _delta_parametric_affine_terms(model, pf)
    delta_quadratic = _delta_parametric_quadratic_terms(model, pf)

    if iszero(delta_constant) &&
       isempty(delta_affine) &&
       isempty(delta_quadratic)
        return  # No changes needed
    end

    _try_incremental_cubic_update!(
        model,
        pf,
        delta_constant,
        delta_affine,
        delta_quadratic,
    )

    return nothing
end

"""
    _try_incremental_cubic_update!(model, pf, delta_constant, delta_affine, delta_quadratic)

Apply incremental coefficient updates to the inner optimizer's objective.
"""
function _try_incremental_cubic_update!(
    model::Optimizer{T},
    pf::ParametricCubicFunction{T},
    delta_constant::T,
    delta_affine::Dict{MOI.VariableIndex,T},
    delta_quadratic::Dict{Tuple{MOI.VariableIndex,MOI.VariableIndex},T},
) where {T}
    # Get the current objective function type from the inner optimizer
    F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())

    # Apply quadratic coefficient changes.
    # For each changed (var1, var2) pair, recompute its new coefficient from the
    # base vv (pf.quadratic_data) data plus current pvv contributions
    # (avoids full copy + full iteration).
    # MOI convention:
    #   - Off-diagonal (v1 != v2): coefficient C means C*v1*v2 (use as-is)
    #   - Diagonal (v1 == v2): coefficient C means (C/2)*v1^2 (multiply by 2)
    for (var1, var2) in keys(delta_quadratic)
        new_coef = get(pf.quadratic_data, (var1, var2), zero(T))
        for term in pf.pvv
            p = term.index_1
            first_is_greater = term.index_2.value > term.index_3.value
            v1 = ifelse(first_is_greater, term.index_3, term.index_2)
            v2 = ifelse(first_is_greater, term.index_2, term.index_3)
            if (v1, v2) == (var1, var2)
                new_coef +=
                    term.coefficient * _effective_param_value(model, p_idx(p))
            end
        end
        moi_coef = new_coef * ifelse(var1 == var2, 2, 1)
        MOI.modify(
            model.optimizer,
            MOI.ObjectiveFunction{F}(),
            MOI.ScalarQuadraticCoefficientChange(var1, var2, moi_coef),
        )
    end

    # Apply affine coefficient changes.
    # For each changed variable, recompute its new coefficient from the base
    # affine_data plus current pv and ppv contributions.
    for var in keys(delta_affine)
        new_coef = get(pf.affine_data, var, zero(T))
        for term in pf.pv
            if term.variable_2 == var
                new_coef +=
                    term.coefficient *
                    _effective_param_value(model, p_idx(term.variable_1))
            end
        end
        for term in pf.ppv
            if term.index_3 == var
                p1_val = _effective_param_value(model, p_idx(term.index_1))
                p2_val = _effective_param_value(model, p_idx(term.index_2))
                new_coef += term.coefficient * p1_val * p2_val
            end
        end
        MOI.modify(
            model.optimizer,
            MOI.ObjectiveFunction{F}(),
            MOI.ScalarCoefficientChange(var, new_coef),
        )
    end

    # Apply constant change using the tracked current_constant (no full recompute).
    if !iszero(delta_constant)
        pf.current_constant += delta_constant
        MOI.modify(
            model.optimizer,
            MOI.ObjectiveFunction{F}(),
            MOI.ScalarConstantChange(pf.current_constant),
        )
    end

    return nothing
end
