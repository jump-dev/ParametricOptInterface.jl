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

    # 6. Store new cache
    model.cubic_objective_cache = cubic_func

    # 7. Store original for retrieval if option is enabled
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

    # Compute full new values (not deltas) for robustness
    # The delta was used to detect changes; now apply full new coefficients
    new_quad_terms = _parametric_quadratic_terms(model, pf)
    new_affine_terms = _parametric_affine_terms(model, pf)
    new_constant = _parametric_constant(model, pf)

    # Apply quadratic coefficient changes
    # MOI convention:
    #   - Off-diagonal (v1 != v2): coefficient C means C*v1*v2 (use as-is)
    #   - Diagonal (v1 == v2): coefficient C means (C/2)*v1^2 (multiply by 2)
    for ((var1, var2), _) in delta_quadratic
        new_coef = new_quad_terms[(var1, var2)]
        # Apply MOI coefficient convention
        moi_coef = var1 == var2 ? new_coef * 2 : new_coef
        MOI.modify(
            model.optimizer,
            MOI.ObjectiveFunction{F}(),
            MOI.ScalarQuadraticCoefficientChange(var1, var2, moi_coef),
        )
    end

    # Apply affine coefficient changes (use full new coefficient)
    for (var, _) in delta_affine
        new_coef = new_affine_terms[var]
        MOI.modify(
            model.optimizer,
            MOI.ObjectiveFunction{F}(),
            MOI.ScalarCoefficientChange(var, new_coef),
        )
    end

    # Apply constant change
    if !iszero(delta_constant)
        pf.current_constant = new_constant
        MOI.modify(
            model.optimizer,
            MOI.ObjectiveFunction{F}(),
            MOI.ScalarConstantChange(pf.current_constant),
        )
    end

    return nothing
end
