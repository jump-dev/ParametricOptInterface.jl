# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function _compute_dual_of_parameters!(model::Optimizer{T}) where {T}
    model.dual_value_of_parameters =
        zeros(T, model.number_of_parameters_in_model)
    _update_duals_from_affine_constraints!(model)
    _update_duals_from_vector_affine_constraints!(model)
    _update_duals_from_quadratic_constraints!(model)
    _update_duals_from_vector_quadratic_constraints!(model)
    if model.affine_objective_cache !== nothing
        _update_duals_from_objective!(model, model.affine_objective_cache)
    end
    if model.quadratic_objective_cache !== nothing
        _update_duals_from_objective!(model, model.quadratic_objective_cache)
    end
    return
end

function _update_duals_from_affine_constraints!(model::Optimizer)
    for (F, S) in keys(model.affine_constraint_cache.dict)
        affine_constraint_cache_inner = model.affine_constraint_cache[F, S]
        # barrier for type instability
        _compute_parameters_in_ci!(model, affine_constraint_cache_inner)
    end
    return
end

function _update_duals_from_vector_affine_constraints!(model::Optimizer)
    for (F, S) in keys(model.vector_affine_constraint_cache.dict)
        vector_affine_constraint_cache_inner =
            model.vector_affine_constraint_cache[F, S]
        # barrier for type instability
        _compute_parameters_in_ci!(model, vector_affine_constraint_cache_inner)
    end
    return
end

function _update_duals_from_quadratic_constraints!(model::Optimizer)
    for (F, S) in keys(model.quadratic_constraint_cache.dict)
        quadratic_constraint_cache_inner =
            model.quadratic_constraint_cache[F, S]
        # barrier for type instability
        _compute_parameters_in_ci!(model, quadratic_constraint_cache_inner)
    end
    return
end

function _compute_parameters_in_ci!(
    model::Optimizer,
    constraint_cache_inner::DoubleDicts.DoubleDictInner{F,S,V},
) where {F,S,V}
    for (inner_ci, pf) in constraint_cache_inner
        _compute_parameters_in_ci!(model, pf, inner_ci)
    end
    return
end

function _compute_parameters_in_ci!(
    model::Optimizer{T},
    pf::ParametricAffineFunction{T},
    ci::MOI.ConstraintIndex{F,S},
) where {F,S,T}
    cons_dual = MOI.get(model.optimizer, MOI.ConstraintDual(), ci)
    for term in pf.p
        model.dual_value_of_parameters[p_val(term.variable)] -=
            cons_dual * term.coefficient
    end
    return
end

function _compute_parameters_in_ci!(
    model::Optimizer{T},
    pf::ParametricQuadraticFunction{T},
    ci::MOI.ConstraintIndex{F,S},
) where {F,S,T}
    cons_dual = MOI.get(model.optimizer, MOI.ConstraintDual(), ci)
    for term in pf.p
        model.dual_value_of_parameters[p_val(term.variable)] -=
            cons_dual * term.coefficient
    end
    for term in pf.pp
        mult = cons_dual * term.coefficient
        if term.variable_1 == term.variable_2
            mult /= 2
        end
        model.dual_value_of_parameters[p_val(term.variable_1)] -=
            mult * model.parameters[p_idx(term.variable_2)]
        model.dual_value_of_parameters[p_val(term.variable_2)] -=
            mult * model.parameters[p_idx(term.variable_1)]
    end
    return
end

function _compute_parameters_in_ci!(
    model::Optimizer{T},
    pf::ParametricVectorAffineFunction{T},
    ci::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorAffineFunction{T},S} where {T}
    cons_dual = MOI.get(model.optimizer, MOI.ConstraintDual(), ci)
    for term in pf.p
        model.dual_value_of_parameters[p_val(term.scalar_term.variable)] -=
            cons_dual[term.output_index] * term.scalar_term.coefficient
    end
    return
end

function _update_duals_from_objective!(model::Optimizer{T}, pf) where {T}
    is_min = MOI.get(model.optimizer, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    for param in pf.p
        model.dual_value_of_parameters[p_val(param.variable)] +=
            ifelse(is_min, 1, -1) * param.coefficient
    end
    return
end

function MOI.get(
    model::Optimizer{T},
    attr::MOI.ConstraintDual,
    cp::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T}
    if !model.evaluate_duals
        msg = "$attr not available when evaluate_duals is set to false. Create an optimizer such as `POI.Optimizer(HiGHS.Optimizer; evaluate_duals = true)` to enable this feature."
        throw(MOI.GetAttributeNotAllowed(attr, msg))
    elseif !_is_additive(model, cp)
        msg = "Cannot compute the dual of a multiplicative parameter"
        throw(MOI.GetAttributeNotAllowed(attr, msg))
    end
    return model.dual_value_of_parameters[p_val(cp)]
end

function _is_additive(model::Optimizer, cp::MOI.ConstraintIndex)
    if cp.value in model.multiplicative_parameters_pv
        return false
    end
    return true
end

function _update_duals_from_vector_quadratic_constraints!(model::Optimizer)
    for (F, S) in keys(model.vector_quadratic_constraint_cache.dict)
        vector_quadratic_constraint_cache_inner =
            model.vector_quadratic_constraint_cache[F, S]
        _compute_parameters_in_ci!(
            model,
            vector_quadratic_constraint_cache_inner,
        )
    end
    return
end

function _compute_parameters_in_ci!(
    model::Optimizer{T},
    pf::ParametricVectorQuadraticFunction{T},
    ci::MOI.ConstraintIndex{F,S},
) where {F,S,T}
    cons_dual = MOI.get(model.optimizer, MOI.ConstraintDual(), ci)
    for term in pf.p
        model.dual_value_of_parameters[p_val(term.scalar_term.variable)] -=
            cons_dual[term.output_index] * term.scalar_term.coefficient
    end
    for t in pf.pp
        mult = cons_dual[t.output_index] * t.scalar_term.coefficient
        if t.scalar_term.variable_1 == t.scalar_term.variable_2
            mult /= 2
        end
        model.dual_value_of_parameters[p_val(t.scalar_term.variable_1)] -=
            mult * model.parameters[p_idx(t.scalar_term.variable_2)]
        model.dual_value_of_parameters[p_val(t.scalar_term.variable_2)] -=
            mult * model.parameters[p_idx(t.scalar_term.variable_1)]
    end
    return
end
