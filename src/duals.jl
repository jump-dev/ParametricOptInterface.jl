# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    _compute_dual_of_parameters!(model::Optimizer)

Populate `model.dual_value_of_parameters` with the sensitivity of the optimal
objective with respect to each parameter.

The dual of parameter `p` is computed as `∂obj*/∂p`, accumulated from:
- Each constraint containing `p`: `dual_λ * ∂f/∂p` (negated, since the
  parameter appears in the constraint's RHS shift)
- The parametric objective: `±∂obj/∂p` (sign depends on `MIN_SENSE`)

For `pp` quadratic terms the product rule gives two symmetric contributions;
diagonal terms (`p_i == p_j`) are halved to avoid double-counting.
"""
function _compute_dual_of_parameters!(model::Optimizer{T}) where {T}
    n = model.number_of_parameters_in_model
    if length(model.dual_value_of_parameters) != n
        model.dual_value_of_parameters = zeros(T, n)
    else
        fill!(model.dual_value_of_parameters, zero(T))
    end
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
    if model.cubic_objective_cache !== nothing
        _update_duals_from_objective!(model, model.cubic_objective_cache)
    end
    return
end

"""
    _update_duals_from_affine_constraints!(model::Optimizer)

Iterate over all scalar affine constraint types and accumulate parameter dual
contributions from each into `model.dual_value_of_parameters`.
The inner-dict call is a type-instability barrier so Julia specializes
`_compute_parameters_in_ci!` on the concrete `(F, S)` types.
"""
function _update_duals_from_affine_constraints!(model::Optimizer)
    for (F, S) in keys(model.affine_constraint_cache.dict)
        affine_constraint_cache_inner = model.affine_constraint_cache[F, S]
        # barrier for type instability
        _compute_parameters_in_ci!(model, affine_constraint_cache_inner)
    end
    return
end

"""
    _update_duals_from_vector_affine_constraints!(model::Optimizer)

Iterate over all vector affine constraint types and accumulate parameter dual
contributions from each into `model.dual_value_of_parameters`.
"""
function _update_duals_from_vector_affine_constraints!(model::Optimizer)
    for (F, S) in keys(model.vector_affine_constraint_cache.dict)
        vector_affine_constraint_cache_inner =
            model.vector_affine_constraint_cache[F, S]
        # barrier for type instability
        _compute_parameters_in_ci!(model, vector_affine_constraint_cache_inner)
    end
    return
end

"""
    _update_duals_from_quadratic_constraints!(model::Optimizer)

Iterate over all scalar quadratic constraint types and accumulate parameter
dual contributions from each into `model.dual_value_of_parameters`.
"""
function _update_duals_from_quadratic_constraints!(model::Optimizer)
    for (F, S) in keys(model.quadratic_constraint_cache.dict)
        quadratic_constraint_cache_inner =
            model.quadratic_constraint_cache[F, S]
        # barrier for type instability
        _compute_parameters_in_ci!(model, quadratic_constraint_cache_inner)
    end
    return
end

"""
    _compute_parameters_in_ci!(model, constraint_cache_inner)
    _compute_parameters_in_ci!(model, pf, ci)

Accumulate the dual contribution of parameters appearing in constraint `ci`
into `model.dual_value_of_parameters`.

For affine terms the contribution is `-λ * c` where `λ` is the constraint
dual and `c` is the parameter coefficient. For `pp` quadratic terms the product
rule applies: each parameter receives `-λ * c * value_of_other_parameter`;
diagonal terms (`p_i == p_j`) are halved to avoid double-counting the
symmetric representation.

The `DoubleDictInner` overload is a function barrier; the concrete `pf`
overloads are where the computation happens.
"""
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
    for term in affine_parameter_terms(pf)
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
    for term in affine_parameter_terms(pf)
        model.dual_value_of_parameters[p_val(term.variable)] -=
            cons_dual * term.coefficient
    end
    for term in quadratic_parameter_parameter_terms(pf)
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
    for term in vector_affine_parameter_terms(pf)
        model.dual_value_of_parameters[p_val(term.scalar_term.variable)] -=
            cons_dual[term.output_index] * term.scalar_term.coefficient
    end
    return
end

"""
    _update_duals_from_objective!(model, pf)

Accumulate the sensitivity of the parametric objective with respect to each
parameter into `model.dual_value_of_parameters`.

The sign convention matches the objective sense: `+` for minimization, `-` for
maximization. Specialized methods exist for `ParametricQuadraticFunction` and
`ParametricCubicFunction` to handle higher-order terms via the product rule.
"""
function _update_duals_from_objective!(model::Optimizer{T}, pf) where {T}
    is_min = MOI.get(model.optimizer, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    for param in affine_parameter_terms(pf)
        model.dual_value_of_parameters[p_val(param.variable)] +=
            ifelse(is_min, 1, -1) * param.coefficient
    end
    return
end

function _update_duals_from_objective!(
    model::Optimizer{T},
    pf::ParametricQuadraticFunction{T},
) where {T}
    is_min = MOI.get(model.optimizer, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    sign = ifelse(is_min, one(T), -one(T))
    # p terms: ∂(c·p_i)/∂p_i = c
    for term in affine_parameter_terms(pf)
        model.dual_value_of_parameters[p_val(term.variable)] +=
            sign * term.coefficient
    end
    # pp terms: ∂(c·p_i·p_j)/∂p_i = c·p_j
    for term in quadratic_parameter_parameter_terms(pf)
        mult = sign * term.coefficient
        if term.variable_1 == term.variable_2
            mult /= 2
        end
        model.dual_value_of_parameters[p_val(term.variable_1)] +=
            mult * model.parameters[p_idx(term.variable_2)]
        model.dual_value_of_parameters[p_val(term.variable_2)] +=
            mult * model.parameters[p_idx(term.variable_1)]
    end
    return
end

function _update_duals_from_objective!(
    model::Optimizer{T},
    pf::ParametricCubicFunction{T},
) where {T}
    is_min = MOI.get(model.optimizer, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    sign = ifelse(is_min, one(T), -one(T))
    # p terms: ∂(c·p_i)/∂p_i = c
    for term in cubic_affine_parameter_terms(pf)
        model.dual_value_of_parameters[p_val(term.variable)] +=
            sign * term.coefficient
    end
    # pp terms: ∂(c·p_i·p_j)/∂p_i = c·p_j (diagonal: c/2·2·p_i = c·p_i)
    for term in cubic_parameter_parameter_terms(pf)
        mult = sign * term.coefficient
        if term.variable_1 == term.variable_2
            mult /= 2
        end
        model.dual_value_of_parameters[p_val(term.variable_1)] +=
            mult * model.parameters[p_idx(term.variable_2)]
        model.dual_value_of_parameters[p_val(term.variable_2)] +=
            mult * model.parameters[p_idx(term.variable_1)]
    end
    # ppp terms: ∂(c·p_i·p_j·p_k)/∂p_i = c·p_j·p_k
    for term in cubic_parameter_parameter_parameter_terms(pf)
        coef = sign * term.coefficient
        p1_val = model.parameters[p_idx(term.index_1)]
        p2_val = model.parameters[p_idx(term.index_2)]
        p3_val = model.parameters[p_idx(term.index_3)]
        model.dual_value_of_parameters[p_val(term.index_1)] +=
            coef * p2_val * p3_val
        model.dual_value_of_parameters[p_val(term.index_2)] +=
            coef * p1_val * p3_val
        model.dual_value_of_parameters[p_val(term.index_3)] +=
            coef * p1_val * p2_val
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
        error("Cannot compute the dual of a multiplicative parameter")
    end
    return model.dual_value_of_parameters[p_val(cp)]
end

"""
    _is_additive(model, cp)

Return `true` if parameter `cp` appears only in additive (affine/constant)
positions. Returns `false` if it appears in any `p*v` product term, in which
case its dual is not well-defined and `ConstraintDual` will error.
"""
function _is_additive(model::Optimizer, cp::MOI.ConstraintIndex)
    if cp.value in model.multiplicative_parameters_pv
        return false
    end
    return true
end

"""
    _update_duals_from_vector_quadratic_constraints!(model::Optimizer)

Iterate over all vector quadratic constraint types and accumulate parameter
dual contributions from each into `model.dual_value_of_parameters`.
"""
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
    for term in vector_affine_parameter_terms(pf)
        model.dual_value_of_parameters[p_val(term.scalar_term.variable)] -=
            cons_dual[term.output_index] * term.scalar_term.coefficient
    end
    for t in vector_quadratic_parameter_parameter_terms(pf)
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
