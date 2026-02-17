# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function _is_variable(v::MOI.VariableIndex)
    return !_is_parameter(v)
end

function _is_parameter(v::MOI.VariableIndex)
    return PARAMETER_INDEX_THRESHOLD < v.value <= PARAMETER_INDEX_THRESHOLD_MAX
end

function _has_parameters(f::MOI.ScalarAffineFunction{T}) where {T}
    for term in f.terms
        if _is_parameter(term.variable)
            return true
        end
    end
    return false
end

function _has_parameters(f::MOI.VectorOfVariables)
    for variable in f.variables
        if _is_parameter(variable)
            return true
        end
    end
    return false
end

function _has_parameters(f::MOI.VectorAffineFunction{T}) where {T}
    for term in f.terms
        if _is_parameter(term.scalar_term.variable)
            return true
        end
    end
    return false
end

function _has_parameters(f::MOI.ScalarQuadraticFunction{T}) where {T}
    for term_l in f.affine_terms
        if _is_parameter(term_l.variable)
            return true
        end
    end
    for term in f.quadratic_terms
        if _is_parameter(term.variable_1) || _is_parameter(term.variable_2)
            return true
        end
    end
    return false
end

function _has_parameters(f::MOI.VectorQuadraticFunction)
    # quadratic part
    for qt in f.quadratic_terms
        if _is_parameter(qt.scalar_term.variable_1) ||
           _is_parameter(qt.scalar_term.variable_2)
            return true
        end
    end
    # affine part
    for at in f.affine_terms
        if _is_parameter(at.scalar_term.variable)
            return true
        end
    end
    return false
end

function _cache_multiplicative_params!(
    model::Optimizer{T},
    f::ParametricQuadraticFunction{T},
) where {T}
    for term in f.pv
        push!(model.multiplicative_parameters_pv, term.variable_1.value)
    end
    for term in f.pp
        push!(model.multiplicative_parameters_pp, term.variable_1.value)
        push!(model.multiplicative_parameters_pp, term.variable_2.value)
    end
    return
end

function _cache_multiplicative_params!(
    model::Optimizer{T},
    f::ParametricVectorQuadraticFunction{T},
) where {T}
    for term in f.pv
        push!(
            model.multiplicative_parameters_pv,
            term.scalar_term.variable_1.value,
        )
    end
    for term in f.pp
        push!(
            model.multiplicative_parameters_pp,
            term.scalar_term.variable_1.value,
        )
        push!(
            model.multiplicative_parameters_pp,
            term.scalar_term.variable_2.value,
        )
    end
    return
end

#
# Empty
#

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.optimizer) &&
           isempty(model.parameters) &&
           isempty(model.parameters_name) &&
           isempty(model.updated_parameters) &&
           model.last_parameter_index_added == PARAMETER_INDEX_THRESHOLD &&
           isempty(model.constraint_outer_to_inner) &&
           # affine ctr
           model.last_affine_added == 0 &&
           isempty(model.affine_outer_to_inner) &&
           isempty(model.affine_constraint_cache) &&
           isempty(model.affine_constraint_cache_set) &&
           # quad ctr
           model.last_quad_add_added == 0 &&
           model.last_vec_quad_add_added == 0 &&
           isempty(model.quadratic_outer_to_inner) &&
           isempty(model.vector_quadratic_outer_to_inner) &&
           isempty(model.quadratic_constraint_cache) &&
           isempty(model.quadratic_constraint_cache_set) &&
           isempty(model.vector_quadratic_constraint_cache) &&
           isempty(model.vector_quadratic_constraint_cache_set) &&
           # obj
           model.affine_objective_cache === nothing &&
           model.quadratic_objective_cache === nothing &&
           MOI.is_empty(model.original_objective_cache) &&
           isempty(model.quadratic_objective_cache_product) &&
           #
           isempty(model.vector_affine_constraint_cache) &&
           #
           isempty(model.multiplicative_parameters_pv) &&
           isempty(model.multiplicative_parameters_pp) &&
           isempty(model.dual_value_of_parameters) &&
           model.number_of_parameters_in_model == 0 &&
           isempty(model.parameters_in_conflict) &&
           isempty(model.ext)
end

function MOI.empty!(model::Optimizer{T}) where {T}
    MOI.empty!(model.optimizer)
    empty!(model.parameters)
    empty!(model.parameters_name)
    empty!(model.updated_parameters)
    model.last_parameter_index_added = PARAMETER_INDEX_THRESHOLD
    empty!(model.constraint_outer_to_inner)
    # affine ctr
    model.last_affine_added = 0
    empty!(model.affine_outer_to_inner)
    empty!(model.affine_constraint_cache)
    empty!(model.affine_constraint_cache_set)
    # quad ctr
    model.last_quad_add_added = 0
    model.last_vec_quad_add_added = 0
    empty!(model.quadratic_outer_to_inner)
    empty!(model.vector_quadratic_outer_to_inner)
    empty!(model.quadratic_constraint_cache)
    empty!(model.quadratic_constraint_cache_set)
    empty!(model.vector_quadratic_constraint_cache)
    empty!(model.vector_quadratic_constraint_cache_set)
    # obj
    model.affine_objective_cache = nothing
    model.quadratic_objective_cache = nothing
    MOI.empty!(model.original_objective_cache)
    empty!(model.quadratic_objective_cache_product)
    #
    empty!(model.vector_affine_constraint_cache)
    #
    empty!(model.multiplicative_parameters_pv)
    empty!(model.multiplicative_parameters_pp)
    empty!(model.dual_value_of_parameters)
    #
    model.number_of_parameters_in_model = 0
    empty!(model.parameters_in_conflict)
    empty!(model.ext)
    return
end

#
# Variables
#

function MOI.is_valid(model::Optimizer, vi::MOI.VariableIndex)
    if haskey(model.parameters, p_idx(vi))
        return true
    elseif MOI.is_valid(model.optimizer, vi)
        return true
    end
    return false
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T}
    vi = MOI.VariableIndex(ci.value)
    if haskey(model.parameters, p_idx(vi))
        return true
    end
    return false
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.VariableName,
    tp::Type{MOI.VariableIndex},
)
    return MOI.supports(model.optimizer, attr, tp)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.VariableName,
    v::MOI.VariableIndex,
    name::String,
)
    if _parameter_in_model(model, v)
        model.parameters_name[v] = name
    else
        MOI.set(model.optimizer, attr, v, name)
    end
    return
end

function MOI.get(model::Optimizer, attr::MOI.VariableName, v::MOI.VariableIndex)
    if _parameter_in_model(model, v)
        return get(model.parameters_name, v, "")
    else
        return MOI.get(model.optimizer, attr, v)
    end
end

function MOI.get(model::Optimizer, tp::Type{MOI.VariableIndex}, attr::String)
    return MOI.get(model.optimizer, tp, attr)
end

function _add_variable(model::Optimizer, inner_vi)
    if _is_parameter(inner_vi)
        error(
            "Attempted to add a variable but got a parameter index. The inner solver should not create variables with index >= $PARAMETER_INDEX_THRESHOLD, (got $(inner_vi.value)).",
        )
    end
    return inner_vi
end

function MOI.add_variable(model::Optimizer)
    return _add_variable(model, MOI.add_variable(model.optimizer))
end

function MOI.supports_add_constrained_variable(
    ::Optimizer{T},
    ::Type{MOI.Parameter{T}},
) where {T}
    return true
end

# this method is necessary due ambiguities created by an MOI function
function MOI.supports_add_constrained_variables(
    model::Optimizer,
    ::Type{MOI.Reals},
)
    return MOI.supports_add_constrained_variables(model.optimizer, MOI.Reals)
end

function MOI.supports_add_constrained_variable(
    model::Optimizer,
    ::Type{S},
) where {S<:MOI.AbstractScalarSet}
    return MOI.supports_add_constrained_variable(model.optimizer, S)
end

function MOI.supports_add_constrained_variables(
    model::Optimizer,
    ::Type{S},
) where {S<:MOI.AbstractVectorSet}
    return MOI.supports_add_constrained_variables(model.optimizer, S)
end

function _assert_parameter_is_finite(set::MOI.Parameter{T}) where {T}
    if !isfinite(set.value)
        throw(
            AssertionError(
                "Parameter value must be a finite number. Got $(set.value)",
            ),
        )
    end
end

function MOI.add_constrained_variable(
    model::Optimizer{T},
    set::MOI.Parameter{T},
) where {T}
    _assert_parameter_is_finite(set)
    _next_parameter_index!(model)
    p = MOI.VariableIndex(model.last_parameter_index_added)
    MOI.Utilities.CleverDicts.add_item(model.parameters, set.value)
    cp = MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}}(
        model.last_parameter_index_added,
    )
    _add_to_constraint_map!(model, cp)
    MOI.Utilities.CleverDicts.add_item(model.updated_parameters, NaN)
    _update_number_of_parameters!(model)
    return p, cp
end

function MOI.add_constrained_variable(
    model::Optimizer,
    set::MOI.AbstractScalarSet,
)
    inner_vi, inner_ci = MOI.add_constrained_variable(model.optimizer, set)
    outer_vi = _add_variable(model, inner_vi)
    outer_ci =
        MOI.ConstraintIndex{MOI.VariableIndex,typeof(set)}(outer_vi.value)
    model.constraint_outer_to_inner[outer_ci] = inner_ci
    return outer_vi, outer_ci
end

function MOI.add_constrained_variables(
    model::Optimizer,
    set::MOI.AbstractVectorSet,
)
    inner_vis, inner_ci = MOI.add_constrained_variables(model.optimizer, set)
    _add_to_constraint_map!(model, inner_ci)
    return _add_variable.(model, inner_vis), inner_ci
end

function _add_to_constraint_map!(model::Optimizer, ci)
    model.constraint_outer_to_inner[ci] = ci
    return
end

function _add_to_constraint_map!(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F,S},
) where {F<:MOI.ScalarAffineFunction,S}
    model.last_affine_added += 1
    model.constraint_outer_to_inner[ci] = ci
    return
end

function _add_to_constraint_map!(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F,S},
) where {F<:MOI.ScalarQuadraticFunction,S}
    model.last_quad_add_added += 1
    model.constraint_outer_to_inner[ci] = ci
    return
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    tp::Type{MOI.VariableIndex},
)
    return MOI.supports(model.optimizer, attr, tp)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimalStart,
    v::MOI.VariableIndex,
)
    if _is_parameter(v)
        if haskey(model.parameters, p_idx(v))
            return model.parameters[p_idx(v)]
        else
            throw(MOI.InvalidIndex(v))
        end
    end
    # inner model will throw if not valid
    return MOI.get(model.optimizer, attr, v)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.VariablePrimalStart,
    v::MOI.VariableIndex,
    val,
)
    if _is_parameter(v)
        if !haskey(model.parameters, p_idx(v))
            throw(MOI.InvalidIndex(v))
        end
        # this is effectivelly a no-op, but we do validation
        _val = model.parameters[p_idx(v)]
        if val != _val
            error(
                "The parameter $v value is $_val, but trying to set VariablePrimalStart $val",
            )
        end
        return
    end
    return MOI.set(model.optimizer, attr, v, val)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    v::MOI.VariableIndex,
    val,
)
    if _is_parameter(v)
        error("$attr is not supported for parameters in ParametricOptInterface")
    end
    return MOI.set(model.optimizer, attr, v, val)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    v::MOI.VariableIndex,
)
    if _is_parameter(v)
        error("$attr is not supported for parameters in ParametricOptInterface")
    end
    return MOI.get(model.optimizer, attr, v)
end

function MOI.delete(model::Optimizer, v::MOI.VariableIndex)
    if !MOI.is_valid(model, v)
        throw(MOI.InvalidIndex(v))
    end
    if _is_parameter(v)
        error("Cannot delete parameters in ParametricOptInterface.")
    end
    MOI.delete(model.optimizer, v)
    MOI.delete(model.original_objective_cache, v)
    # TODO - what happens if the variable was in a SAF that was converted to bounds?
    # solution: do not allow if that is the case (requires going through the scalar affine cache)
    # TODO - deleting a variable also deletes constraints
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(
        model.constraint_outer_to_inner,
    )
        _delete_variable_index_constraint(
            model,
            model.constraint_outer_to_inner,
            F,
            S,
            v.value,
        )
    end
    return
end

function _delete_variable_index_constraint(model, d, F, S, v)
    return
end

function _delete_variable_index_constraint(
    model,
    d,
    F::Type{MOI.VectorOfVariables},
    S,
    v,
)
    inner = d[F, S]
    for (key, val) in inner
        if !MOI.is_valid(model.optimizer, val)
            delete!(inner, key)
        end
    end
    return
end

function _delete_variable_index_constraint(
    model,
    d,
    F::Type{MOI.VariableIndex},
    S,
    value,
)
    inner = d[F, S]
    key = MOI.ConstraintIndex{F,S}(value)
    delete!(inner, key)
    return
end

#
# Constraints
#

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {F<:Union{MOI.VariableIndex,MOI.VectorOfVariables},S<:MOI.AbstractSet}
    return MOI.is_valid(model.optimizer, c)
end

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction,
        MOI.ScalarQuadraticFunction,
        MOI.VectorAffineFunction,
        MOI.VectorQuadraticFunction,
    },
    S<:MOI.AbstractSet,
}
    if haskey(model.constraint_outer_to_inner, c)
        return true
    end
    return MOI.is_valid(model.optimizer, c)
end

function MOI.supports_constraint(
    model::Optimizer,
    F::Union{
        Type{MOI.VariableIndex},
        Type{MOI.ScalarAffineFunction{T}},
        Type{MOI.VectorOfVariables},
        Type{MOI.VectorAffineFunction{T}},
    },
    S::Type{<:MOI.AbstractSet},
) where {T}
    return MOI.supports_constraint(model.optimizer, F, S)
end

function MOI.supports_constraint(
    model::Optimizer,
    ::Type{MOI.ScalarQuadraticFunction{T}},
    S::Type{<:MOI.AbstractSet},
) where {T}
    return MOI.supports_constraint(
        model.optimizer,
        MOI.ScalarAffineFunction{T},
        S,
    )
end

function MOI.supports_constraint(
    model::Optimizer,
    ::Type{MOI.VectorQuadraticFunction{T}},
    S::Type{<:MOI.AbstractSet},
) where {T}
    return MOI.supports_constraint(
        model.optimizer,
        MOI.VectorAffineFunction{T},
        S,
    )
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.ConstraintName,
    tp::Type{<:MOI.ConstraintIndex},
)
    return MOI.supports(model.optimizer, attr, tp)
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.ConstraintName,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {T,F<:MOI.ScalarQuadraticFunction{T},S}
    G = MOI.ScalarAffineFunction{T}
    # We can't tell at type-time whether the constraints will be quadratic or
    # lowered to affine, so we return the conservative choice for supports of
    # needing to support names for both quadratic and affine constraints.
    if MOI.supports_constraint(model.optimizer, F, S)
        return MOI.supports(model.optimizer, attr, MOI.ConstraintIndex{F,S}) &&
               MOI.supports(model.optimizer, attr, MOI.ConstraintIndex{G,S})
    end
    return MOI.supports(model.optimizer, attr, MOI.ConstraintIndex{G,S})
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.ConstraintName,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {T,F<:MOI.VectorQuadraticFunction{T},S}
    G = MOI.VectorAffineFunction{T}
    # We can't tell at type-time whether the constraints will be quadratic or
    # lowered to affine, so we return the conservative choice for supports of
    # needing to support names for both quadratic and affine constraints.
    # TODO:
    # switch to only check support name for the case of linear
    # is a solver does not support quadratic constraints it will fain in add_
    if MOI.supports_constraint(model.optimizer, F, S)
        return MOI.supports(model.optimizer, attr, MOI.ConstraintIndex{F,S}) &&
               MOI.supports(model.optimizer, attr, MOI.ConstraintIndex{G,S})
    end
    return MOI.supports(model.optimizer, attr, MOI.ConstraintIndex{G,S})
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S},
    name::String,
) where {T,S<:MOI.AbstractSet}
    c_aux = c
    if haskey(model.quadratic_outer_to_inner, c)
        c_aux = model.quadratic_outer_to_inner[c]
    end
    MOI.set(model.optimizer, attr, c_aux, name)
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.VectorQuadraticFunction{T},S},
    name::String,
) where {T,S<:MOI.AbstractSet}
    c_aux = c
    if haskey(model.vector_quadratic_outer_to_inner, c)
        c_aux = model.vector_quadratic_outer_to_inner[c]
    end
    MOI.set(model.optimizer, attr, c_aux, name)
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S},
    name::String,
) where {T,S<:MOI.AbstractSet}
    if haskey(model.affine_outer_to_inner, c)
        MOI.set(model.optimizer, attr, model.affine_outer_to_inner[c], name)
    else
        MOI.set(model.optimizer, attr, c, name)
    end
    return
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S},
) where {T,S<:MOI.AbstractSet}
    c_aux = c
    if haskey(model.quadratic_outer_to_inner, c)
        c_aux = model.quadratic_outer_to_inner[c]
    end
    return MOI.get(model.optimizer, attr, c_aux)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.VectorQuadraticFunction{T},S},
) where {T,S<:MOI.AbstractSet}
    c_aux = c
    if haskey(model.vector_quadratic_outer_to_inner, c)
        c_aux = model.vector_quadratic_outer_to_inner[c]
    end
    return MOI.get(model.optimizer, attr, c_aux)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S},
) where {T,S}
    if haskey(model.affine_outer_to_inner, c)
        inner_ci = model.affine_outer_to_inner[c]
        # This SAF constraint was transformed into variable bound
        if typeof(inner_ci) === MOI.ConstraintIndex{MOI.VariableIndex,S}
            v = MOI.get(model.optimizer, MOI.ConstraintFunction(), inner_ci)
            variable_name = MOI.get(model.optimizer, MOI.VariableName(), v)
            return "ParametricBound_$(S)_$(variable_name)"
        end
        return MOI.get(model.optimizer, attr, inner_ci)
    else
        return MOI.get(model.optimizer, attr, c)
    end
end

function MOI.get(
    model::Optimizer,
    tp::Type{MOI.ConstraintIndex{F,S}},
    name::String,
) where {F,S}
    return MOI.get(model.optimizer, tp, name)
end

function MOI.get(model::Optimizer, tp::Type{MOI.ConstraintIndex}, name::String)
    return MOI.get(model.optimizer, tp, name)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{F},
    f::F,
) where {F}
    MOI.set(model.optimizer, MOI.ConstraintFunction(), c, f)
    return
end

function MOI.get(
    model::Optimizer{T},
    attr::MOI.ConstraintFunction,
    ci::MOI.ConstraintIndex,
) where {T}
    if haskey(model.quadratic_outer_to_inner, ci)
        inner_ci = model.quadratic_outer_to_inner[ci]
        if haskey(model.quadratic_constraint_cache, inner_ci)
            return _original_function(
                model.quadratic_constraint_cache[inner_ci],
            )
        else
            return convert(
                MOI.ScalarQuadraticFunction{T},
                MOI.get(model.optimizer, attr, inner_ci),
            )
        end
    elseif haskey(model.vector_quadratic_outer_to_inner, ci)
        inner_ci = model.vector_quadratic_outer_to_inner[ci]
        if haskey(model.vector_quadratic_constraint_cache, inner_ci)
            return _original_function(
                model.vector_quadratic_constraint_cache[inner_ci],
            )
        else
            return convert(
                MOI.VectorQuadraticFunction{T},
                MOI.get(model.optimizer, attr, inner_ci),
            )
        end
    elseif haskey(model.affine_outer_to_inner, ci)
        inner_ci = model.affine_outer_to_inner[ci]
        return _original_function(model.affine_constraint_cache[inner_ci])
    else
        MOI.throw_if_not_valid(model, ci)
        return MOI.get(model.optimizer, attr, ci)
    end
end

function MOI.get(
    model::Optimizer{T},
    ::MOI.ConstraintFunction,
    cp::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T}
    p = MOI.VariableIndex(cp.value)
    if !_parameter_in_model(model, p)
        throw(MOI.InvalidIndex(cp))
    end
    return p
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,S},
    s::S,
) where {F,S}
    MOI.set(model.optimizer, MOI.ConstraintSet(), c, s)
    return
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex,
)
    if haskey(model.quadratic_outer_to_inner, ci)
        inner_ci = model.quadratic_outer_to_inner[ci]
        return model.quadratic_constraint_cache_set[inner_ci]
    elseif haskey(model.vector_quadratic_outer_to_inner, ci)
        inner_ci = model.vector_quadratic_outer_to_inner[ci]
        return model.vector_quadratic_constraint_cache_set[inner_ci]
    elseif haskey(model.affine_outer_to_inner, ci)
        inner_ci = model.affine_outer_to_inner[ci]
        return model.affine_constraint_cache_set[inner_ci]
    else
        MOI.throw_if_not_valid(model, ci)
        return MOI.get(model.optimizer, attr, ci)
    end
end

function MOI.set(
    model::Optimizer{T},
    ::MOI.ConstraintSet,
    cp::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
    set::MOI.Parameter{T},
) where {T}
    _assert_parameter_is_finite(set)
    p = MOI.VariableIndex(cp.value)
    if !_parameter_in_model(model, p)
        throw(MOI.InvalidIndex(cp))
    end
    return model.updated_parameters[p_idx(p)] = set.value
end

function MOI.get(
    model::Optimizer{T},
    ::MOI.ConstraintSet,
    cp::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T}
    p = MOI.VariableIndex(cp.value)
    if !_parameter_in_model(model, p)
        throw(MOI.InvalidIndex(cp))
    end
    val = model.updated_parameters[p_idx(p)]
    if isnan(val)
        return MOI.Parameter{T}(model.parameters[p_idx(p)])
    end
    return MOI.Parameter{T}(val)
end

function MOI.modify(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
    chg::Union{MOI.ScalarConstantChange{T},MOI.ScalarCoefficientChange{T}},
) where {F,S,T}
    if haskey(model.quadratic_outer_to_inner, c) ||
       haskey(model.vector_quadratic_outer_to_inner, c) ||
       haskey(model.affine_outer_to_inner, c)
        error(
            "Parametric constraint cannot be modified in ParametricOptInterface, because it would conflict with parameter updates. You can update the parameters instead.",
        )
    end
    MOI.modify(model.optimizer, c, chg)
    return
end

function _add_constraint_direct_and_cache_map!(model::Optimizer, f, set)
    ci = MOI.add_constraint(model.optimizer, f, set)
    _add_to_constraint_map!(model, ci)
    return ci
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.VariableIndex,
    set::MOI.AbstractScalarSet,
)
    if _is_parameter(f)
        error("Cannot constrain a parameter in ParametricOptInterface.")
    elseif !MOI.is_valid(model, f)
        throw(MOI.InvalidIndex(f))
    end
    return _add_constraint_direct_and_cache_map!(model, f, set)
end

function _add_constraint_with_parameters_on_function(
    model::Optimizer,
    f::MOI.ScalarAffineFunction{T},
    set::S,
) where {T,S}
    pf = ParametricAffineFunction(f)
    if model.constraints_interpretation == ONLY_BOUNDS
        if length(pf.v) == 1 && isone(MOI.coefficient(pf.v[]))
            poi_ci = _add_vi_constraint(model, pf, set)
        else
            error(
                "It was not possible to interpret this constraint as a variable bound. You can change the `ConstraintsInterpretation` to BOUNDS_AND_CONSTRAINTS or ONLY_CONSTRAINTS to allow this constraint to be added as a general constraint.",
            )
        end
    elseif model.constraints_interpretation == ONLY_CONSTRAINTS
        poi_ci = MOI.add_constraint(model, pf, set)
    elseif model.constraints_interpretation == BOUNDS_AND_CONSTRAINTS
        if length(pf.v) == 1 && isone(MOI.coefficient(pf.v[]))
            poi_ci = _add_vi_constraint(model, pf, set)
        else
            poi_ci = MOI.add_constraint(model, pf, set)
        end
    end
    return poi_ci
end

function MOI.add_constraint(
    model::Optimizer,
    pf::ParametricAffineFunction{T},
    set::S,
) where {T,S}
    _cache_set_constant!(pf, set)
    _update_cache!(pf, model)
    inner_ci = MOI.add_constraint(
        model.optimizer,
        MOI.ScalarAffineFunction{T}(pf.v, 0.0),
        _set_with_new_constant(set, pf.current_constant),
    )
    model.last_affine_added += 1
    outer_ci = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S}(
        model.last_affine_added,
    )
    model.affine_outer_to_inner[outer_ci] = inner_ci
    model.constraint_outer_to_inner[outer_ci] = inner_ci
    model.affine_constraint_cache[inner_ci] = pf
    model.affine_constraint_cache_set[inner_ci] = set
    return outer_ci
end

function _add_vi_constraint(
    model::Optimizer,
    pf::ParametricAffineFunction{T},
    set::S,
) where {T,S}
    _cache_set_constant!(pf, set)
    _update_cache!(pf, model)
    inner_ci = MOI.add_constraint(
        model.optimizer,
        pf.v[].variable,
        _set_with_new_constant(set, pf.current_constant),
    )
    model.last_affine_added += 1
    outer_ci = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S}(
        model.last_affine_added,
    )
    model.affine_outer_to_inner[outer_ci] = inner_ci
    model.constraint_outer_to_inner[outer_ci] = inner_ci
    model.affine_constraint_cache[inner_ci] = pf
    model.affine_constraint_cache_set[inner_ci] = set
    return outer_ci
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.ScalarAffineFunction{T},
    set::MOI.AbstractScalarSet,
) where {T}
    if !_has_parameters(f)
        return _add_constraint_direct_and_cache_map!(model, f, set)
    else
        return _add_constraint_with_parameters_on_function(model, f, set)
    end
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.VectorOfVariables,
    set::MOI.AbstractVectorSet,
)
    if _has_parameters(f)
        error(
            "VectorOfVariables does not allow parameters in ParametricOptInterface.",
        )
    end
    return _add_constraint_direct_and_cache_map!(model, f, set)
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.VectorAffineFunction{T},
    set::MOI.AbstractVectorSet,
) where {T}
    if !_has_parameters(f)
        return _add_constraint_direct_and_cache_map!(model, f, set)
    else
        return _add_constraint_with_parameters_on_function(model, f, set)
    end
end

function _add_constraint_with_parameters_on_function(
    model::Optimizer,
    f::MOI.VectorAffineFunction{T},
    set::MOI.AbstractVectorSet,
) where {T}
    pf = ParametricVectorAffineFunction(f)
    # _cache_set_constant!(pf, set) # there is no constant is vector sets
    _update_cache!(pf, model)
    inner_ci = MOI.add_constraint(model.optimizer, _current_function(pf), set)
    model.vector_affine_constraint_cache[inner_ci] = pf
    _add_to_constraint_map!(model, inner_ci)
    return inner_ci
end

function _add_constraint_with_parameters_on_function(
    model::Optimizer,
    f::MOI.ScalarQuadraticFunction{T},
    s::S,
) where {T,S<:MOI.AbstractScalarSet}
    pf = ParametricQuadraticFunction(f)
    _cache_multiplicative_params!(model, pf)
    _cache_set_constant!(pf, s)
    _update_cache!(pf, model)

    func = _current_function(pf)
    if !_is_affine(func)
        fq = func
        inner_ci =
            MOI.Utilities.normalize_and_add_constraint(model.optimizer, fq, s)
        model.last_quad_add_added += 1
        outer_ci = MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S}(
            model.last_quad_add_added,
        )
        model.quadratic_outer_to_inner[outer_ci] = inner_ci
        model.constraint_outer_to_inner[outer_ci] = inner_ci
    else
        fa = MOI.ScalarAffineFunction(func.affine_terms, func.constant)
        inner_ci =
            MOI.Utilities.normalize_and_add_constraint(model.optimizer, fa, s)
        model.last_quad_add_added += 1
        outer_ci = MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S}(
            model.last_quad_add_added,
        )
        # This part is used to remember that ci came from a quadratic function
        # It is particularly useful because sometimes the constraint mutates
        model.quadratic_outer_to_inner[outer_ci] = inner_ci
        model.constraint_outer_to_inner[outer_ci] = inner_ci
    end
    model.quadratic_constraint_cache[inner_ci] = pf
    model.quadratic_constraint_cache_set[inner_ci] = s
    return outer_ci
end

function _is_affine(f::MOI.ScalarQuadraticFunction)
    if isempty(f.quadratic_terms)
        return true
    end
    return false
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.ScalarQuadraticFunction{T},
    set::MOI.AbstractScalarSet,
) where {T}
    if !_has_parameters(f)
        # The user might construct the expression `*(f::Vector{AffExpr}, p)`
        # where `p` is a parameter. This results in a `Vector{QuadExpr}`
        # and hence in `ScalarQuadraticFunction` constraints.
        # If some entries of `f` are zero, then `has_parameters` will be zero for
        # the resulting constraint. We should however still turn it into an affine
        # function like the other entries.
        if _is_affine(f)
            fa = MOI.ScalarAffineFunction(f.affine_terms, f.constant)
            inner_ci = MOI.Utilities.normalize_and_add_constraint(
                model.optimizer,
                fa,
                set,
            )
            model.last_quad_add_added += 1
            outer_ci =
                MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},typeof(set)}(
                    model.last_quad_add_added,
                )
            model.quadratic_outer_to_inner[outer_ci] = inner_ci
            model.constraint_outer_to_inner[outer_ci] = inner_ci
            model.quadratic_constraint_cache_set[inner_ci] = set
            return outer_ci
        end
        return _add_constraint_direct_and_cache_map!(model, f, set)
    else
        return _add_constraint_with_parameters_on_function(model, f, set)
    end
end

function _is_vector_affine(f::MOI.VectorQuadraticFunction{T}) where {T}
    return isempty(f.quadratic_terms)
end

function _add_constraint_with_parameters_on_function(
    model::Optimizer,
    f::MOI.VectorQuadraticFunction{T},
    set::S,
) where {T,S}
    # Create parametric vector quadratic function
    pf = ParametricVectorQuadraticFunction(f)
    _cache_multiplicative_params!(model, pf)
    _update_cache!(pf, model)

    # Get the current function after parameter substitution
    func = _current_function(pf)
    if !_is_vector_affine(func)
        fq = func
        inner_ci = MOI.add_constraint(model.optimizer, fq, set)
        model.last_vec_quad_add_added += 1
        outer_ci = MOI.ConstraintIndex{MOI.VectorQuadraticFunction{T},S}(
            model.last_vec_quad_add_added,
        )
        model.vector_quadratic_outer_to_inner[outer_ci] = inner_ci
        model.constraint_outer_to_inner[outer_ci] = inner_ci
    else
        fa = MOI.VectorAffineFunction(func.affine_terms, func.constants)
        inner_ci = MOI.add_constraint(model.optimizer, fa, set)
        model.last_vec_quad_add_added += 1
        outer_ci = MOI.ConstraintIndex{MOI.VectorQuadraticFunction{T},S}(
            model.last_vec_quad_add_added,
        )
        # This part is used to remember that ci came from a quadratic function
        # It is particularly useful because sometimes the constraint mutates
        model.vector_quadratic_outer_to_inner[outer_ci] = inner_ci
        model.constraint_outer_to_inner[outer_ci] = inner_ci
    end
    model.vector_quadratic_constraint_cache[inner_ci] = pf
    model.vector_quadratic_constraint_cache_set[inner_ci] = set
    return outer_ci
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.VectorQuadraticFunction{T},
    set::MOI.AbstractVectorSet,
) where {T}
    if !_has_parameters(f)
        return _add_constraint_direct_and_cache_map!(model, f, set)
    else
        return _add_constraint_with_parameters_on_function(model, f, set)
    end
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorQuadraticFunction,S<:MOI.AbstractSet}
    if !MOI.is_valid(model, c)
        throw(MOI.InvalidIndex(c))
    end
    c_aux = c
    if haskey(model.vector_quadratic_outer_to_inner, c)
        ci_inner = model.vector_quadratic_outer_to_inner[c]
        delete!(model.vector_quadratic_outer_to_inner, c)
        delete!(model.vector_quadratic_constraint_cache, ci_inner)
        delete!(model.vector_quadratic_constraint_cache_set, ci_inner)
        c_aux = ci_inner
    end
    MOI.delete(model.optimizer, c_aux)
    delete!(model.constraint_outer_to_inner, c)
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {F<:MOI.ScalarQuadraticFunction,S<:MOI.AbstractSet}
    if !MOI.is_valid(model, c)
        throw(MOI.InvalidIndex(c))
    end
    c_aux = c
    if haskey(model.quadratic_outer_to_inner, c)
        ci_inner = model.quadratic_outer_to_inner[c]
        delete!(model.quadratic_outer_to_inner, c)
        delete!(model.quadratic_constraint_cache, ci_inner)
        delete!(model.quadratic_constraint_cache_set, ci_inner)
        c_aux = ci_inner
    end
    MOI.delete(model.optimizer, c_aux)
    delete!(model.constraint_outer_to_inner, c)
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {F<:MOI.ScalarAffineFunction,S<:MOI.AbstractSet}
    if !MOI.is_valid(model, c)
        throw(MOI.InvalidIndex(c))
    end
    if haskey(model.affine_outer_to_inner, c)
        ci_inner = model.affine_outer_to_inner[c]
        delete!(model.affine_outer_to_inner, c)
        delete!(model.affine_constraint_cache, ci_inner)
        delete!(model.affine_constraint_cache_set, ci_inner)
        MOI.delete(model.optimizer, ci_inner)
    else
        MOI.delete(model.optimizer, c)
    end
    delete!(model.constraint_outer_to_inner, c)
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {F<:Union{MOI.VariableIndex,MOI.VectorOfVariables},S<:MOI.AbstractSet}
    if !MOI.is_valid(model, c)
        throw(MOI.InvalidIndex(c))
    end
    MOI.delete(model.optimizer, c)
    delete!(model.constraint_outer_to_inner, c)
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorAffineFunction,S<:MOI.AbstractSet}
    if !MOI.is_valid(model, c)
        throw(MOI.InvalidIndex(c))
    end
    ci_inner = model.constraint_outer_to_inner[c]
    if haskey(model.vector_affine_constraint_cache, ci_inner)
        delete!(model.vector_affine_constraint_cache, ci_inner)
        MOI.delete(model.optimizer, ci_inner)
    else
        MOI.delete(model.optimizer, c)
    end
    delete!(model.constraint_outer_to_inner, c)
    return
end

#
# Objective
#

function MOI.supports(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{T},
) where {T}
    return false
end

function MOI.supports(
    model::Optimizer,
    attr::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{MOI.VariableIndex},
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}},
    },
) where {T}
    return MOI.supports(model.optimizer, attr)
end

function MOI.supports(
    model::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}},
) where {T}
    return MOI.supports(
        model.optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
    )
end

function MOI.modify(
    model::Optimizer,
    c::MOI.ObjectiveFunction{F},
    chg::Union{MOI.ScalarConstantChange{T},MOI.ScalarCoefficientChange{T}},
) where {F<:MathOptInterface.AbstractScalarFunction,T}
    if model.quadratic_objective_cache !== nothing ||
       model.affine_objective_cache !== nothing ||
       !isempty(model.quadratic_objective_cache_product)
        error(
            "A parametric objective cannot be modified as it would conflict with the parameter update mechanism. Please set a new objective or use parameters to perform such updates.",
        )
    end
    MOI.modify(model.optimizer, c, chg)
    MOI.modify(model.original_objective_cache, c, chg)
    return
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveSense)
    return MOI.get(model.optimizer, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveFunctionType)
    return MOI.get(model.original_objective_cache, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveFunction)
    return MOI.get(model.original_objective_cache, attr)
end

function _empty_objective_function_caches!(model::Optimizer{T}) where {T}
    model.affine_objective_cache = nothing
    model.quadratic_objective_cache = nothing
    model.cubic_objective_cache = nothing
    model.original_objective_cache = MOI.Utilities.ObjectiveContainer{T}()
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction,
    f::MOI.ScalarAffineFunction{T},
) where {T}
    # clear previously defined objetive function cache
    _empty_objective_function_caches!(model)
    if !_has_parameters(f)
        MOI.set(model.optimizer, attr, f)
    else
        pf = ParametricAffineFunction(f)
        _update_cache!(pf, model)
        MOI.set(model.optimizer, attr, _current_function(pf))
        model.affine_objective_cache = pf
    end
    MOI.set(model.original_objective_cache, attr, f)
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{F},
    f::F,
) where {F<:MOI.ScalarQuadraticFunction{T}} where {T}
    # clear previously defined objetive function cache
    _empty_objective_function_caches!(model)
    if !_has_parameters(f)
        MOI.set(model.optimizer, attr, f)
    else
        pf = ParametricQuadraticFunction(f)
        _cache_multiplicative_params!(model, pf)
        _update_cache!(pf, model)
        func = _current_function(pf)
        MOI.set(
            model.optimizer,
            MOI.ObjectiveFunction{(
                _is_affine(func) ? MOI.ScalarAffineFunction{T} :
                MOI.ScalarQuadraticFunction{T}
            )}(),
            # func,
            (
                _is_affine(func) ?
                MOI.ScalarAffineFunction(func.affine_terms, func.constant) :
                func
            ),
        )
        model.quadratic_objective_cache = pf
    end
    MOI.set(model.original_objective_cache, attr, f)
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction,
    v::MOI.VariableIndex,
)
    if _is_parameter(v)
        # TODO
        error(
            "Cannot use a parameter as objective function alone in ParametricOptInterface.",
        )
    elseif !MOI.is_valid(model, v)
        throw(MOI.InvalidIndex(v))
    end
    MOI.set(model.optimizer, attr, v)
    MOI.set(model.original_objective_cache, attr, v)
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    MOI.set(model.optimizer, attr, sense)
    return
end

#
# Other
#

function MOI.supports_incremental_interface(model::Optimizer)
    return MOI.supports_incremental_interface(model.optimizer)
end

#
# Attributes
#

function MOI.supports(model::Optimizer, ::MOI.Name)
    return MOI.supports(model.optimizer, MOI.Name())
end

MOI.get(model::Optimizer, ::MOI.Name) = MOI.get(model.optimizer, MOI.Name())

function MOI.set(model::Optimizer, ::MOI.Name, name::String)
    return MOI.set(model.optimizer, MOI.Name(), name)
end

function MOI.get(model::Optimizer, ::MOI.ListOfModelAttributesSet)
    list = MOI.get(model.optimizer, MOI.ListOfModelAttributesSet())
    # find subtypes of ObjectiveFunction and replace them with the correct one
    for (i, attr) in enumerate(list)
        if typeof(attr) <: MOI.ObjectiveFunction
            tp = MOI.get(model, MOI.ObjectiveFunctionType())
            list[i] = MOI.ObjectiveFunction{tp}()
            break
        end
    end
    return list
end

function MOI.get(model::Optimizer, ::MOI.ListOfVariableAttributesSet)
    return MOI.get(model.optimizer, MOI.ListOfVariableAttributesSet())
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintAttributesSet{F,S},
) where {T,F<:MOI.ScalarQuadraticFunction{T},S}
    if MOI.supports_constraint(model.optimizer, F, S)
        # in this case we cant tell if the constraint will be quadratic or
        # lowered to affine
        if model.warn_quad_affine_ambiguous
            println(
                "MOI.ListOfConstraintAttributesSet is not supported for ScalarQuadraticFunction in ParametricOptInterface, an empty list will be returned. This message can be suppressed by setting `POI._WarnIfQuadraticOfAffineFunctionAmbiguous` to false.",
            )
        end
        return []
    end
    return MOI.get(
        model.optimizer,
        MOI.ListOfConstraintAttributesSet{MOI.ScalarAffineFunction{T},S}(),
    )
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintAttributesSet{F,S},
) where {T,F<:MOI.VectorQuadraticFunction{T},S}
    if MOI.supports_constraint(model.optimizer, F, S)
        # in this case we cant tell if the constraint will be quadratic or
        # lowered to affine
        if model.warn_quad_affine_ambiguous
            println(
                "MOI.ListOfConstraintAttributesSet is not supported for VectorQuadraticFunction in ParametricOptInterface, an empty list will be returned. This message can be suppressed by setting `POI._WarnIfQuadraticOfAffineFunctionAmbiguous` to false.",
            )
        end
        return []
    end
    return MOI.get(
        model.optimizer,
        MOI.ListOfConstraintAttributesSet{MOI.VectorAffineFunction{T},S}(),
    )
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintAttributesSet{F,S},
) where {F,S}
    return MOI.get(model.optimizer, MOI.ListOfConstraintAttributesSet{F,S}())
end

function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    return MOI.get(model, NumberOfPureVariables()) +
           MOI.get(model, NumberOfParameters())
end

function MOI.get(model::Optimizer, ::MOI.NumberOfConstraints{F,S}) where {S,F}
    return Int64(length(model.constraint_outer_to_inner[F, S]))
end

function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return vcat(
        MOI.get(model, ListOfPureVariableIndices()),
        v_idx.(MOI.get(model, ListOfParameterIndices())),
    )
end

function MOI.get(model::Optimizer, ::MOI.ListOfConstraintTypesPresent)
    constraint_types = MOI.Utilities.DoubleDicts.nonempty_outer_keys(
        model.constraint_outer_to_inner,
    )
    return collect(constraint_types)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{F,S},
) where {S,F}
    list = collect(keys(model.constraint_outer_to_inner[F, S]))
    sort!(list, lt = (x, y) -> (x.value < y.value))
    return list
end

function MOI.supports(model::Optimizer, attr::MOI.AbstractOptimizerAttribute)
    return MOI.supports(model.optimizer, attr)
end

function MOI.get(model::Optimizer, attr::MOI.AbstractOptimizerAttribute)
    return MOI.get(model.optimizer, attr)
end

function MOI.set(model::Optimizer, attr::MOI.AbstractOptimizerAttribute, value)
    MOI.set(model.optimizer, attr, value)
    return
end

function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, val::Any)
    MOI.set(model.optimizer, attr, val)
    return
end

function MOI.get(model::Optimizer, ::MOI.SolverName)
    name = MOI.get(model.optimizer, MOI.SolverName())
    return "Parametric Optimizer with $(name) attached"
end

function MOI.get(model::Optimizer, ::MOI.SolverVersion)
    return MOI.get(model.optimizer, MOI.SolverVersion())
end

#
# Solutions Attributes
#

function MOI.supports(::Optimizer, attr::MOI.NLPBlock)
    return false
end

function MOI.set(::Optimizer, attr::MOI.NLPBlock, val)
    throw(MOI.UnsupportedAttribute(attr))
end

function MOI.get(::Optimizer, attr::MOI.NLPBlock)
    throw(MOI.UnsupportedAttribute(attr))
end

function MOI.supports(model::Optimizer, attr::MOI.AbstractModelAttribute)
    return MOI.supports(model.optimizer, attr)
end

function MOI.get(model::Optimizer, attr::MOI.AbstractModelAttribute)
    return MOI.get(model.optimizer, attr)
end

function MOI.set(model::Optimizer, attr::MOI.AbstractModelAttribute, val)
    return MOI.set(model.optimizer, attr, val)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    v::MOI.VariableIndex,
)
    if _is_parameter(v)
        if haskey(model.parameters, p_idx(v))
            return model.parameters[p_idx(v)]
        else
            throw(MOI.InvalidIndex(v))
        end
    end
    return MOI.get(model.optimizer, attr, v)
end

function MOI.get(
    model::Optimizer,
    attr::T,
) where {
    T<:Union{
        MOI.TerminationStatus,
        MOI.ObjectiveValue,
        MOI.DualObjectiveValue,
        MOI.PrimalStatus,
        MOI.DualStatus,
    },
}
    return MOI.get(model.optimizer, attr)
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ::Type{T},
) where {T}
    return MOI.supports(model.optimizer, attr, T)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    c::MOI.ConstraintIndex,
)
    optimizer_ci = get(model.constraint_outer_to_inner, c, c)
    return MOI.get(model.optimizer, attr, optimizer_ci)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex,
)
    # optimizer_ci = get(model.constraint_outer_to_inner, c, c)
    # value = MOI.get(model.optimizer, attr, optimizer_ci)
    # inner_ci = model.constraint_outer_to_inner[c]
    # if haskey(model.quadratic_constraint_cache_set, inner_ci)
    #     set = model.quadratic_constraint_cache_set[inner_ci]
    # # TODO : this method will not work well due to the usage of
    # of normalize and add. We need to add more info to cache
    # end
    return MOI.Utilities.get_fallback(model, attr, c)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    c::MOI.ConstraintIndex,
    val,
)
    optimizer_ci = get(model.constraint_outer_to_inner, c, nothing)
    if optimizer_ci === nothing
        throw(MOI.InvalidIndex(c))
    end
    return MOI.set(model.optimizer, attr, optimizer_ci, val)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T}
    v = MOI.VariableIndex(c.value)
    return model.parameters[p_idx(v)]
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintPrimalStart,
    c::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T}
    v = MOI.VariableIndex(c.value)
    if _parameter_in_model(model, v)
        return model.parameters[p_idx(v)]
    else
        throw(MOI.InvalidIndex(c))
    end
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintPrimalStart,
    c::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
    val,
) where {T}
    v = MOI.VariableIndex(c.value)
    if _parameter_in_model(model, v)
        _val = model.parameters[p_idx(v)]
        if val != _val
            error(
                "The parameter $v (from constraint $c) value is $_val, but trying to set ConstraintPrimalStart $val",
            )
        end
    else
        throw(MOI.InvalidIndex(c))
    end
    return
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    c::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
    val,
) where {T}
    v = MOI.VariableIndex(c.value)
    if !_parameter_in_model(model, v)
        throw(MOI.InvalidIndex(c))
    end
    return
end

#
# Special Attributes
#

"""
    ParametricObjectiveType <: MOI.AbstractModelAttribute

A model attribute for the type `P` of the ParametricOptInterface's parametric
function type in the objective function. The value os `P` can be `Nothing` if
the objective function is not parametric. The parametric function type can be
queried using the [`ParametricObjectiveFunction{P}`](@ref) attribute. The type
`P` can be `ParametricAffineFunction{T}` or `ParametricQuadraticFunction{T}`.
"""
struct ParametricObjectiveType <: MOI.AbstractModelAttribute end

function MOI.get(model::Optimizer{T}, ::ParametricObjectiveType) where {T}
    if model.quadratic_objective_cache !== nothing
        return ParametricQuadraticFunction{T}
    elseif model.affine_objective_cache !== nothing
        return ParametricAffineFunction{T}
    end
    return Nothing
end

"""
    ParametricObjectiveFunction{P} <: MOI.AbstractModelAttribute

A model attribute for the parametric objective function of type `P`. The type
`P` can be `ParametricAffineFunction{T}` or `ParametricQuadraticFunction{T}`.
"""
struct ParametricObjectiveFunction{T} <: MOI.AbstractModelAttribute end

function MOI.get(
    model::Optimizer{T},
    ::ParametricObjectiveFunction{ParametricQuadraticFunction{T}},
) where {T}
    if model.quadratic_objective_cache === nothing
        error("
            There is no parametric quadratic objective function in the model.
        ")
    end
    return model.quadratic_objective_cache
end

function MOI.get(
    model::Optimizer{T},
    ::ParametricObjectiveFunction{ParametricAffineFunction{T}},
) where {T}
    if model.affine_objective_cache === nothing
        error("
            There is no parametric affine objective function in the model.
        ")
    end
    return model.affine_objective_cache
end

"""
    ListOfParametricConstraintTypesPresent()

A model attribute for the list of tuples of the form `(F,S,P)`, where `F` is a
MOI function type, `S` is a set type and `P` is a ParametricOptInterface
parametric function type indicating that the attribute
[`DictOfParametricConstraintIndicesAndFunctions{F,S,P}`](@ref) returns a
non-empty dictionary.
"""
struct ListOfParametricConstraintTypesPresent <: MOI.AbstractModelAttribute end

function MOI.get(
    model::Optimizer{T},
    ::ListOfParametricConstraintTypesPresent,
) where {T}
    output = Set{Tuple{DataType,DataType,DataType}}()
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(
        model.affine_constraint_cache,
    )
        push!(output, (F, S, ParametricAffineFunction{T}))
    end
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(
        model.vector_affine_constraint_cache,
    )
        push!(output, (F, S, ParametricVectorAffineFunction{T}))
    end
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(
        model.quadratic_constraint_cache,
    )
        push!(output, (F, S, ParametricQuadraticFunction{T}))
    end
    return collect(output)
end

"""
    DictOfParametricConstraintIndicesAndFunctions{F,S,P}

A model attribute for a dictionary mapping constraint indices to parametric
functions. The key is a constraint index with scalar function type `F`
and set type `S` and the value is a parametric function of type `P`.
"""
struct DictOfParametricConstraintIndicesAndFunctions{F,S,P} <:
       MOI.AbstractModelAttribute end

function MOI.get(
    model::Optimizer,
    ::DictOfParametricConstraintIndicesAndFunctions{F,S,P},
) where {F,S,P<:ParametricAffineFunction}
    return model.affine_constraint_cache[F, S]
end

function MOI.get(
    model::Optimizer,
    ::DictOfParametricConstraintIndicesAndFunctions{F,S,P},
) where {F,S,P<:ParametricVectorAffineFunction}
    return model.vector_affine_constraint_cache[F, S]
end

function MOI.get(
    model::Optimizer,
    ::DictOfParametricConstraintIndicesAndFunctions{F,S,P},
) where {F,S,P<:ParametricQuadraticFunction}
    return model.quadratic_constraint_cache[F, S]
end

function MOI.get(
    model::Optimizer,
    ::DictOfParametricConstraintIndicesAndFunctions{F,S,P},
) where {F,S,P<:ParametricVectorQuadraticFunction}
    return model.vector_quadratic_constraint_cache[F, S]
end

"""
    NumberOfPureVariables

A model attribute for the number of pure variables in the model.
"""
struct NumberOfPureVariables <: MOI.AbstractModelAttribute end

function MOI.get(model::Optimizer, ::NumberOfPureVariables)
    return MOI.get(model.optimizer, MOI.NumberOfVariables())
end

"""
    ListOfPureVariableIndices

A model attribute for the list of pure variable indices in the model.
"""
struct ListOfPureVariableIndices <: MOI.AbstractModelAttribute end

function MOI.get(model::Optimizer, ::ListOfPureVariableIndices)
    return MOI.get(model.optimizer, MOI.ListOfVariableIndices())
end

"""
    NumberOfParameters

A model attribute for the number of parameters in the model.
"""
struct NumberOfParameters <: MOI.AbstractModelAttribute end

function MOI.get(model::Optimizer, ::NumberOfParameters)
    return length(model.parameters)
end

"""
    ListOfParameterIndices

A model attribute for the list of parameter indices in the model.
"""
struct ListOfParameterIndices <: MOI.AbstractModelAttribute end

function MOI.get(model::Optimizer, ::ListOfParameterIndices)
    return collect(keys(model.parameters))::Vector{ParameterIndex}
end

"""
    ParameterValue <: MOI.AbstractVariableAttribute

Attribute defined to set and get parameter values

# Example

```julia
MOI.set(model, POI.ParameterValue(), p, 2.0)
MOI.get(model, POI.ParameterValue(), p)
```
"""
struct ParameterValue <: MOI.AbstractVariableAttribute end

# We need a CachingOptimizer fallback to
# get ParameterValue working correctly on JuMP
# TODO: Think of a better solution for this

function MOI.set(
    opt::MOI.Utilities.CachingOptimizer,
    ::ParameterValue,
    var::MOI.VariableIndex,
    val::Float64,
)
    ci =
        MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{Float64}}(var.value)
    set = MOI.set(opt, MOI.ConstraintSet(), ci, MOI.Parameter(val))
    return nothing
end

function MOI.set(
    model::Optimizer,
    ::ParameterValue,
    var::MOI.VariableIndex,
    val::Float64,
)
    ci =
        MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{Float64}}(var.value)
    set = MOI.set(model, MOI.ConstraintSet(), ci, MOI.Parameter(val))
    return nothing
end

function MOI.set(
    opt::MOI.Utilities.CachingOptimizer,
    ::ParameterValue,
    vi::MOI.VariableIndex,
    val::Real,
)
    return MOI.set(opt, ParameterValue(), vi, convert(Float64, val))
end

function MOI.set(
    model::Optimizer,
    ::ParameterValue,
    vi::MOI.VariableIndex,
    val::Real,
)
    return MOI.set(model, ParameterValue(), vi, convert(Float64, val))
end

function MOI.get(
    opt::MOI.Utilities.CachingOptimizer,
    ::ParameterValue,
    var::MOI.VariableIndex,
)
    ci =
        MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{Float64}}(var.value)
    set = MOI.get(opt, MOI.ConstraintSet(), ci)
    return set.value
end

function MOI.get(model::Optimizer, ::ParameterValue, var::MOI.VariableIndex)
    return model.parameters[p_idx(var)]
end

"""
    ConstraintsInterpretation <: MOI.AbstractOptimizerAttribute

Attribute to define how [`Optimizer`](@ref) should interpret constraints.

- `POI.ONLY_CONSTRAINTS`: always interpret `ScalarAffineFunction` constraints as
  linear constraints. If an expression such as `x >= p1 + p2` appears, it will
  be treated like an affine constraint.
  **This is the default behaviour of [`Optimizer`](@ref)**

- `POI.ONLY_BOUNDS`: always interpret `ScalarAffineFunction` constraints as a
  variable bound. This is valid for constraints such as `x >= p` or
  `x >= p1 + p2`. If a constraint `x1 + x2 >= p` appears which is not a valid
  variable bound, an error will be thrown.

- `POI.BOUNDS_AND_CONSTRAINTS`: interpret `ScalarAffineFunction` constraints as
  a variable bound if they are a valid variable bound, for example, `x >= p` or
  `x >= p1 + p2`, and interpret them as linear constraints otherwise.

# Example

```jldoctest
julia> import MathOptInterface as MOI

julia> import ParametricOptInterface as POI

julia> model = POI.Optimizer(MOI.Utilities.Model{Float64}())
ParametricOptInterface.Optimizer{Float64, MOIU.Model{Float64}}
├ ObjectiveSense: FEASIBILITY_SENSE
├ ObjectiveFunctionType: MOI.ScalarAffineFunction{Float64}
├ NumberOfVariables: 0
└ NumberOfConstraints: 0

julia> MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_BOUNDS)
ONLY_BOUNDS::ConstraintsInterpretationCode = 1

julia> MOI.set(model, POI.ConstraintsInterpretation(), POI.ONLY_CONSTRAINTS)
ONLY_CONSTRAINTS::ConstraintsInterpretationCode = 0

julia> MOI.set(model, POI.ConstraintsInterpretation(), POI.BOUNDS_AND_CONSTRAINTS)
BOUNDS_AND_CONSTRAINTS::ConstraintsInterpretationCode = 2
```
"""
struct ConstraintsInterpretation <: MOI.AbstractOptimizerAttribute end

function MOI.set(
    model::Optimizer,
    ::ConstraintsInterpretation,
    value::ConstraintsInterpretationCode,
)
    return model.constraints_interpretation = value
end

struct QuadraticObjectiveCoef <: MOI.AbstractModelAttribute end

function _set_quadratic_product_in_obj!(model::Optimizer{T}) where {T}
    n = length(model.quadratic_objective_cache_product)

    f = if model.affine_objective_cache !== nothing
        _current_function(model.affine_objective_cache)
    elseif model.quadratic_objective_cache !== nothing
        _current_function(model.quadratic_objective_cache)
    else
        F = MOI.get(model.original_objective_cache, MOI.ObjectiveFunctionType())
        MOI.get(model.original_objective_cache, MOI.ObjectiveFunction{F}())
    end
    F = typeof(f)

    quadratic_prods_vector = MOI.ScalarQuadraticTerm{T}[]
    sizehint!(quadratic_prods_vector, n)

    for ((x, y), fparam) in model.quadratic_objective_cache_product
        # x, y = prod_var
        evaluated_fparam = _evaluate_parametric_expression(model, fparam)
        push!(
            quadratic_prods_vector,
            MOI.ScalarQuadraticTerm(evaluated_fparam, x, y),
        )
    end

    f_new = if F <: MOI.VariableIndex
        MOI.ScalarQuadraticFunction(
            quadratic_prods_vector,
            MOI.ScalarAffineTerm{T}[MOI.ScalarAffineTerm{T}(1.0, f)],
            0.0,
        )
    elseif F <: MOI.ScalarAffineFunction{T}
        MOI.ScalarQuadraticFunction(quadratic_prods_vector, f.terms, f.constant)
    elseif F <: MOI.ScalarQuadraticFunction{T}
        quadratic_terms = vcat(f.quadratic_terms, quadratic_prods_vector)
        MOI.ScalarQuadraticFunction(quadratic_terms, f.affine_terms, f.constant)
    end

    MOI.set(
        model.optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}(),
        f_new,
    )

    return
end

function _evaluate_parametric_expression(model::Optimizer, p::MOI.VariableIndex)
    return model.parameters[p_idx(p)]
end

function _evaluate_parametric_expression(
    model::Optimizer,
    fparam::MOI.ScalarAffineFunction{T},
) where {T}
    constant = fparam.constant
    terms = fparam.terms
    evaluated_parameter_expression = zero(T)
    for term in terms
        coef = term.coefficient
        p = term.variable
        evaluated_parameter_expression += coef * model.parameters[p_idx(p)]
        evaluated_parameter_expression += constant
    end
    return evaluated_parameter_expression
end

function MOI.set(
    model::Optimizer,
    ::QuadraticObjectiveCoef,
    (x1, x2)::Tuple{MOI.VariableIndex,MOI.VariableIndex},
    ::Nothing,
)
    if x1.value > x2.value
        aux = x1
        x1 = x2
        x2 = aux
    end
    delete!(model.quadratic_objective_cache_product, (x1, x2))
    model.quadratic_objective_cache_product_changed = true
    return
end

function MOI.set(
    model::Optimizer,
    ::QuadraticObjectiveCoef,
    (x1, x2)::Tuple{MOI.VariableIndex,MOI.VariableIndex},
    f_param::Union{MOI.VariableIndex,MOI.ScalarAffineFunction{T}},
) where {T}
    if x1.value > x2.value
        aux = x1
        x1 = x2
        x2 = aux
    end
    model.quadratic_objective_cache_product[(x1, x2)] = f_param
    model.quadratic_objective_cache_product_changed = true
    return
end

function MOI.get(
    model::Optimizer,
    ::QuadraticObjectiveCoef,
    (x1, x2)::Tuple{MOI.VariableIndex,MOI.VariableIndex},
)
    if x1.value > x2.value
        aux = x1
        x1 = x2
        x2 = aux
    end
    if haskey(model.quadratic_objective_cache_product, (x1, x2))
        return model.quadratic_objective_cache_product[(x1, x2)]
    else
        throw(
            ErrorException(
                "Parameter not set in product of variables ($x1,$x2)",
            ),
        )
    end
end

#
# Optimize
#

function MOI.optimize!(model::Optimizer)
    if !isempty(model.updated_parameters)
        update_parameters!(model)
    end
    if (
        !isempty(model.quadratic_objective_cache_product) ||
        model.quadratic_objective_cache_product_changed
    )
        model.quadratic_objective_cache_product_changed = false
        _set_quadratic_product_in_obj!(model)
    end
    MOI.optimize!(model.optimizer)
    if MOI.get(model, MOI.DualStatus()) != MOI.NO_SOLUTION &&
       model.evaluate_duals
        _compute_dual_of_parameters!(model)
    end
    return
end

#
# compute_conflict!
#

function MOI.compute_conflict!(model::Optimizer)
    empty!(model.parameters_in_conflict)
    MOI.compute_conflict!(model.optimizer)
    if MOI.get(model.optimizer, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
        for (F, S) in keys(model.affine_constraint_cache.dict)
            affine_constraint_cache_inner = model.affine_constraint_cache[F, S]
            for (inner_ci, pf) in affine_constraint_cache_inner
                if MOI.get(
                    model.optimizer,
                    MOI.ConstraintConflictStatus(),
                    inner_ci,
                ) == MOI.NOT_IN_CONFLICT
                    continue
                end
                for term in pf.p
                    push!(model.parameters_in_conflict, term.variable)
                end
            end
        end
        for (F, S) in keys(model.quadratic_constraint_cache.dict)
            quadratic_constraint_cache_inner =
                model.quadratic_constraint_cache[F, S]
            for (inner_ci, pf) in quadratic_constraint_cache_inner
                if MOI.get(
                    model.optimizer,
                    MOI.ConstraintConflictStatus(),
                    inner_ci,
                ) == MOI.NOT_IN_CONFLICT
                    continue
                end
                for term in pf.p
                    push!(model.parameters_in_conflict, term.variable)
                end
                for term in pf.pp
                    push!(model.parameters_in_conflict, term.variable_1)
                    push!(model.parameters_in_conflict, term.variable_2)
                end
                for term in pf.pv
                    push!(model.parameters_in_conflict, term.variable_1)
                end
            end
        end
        for (F, S) in keys(model.vector_affine_constraint_cache.dict)
            vector_affine_constraint_cache_inner =
                model.vector_affine_constraint_cache[F, S]
            for (inner_ci, pf) in vector_affine_constraint_cache_inner
                if MOI.get(
                    model.optimizer,
                    MOI.ConstraintConflictStatus(),
                    inner_ci,
                ) == MOI.NOT_IN_CONFLICT
                    continue
                end
                for term in pf.p
                    push!(
                        model.parameters_in_conflict,
                        term.scalar_term.variable,
                    )
                end
            end
        end
    end
    return
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintConflictStatus,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:MOI.Parameter},
)
    return MOI.VariableIndex(ci.value) in model.parameters_in_conflict ?
           MOI.MAYBE_IN_CONFLICT : MOI.NOT_IN_CONFLICT
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(dest, src)
end

function MOI.Utilities.final_touch(model::Optimizer, index_map)
    return MOI.Utilities.final_touch(model.optimizer, index_map)
end

"""
    _WarnIfQuadraticOfAffineFunctionAmbiguous

Some attributes such as `MOI.ListOfConstraintAttributesSet` are ambiguous
when the model contains parametric quadratic functions that can be lowered
to affine functions. This attribute can be set to `false` to skip the warning
when such ambiguity arises. The default value is `true`.
"""
struct _WarnIfQuadraticOfAffineFunctionAmbiguous <:
       MOI.AbstractOptimizerAttribute end

function MOI.set(
    model::Optimizer,
    ::_WarnIfQuadraticOfAffineFunctionAmbiguous,
    value::Bool,
)
    model.warn_quad_affine_ambiguous = value
    return
end

function MOI.get(model::Optimizer, ::_WarnIfQuadraticOfAffineFunctionAmbiguous)
    return model.warn_quad_affine_ambiguous
end
