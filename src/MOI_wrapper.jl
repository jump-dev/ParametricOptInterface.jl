#
# Helpers
#

function _is_variable(v::MOI.VariableIndex)
    return v.value < PARAMETER_INDEX_THRESHOLD
end

function _is_parameter(v::MOI.VariableIndex)
    return v.value > PARAMETER_INDEX_THRESHOLD
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

function _cache_multiplicative_params!(
    model::Optimizer{T},
    f::ParametricQuadraticFunction{T},
) where {T}
    for term in f.pv
        push!(model.multiplicative_parameters, term.variable_1.value)
    end
    # TODO compute these duals might be feasible
    for term in f.pp
        push!(model.multiplicative_parameters, term.variable_1.value)
        push!(model.multiplicative_parameters, term.variable_2.value)
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
           isempty(model.variables) &&
           model.last_variable_index_added == 0 &&
           model.last_parameter_index_added == PARAMETER_INDEX_THRESHOLD &&
           isempty(model.constraint_outer_to_inner) &&
           # affine ctr
           model.last_affine_added == 0 &&
           isempty(model.affine_outer_to_inner) &&
           isempty(model.affine_constraint_cache) &&
           isempty(model.affine_constraint_cache_set) &&
           # quad ctr
           model.last_quad_add_added == 0 &&
           isempty(model.quadratic_outer_to_inner) &&
           isempty(model.quadratic_constraint_cache) &&
           isempty(model.quadratic_constraint_cache_set) &&
           # obj
           model.affine_objective_cache === nothing &&
           model.quadratic_objective_cache === nothing &&
           MOI.is_empty(model.original_objective_cache) &&
           isempty(model.quadratic_objective_cache_product) &&
           #
           isempty(model.vector_affine_constraint_cache) &&
           #
           isempty(model.multiplicative_parameters) &&
           isempty(model.dual_value_of_parameters) &&
           model.number_of_parameters_in_model == 0 &&
           isempty(model.ext)
end

function MOI.empty!(model::Optimizer{T}) where {T}
    MOI.empty!(model.optimizer)
    empty!(model.parameters)
    empty!(model.parameters_name)
    empty!(model.updated_parameters)
    empty!(model.variables)
    model.last_variable_index_added = 0
    model.last_parameter_index_added = PARAMETER_INDEX_THRESHOLD
    empty!(model.constraint_outer_to_inner)
    # affine ctr
    model.last_affine_added = 0
    empty!(model.affine_outer_to_inner)
    empty!(model.affine_constraint_cache)
    empty!(model.affine_constraint_cache_set)
    # quad ctr
    model.last_quad_add_added = 0
    empty!(model.quadratic_outer_to_inner)
    empty!(model.quadratic_constraint_cache)
    empty!(model.quadratic_constraint_cache_set)
    # obj
    model.affine_objective_cache = nothing
    model.quadratic_objective_cache = nothing
    MOI.empty!(model.original_objective_cache)
    empty!(model.quadratic_objective_cache_product)
    #
    empty!(model.vector_affine_constraint_cache)
    #
    empty!(model.multiplicative_parameters)
    empty!(model.dual_value_of_parameters)
    #
    model.number_of_parameters_in_model = 0
    empty!(model.ext)
    return
end

#
# Variables
#

function MOI.is_valid(model::Optimizer, vi::MOI.VariableIndex)
    return haskey(model.variables, vi) || haskey(model.parameters, p_idx(vi))
end

function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}}) where {T}
    return haskey(model.parameters, p_idx(MOI.VariableIndex(ci.value)))
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

function MOI.add_variable(model::Optimizer)
    _next_variable_index!(model)
    return MOI.Utilities.CleverDicts.add_item(
        model.variables,
        MOI.add_variable(model.optimizer),
    )
end

function MOI.supports_add_constrained_variable(
    ::Optimizer{T},
    ::Type{MOI.Parameter{T}},
) where {T}
    return true
end

function MOI.supports_add_constrained_variables(
    model::Optimizer,
    ::Type{MOI.Reals},
)
    return MOI.supports_add_constrained_variables(model.optimizer, MOI.Reals)
end

function MOI.add_constrained_variable(
    model::Optimizer{T},
    set::MOI.Parameter{T},
) where {T}
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

function _add_to_constraint_map!(model::Optimizer, ci)
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

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    v::MOI.VariableIndex,
    val,
)
    if _variable_in_model(model, v)
        MOI.set(model.optimizer, attr, v, val)
    else
        error("$attr is not supported for parameters")
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    v::MOI.VariableIndex,
)
    if _variable_in_model(model, v)
        return MOI.get(model.optimizer, attr, model.variables[v])
    else
        error("$attr is not supported for parameters")
    end
end

function MOI.delete(model::Optimizer, v::MOI.VariableIndex)
    delete!(model.variables, v)
    MOI.delete(model.optimizer, v)
    MOI.delete(model.original_objective_cache, v)
    # TODO - what happens if the variable was in a SAF that was converted to bounds?
    # solution: do not allow if that is the case (requires going trhought the scalar affine cache)
    # TODO - deleting a variable also deletes constraints
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(
        model.constraint_outer_to_inner,
    )
        _delete_variable_index_constraint(
            model.constraint_outer_to_inner,
            F,
            S,
            v.value,
        )
    end
    return
end

function _delete_variable_index_constraint(d, F, S, v)
    return
end

function _delete_variable_index_constraint(
    d,
    F::Type{MOI.VariableIndex},
    S,
    value,
)
    inner = d[F, S]
    for k in keys(inner)
        if k.value == value
            delete!(inner, k)
        end
    end
    return
end

#
# Constraints
#

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{MOI.VariableIndex,MOI.VectorOfVariables,MOI.VectorAffineFunction},
    S<:MOI.AbstractSet,
}
    return MOI.is_valid(model.optimizer, c)
end

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {F<:MOI.ScalarAffineFunction,S<:MOI.AbstractSet}
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

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S},
    name::String,
) where {T,S<:MOI.AbstractSet}
    if haskey(model.quadratic_outer_to_inner, c)
        MOI.set(model.optimizer, attr, model.quadratic_outer_to_inner[c], name)
    else
        MOI.set(model.optimizer, attr, c, name)
    end
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

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex,
    name::String,
)
    MOI.set(model.optimizer, attr, c, name)
    return
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S},
) where {T,S<:MOI.AbstractSet}
    if haskey(model.quadratic_outer_to_inner, c)
        return MOI.get(model.optimizer, attr, model.quadratic_outer_to_inner[c])
    else
        return MOI.get(model.optimizer, attr, c)
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex,
)
    return MOI.get(model.optimizer, attr, c)
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
    c::MOI.ConstraintIndex{F,S},
    f::F,
) where {F,S}
    MOI.set(model.optimizer, MOI.ConstraintFunction(), c, f)
    return
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintFunction,
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    if haskey(model.quadratic_outer_to_inner, ci)
        inner_ci = model.quadratic_outer_to_inner[ci]
        return _original_function(model.quadratic_constraint_cache[inner_ci])
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
        error("Parameter not in the model")
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
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    if haskey(model.quadratic_outer_to_inner, ci)
        inner_ci = model.quadratic_outer_to_inner[ci]
        return model.quadratic_constraint_cache_set[inner_ci]
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
    p = MOI.VariableIndex(cp.value)
    if !_parameter_in_model(model, p)
        error("Parameter not in the model")
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
        error("Parameter not in the model")
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
    chg::MOI.ScalarCoefficientChange{T},
) where {F,S,T}
    if haskey(model.quadratic_constraint_cache, c) ||
       haskey(model.affine_constraint_cache, c)
        error("Parametric constraint cannot be modified")
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
    if !_is_variable(f)
        error("Cannot constrain a parameter")
    elseif !_variable_in_model(model, f)
        error("Variable not in the model")
    end
    return _add_constraint_direct_and_cache_map!(model, f, set)
end

function _add_constraint_with_parameters_on_function(
    model::Optimizer,
    f::MOI.ScalarAffineFunction{T},
    set::S,
) where {T,S}
    pf = ParametricAffineFunction(f)
    _cache_set_constant!(pf, set)
    if model.constraints_interpretation == ONLY_BOUNDS
        if length(pf.v) == 1 && isone(MOI.coefficient(pf.v[]))
            poi_ci = _add_vi_constraint(model, pf, set)
        else
            error(
                "It was not possible to interpret this constraint as a variable bound.",
            )
        end
    elseif model.constraints_interpretation == ONLY_CONSTRAINTS
        poi_ci = _add_saf_constraint(model, pf, set)
    elseif model.constraints_interpretation == BOUNDS_AND_CONSTRAINTS
        if length(pf.v) == 1 && isone(MOI.coefficient(pf.v[]))
            poi_ci = _add_vi_constraint(model, pf, set)
        else
            poi_ci = _add_saf_constraint(model, pf, set)
        end
    end
    return poi_ci
end

function _add_saf_constraint(
    model::Optimizer,
    pf::ParametricAffineFunction{T},
    set::S,
) where {T,S}
    _update_cache!(pf, model)
    inner_ci = MOI.Utilities.normalize_and_add_constraint(
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
    _update_cache!(pf, model)
    inner_ci = MOI.Utilities.normalize_and_add_constraint(
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
        error("VectorOfVariables does not allow parameters")
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
    f_quad = if !_is_affine(func)
        fq = func
        inner_ci = MOI.Utilities.normalize_and_add_constraint(
            model.optimizer,
            fq,
            s,
        )
        model.last_quad_add_added += 1
        outer_ci = MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S}(
            model.last_quad_add_added,
        )
        model.quadratic_outer_to_inner[outer_ci] = inner_ci
        model.constraint_outer_to_inner[outer_ci] = inner_ci
    else
        fa = MOI.ScalarAffineFunction(func.affine_terms, func.constant)
        inner_ci = MOI.Utilities.normalize_and_add_constraint(
            model.optimizer,
            fa,
            s,
        )
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
        return _add_constraint_direct_and_cache_map!(model, f, set)
    else
        return _add_constraint_with_parameters_on_function(model, f, set)
    end
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {F<:MOI.ScalarQuadraticFunction,S<:MOI.AbstractSet}
    if haskey(model.quadratic_outer_to_inner, c)
        ci_inner = model.quadratic_outer_to_inner[c]
        deleteat!(model.quadratic_outer_to_inner, c)
        deleteat!(model.quadratic_constraint_cache, c)
        deleteat!(model.quadratic_constraint_cache_set, c)
        MOI.delete(model.optimizer, ci_inner)
    else
        MOI.delete(model.optimizer, c)
    end
    deleteat!(model.constraint_outer_to_inner, c)
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {F<:MOI.ScalarAffineFunction,S<:MOI.AbstractSet}
    if haskey(model.affine_outer_to_inner, c)
        ci_inner = model.affine_outer_to_inner[c]
        delete!(model.affine_outer_to_inner, c)
        delete!(model.affine_constraint_cache, c)
        delete!(model.affine_constraint_cache_set, c)
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
    MOI.delete(model.optimizer, c)
    delete!(model.constraint_outer_to_inner, c)
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorAffineFunction,S<:MOI.AbstractSet}
    MOI.delete(model.optimizer, c)
    delete!(model.constraint_outer_to_inner, c)
    deleteat!(model.vector_affine_constraint_cache, c)
    return
end

#
# Objective
#

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
        error("Parametric objective cannot be modified")
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
        error("Cannot use a parameter as objective function alone")
    elseif !_variable_in_model(model, v)
        error("Variable not in the model")
    end
    MOI.set(model.optimizer, attr, model.variables[v])
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
# NLP
#

function MOI.supports(model::Optimizer, ::MOI.NLPBlock)
    return MOI.supports(model.optimizer, MOI.NLPBlock())
end

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    return MOI.set(model.optimizer, MOI.NLPBlock(), nlp_data)
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
    return MOI.get(model.optimizer, MOI.ListOfModelAttributesSet())
end

function MOI.get(model::Optimizer, ::MOI.ListOfVariableAttributesSet)
    return MOI.get(model.optimizer, MOI.ListOfVariableAttributesSet())
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintAttributesSet{F,S},
) where {F,S}
    if F === MOI.ScalarQuadraticFunction
        error(
            "MOI.ListOfConstraintAttributesSet is not implemented for ScalarQuadraticFunction.",
        )
    end
    return MOI.get(model.optimizer, MOI.ListOfConstraintAttributesSet{F,S}())
end

function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    return length(model.parameters) + length(model.variables)
end

function MOI.get(model::Optimizer, ::MOI.NumberOfConstraints{F,S}) where {S,F}
    return length(model.constraint_outer_to_inner[F, S])
end

function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return MOI.get(model.optimizer, MOI.ListOfVariableIndices())
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
    list = collect(values(model.constraint_outer_to_inner[F, S]))
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

function MOI.get(model::Optimizer, attr::MOI.AbstractModelAttribute)
    return MOI.get(model.optimizer, attr)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    v::MOI.VariableIndex,
)
    if _parameter_in_model(model, v)
        return model.parameters[p_idx(v)]
    elseif _variable_in_model(model, v)
        return MOI.get(model.optimizer, attr, model.variables[v])
    else
        error("Variable not in the model")
    end
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

function MOI.get(
    model::Optimizer,
    attr::T,
    c::MOI.ConstraintIndex,
) where {
    T<:Union{MOI.ConstraintPrimal,MOI.ConstraintDual,MOI.ConstraintBasisStatus},
}
    return MOI.get(model.optimizer, attr, c)
end

function MOI.get(
    model::Optimizer,
    attr::AT,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S},
) where {
    AT<:Union{
        MOI.ConstraintPrimal,
        MOI.ConstraintDual,
        MOI.ConstraintBasisStatus,
    },
    T,
    S<:MOI.AbstractScalarSet,
}
    moi_ci = get(model.affine_outer_to_inner, c, c)
    return MOI.get(model.optimizer, attr, moi_ci)
end

#
# Special Attributes
#

struct ParametricObjectiveType <: MOI.AbstractModelAttribute end

function MOI.get(model::Optimizer{T}, ::ParametricObjectiveType) where {T}
    if model.quadratic_objective_cache !== nothing
        return ParametricQuadraticFunction{T}
    elseif model.affine_objective_cache !== nothing
        return ParametricAffineFunction{T}
    end
    return Nothing
end

struct ParametricObjectiveFunction{T} <: MOI.AbstractModelAttribute end

function MOI.get(model::Optimizer{T}, ::ParametricObjectiveFunction{ParametricQuadraticFunction{T}}) where {T}
    if model.quadratic_objective_cache === nothing
        error("
            There is no parametric quadratic objective function in the model.
        ")
    end
    return model.quadratic_objective_cache
end

function MOI.get(model::Optimizer{T}, ::ParametricObjectiveFunction{ParametricAffineFunction{T}}) where {T}
    if model.affine_objective_cache === nothing
        error("
            There is no parametric affine objective function in the model.
        ")
    end
    return model.affine_objective_cache
end

struct ListOfParametricConstraintTypesPresent <: MOI.AbstractModelAttribute end

function MOI.get(model::Optimizer{T}, ::ListOfParametricConstraintTypesPresent) where {T}
    output = Set{Tuple{DataType,DataType,DataType}}()
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(model.affine_constraint_cache)
        push!(output, (F, S, ParametricAffineFunction{T}))
    end
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(model.vector_affine_constraint_cache)
        push!(output, (F, S, ParametricVectorAffineFunction{T}))
    end
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(model.quadratic_constraint_cache)
        push!(output, (F, S, ParametricQuadraticFunction{T}))
    end
    return collect(output)
end

struct DictOfParametricConstraintIndicesAndFunctions{F,S,P} <: MOI.AbstractModelAttribute end

function MOI.get(
    model::Optimizer,
    ::DictOfParametricConstraintIndicesAndFunctions{F,S,P},
) where {F,S,P <: ParametricAffineFunction}
    return model.affine_constraint_cache[F, S]
end

function MOI.get(
    model::Optimizer,
    ::DictOfParametricConstraintIndicesAndFunctions{F,S,P},
) where {F,S,P <: ParametricVectorAffineFunction}
    return model.vector_affine_constraint_cache[F, S]
end

function MOI.get(
    model::Optimizer,
    ::DictOfParametricConstraintIndicesAndFunctions{F,S,P},
) where {F,S, P <: ParametricQuadraticFunction}
    return model.quadratic_constraint_cache[F, S]
end

struct ListOfPureVariableIndices <: MOI.AbstractModelAttribute end

function MOI.get(model::Optimizer, ::ListOfPureVariableIndices)
    return collect(keys(model.variables))
end

struct ListOfParameterIndices <: MOI.AbstractModelAttribute end

function MOI.get(model::Optimizer, ::ListOfParameterIndices)
    return collect(keys(model.parameters))
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

Attribute to define how [`POI.Optimizer`](@ref) should interpret constraints.

- `POI.ONLY_CONSTRAINTS`: Only interpret `ScalarAffineFunction` constraints as linear constraints
  If an expression such as `x >= p1 + p2` appears it will be trated like a new constraint.
  **This is the default behaviour of [`POI.Optimizer`](@ref)**

- `POI.ONLY_BOUNDS`: Only interpret `ScalarAffineFunction` constraints as a variable bound.
  This is valid for constraints such as `x >= p` or `x >= p1 + p2`. If a constraint `x1 + x2 >= p` appears,
  which is not a valid variable bound it will throw an error.

- `POI.BOUNDS_AND_CONSTRAINTS`: Interpret `ScalarAffineFunction` constraints as a variable bound if they
  are a valid variable bound, i.e., `x >= p` or `x >= p1 + p2` and interpret them as linear constraints
  otherwise.

# Example

```julia
MOI.set(model, POI.InterpretConstraintsAsBounds(), POI.ONLY_BOUNDS)
MOI.set(model, POI.InterpretConstraintsAsBounds(), POI.ONLY_CONSTRAINTS)
MOI.set(model, POI.InterpretConstraintsAsBounds(), POI.BOUNDS_AND_CONSTRAINTS)
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
# Copy
#

function MOI.Utilities.default_copy_to(
    dest::MOI.Bridges.LazyBridgeOptimizer{Optimizer{T,OT}},
    src::MOI.ModelLike,
) where {T,OT}
    return _poi_default_copy_to(dest, src)
end

function MOI.Utilities.default_copy_to(
    dest::Optimizer{T,OT},
    src::MOI.ModelLike,
) where {T,OT}
    return _poi_default_copy_to(dest, src)
end

function _poi_default_copy_to(dest::T, src::MOI.ModelLike) where {T}
    if !MOI.supports_incremental_interface(dest)
        error("Model $(typeof(dest)) does not support copy_to.")
    end
    MOI.empty!(dest)
    vis_src = MOI.get(src, MOI.ListOfVariableIndices())
    index_map = MOI.IndexMap()
    # The `NLPBlock` assumes that the order of variables does not change (#849)
    # Therefore, all VariableIndex and VectorOfVariable constraints are added
    # seprately, and no variables constrained-on-creation are added.

    # This is not valid for NLPs with Parameters, they should enter
    has_nlp = MOI.NLPBlock() in MOI.get(src, MOI.ListOfModelAttributesSet())
    constraints_not_added = if has_nlp
        vcat(
            Any[
                MOI.get(src, MOI.ListOfConstraintIndices{F,S}()) for
                (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent()) if
                MOI.Utilities._is_variable_function(F) &&
                    S != MOI.Parameter{Float64}
            ],
            Any[MOI.Utilities._try_constrain_variables_on_creation(
                dest,
                src,
                index_map,
                MOI.Parameter{Float64},
            )],
        )
    else
        Any[
            MOI.Utilities._try_constrain_variables_on_creation(
                dest,
                src,
                index_map,
                S,
            ) for S in MOI.Utilities.sorted_variable_sets_by_cost(dest, src)
        ]
    end
    MOI.Utilities._copy_free_variables(dest, index_map, vis_src)
    # Copy variable attributes
    MOI.Utilities.pass_attributes(dest, src, index_map, vis_src)
    # Copy model attributes
    MOI.Utilities.pass_attributes(dest, src, index_map)
    # Copy constraints
    MOI.Utilities._pass_constraints(dest, src, index_map, constraints_not_added)
    MOI.Utilities.final_touch(dest, index_map)
    return index_map
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
    if MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION &&
       model.evaluate_duals
        @warn "Dual solution not available, ignoring `evaluate_duals`"
    elseif model.evaluate_duals
        _compute_dual_of_parameters!(model)
    end
    return
end
