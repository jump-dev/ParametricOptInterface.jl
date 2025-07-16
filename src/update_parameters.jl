# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function _set_with_new_constant(s::MOI.LessThan{T}, val::T) where {T}
    return MOI.LessThan{T}(s.upper - val)
end

function _set_with_new_constant(s::MOI.GreaterThan{T}, val::T) where {T}
    return MOI.GreaterThan{T}(s.lower - val)
end

function _set_with_new_constant(s::MOI.EqualTo{T}, val::T) where {T}
    return MOI.EqualTo{T}(s.value - val)
end

function _set_with_new_constant(s::MOI.Interval{T}, val::T) where {T}
    return MOI.Interval{T}(s.lower - val, s.upper - val)
end

# Affine
# change to use only inner_ci all around so tha tupdates are faster
# modifications should not be used any ways, afterall we have param all around
function _update_affine_constraints!(model::Optimizer)
    for (F, S) in keys(model.affine_constraint_cache.dict)
        affine_constraint_cache_inner = model.affine_constraint_cache[F, S]
        affine_constraint_cache_set_inner =
            model.affine_constraint_cache_set[F, S]
        if !isempty(affine_constraint_cache_inner)
            # barrier to avoid type instability of inner dicts
            _update_affine_constraints!(
                model,
                affine_constraint_cache_inner,
                affine_constraint_cache_set_inner,
            )
        end
    end
    return
end

# TODO: cache changes and then batch them instead

function _update_affine_constraints!(
    model::Optimizer,
    affine_constraint_cache_inner::DoubleDictInner{F,S,V},
    affine_constraint_cache_set_inner::DoubleDictInner{
        F,
        S,
        MOI.AbstractScalarSet,
    },
) where {F,S<:SIMPLE_SCALAR_SETS{T},V} where {T}
    # cis = MOI.ConstraintIndex{F,S}[]
    # sets = S[]
    # sizehint!(cis, length(affine_constraint_cache_inner))
    # sizehint!(sets, length(affine_constraint_cache_inner))
    for (inner_ci, pf) in affine_constraint_cache_inner
        delta_constant = _delta_parametric_constant(model, pf)
        if !iszero(delta_constant)
            pf.current_constant += delta_constant
            new_set = S(pf.set_constant - pf.current_constant)
            # new_set = _set_with_new_constant(set, param_constant)
            MOI.set(model.optimizer, MOI.ConstraintSet(), inner_ci, new_set)
            # push!(cis, inner_ci)
            # push!(sets, new_set)
        end
    end
    # if !isempty(cis)
    #     MOI.set(model.optimizer, MOI.ConstraintSet(), cis, sets)
    # end
    return
end

function _update_affine_constraints!(
    model::Optimizer,
    affine_constraint_cache_inner::DoubleDictInner{F,S,V},
    affine_constraint_cache_set_inner::DoubleDictInner{
        F,
        S,
        MOI.AbstractScalarSet,
    },
) where {F,S<:MOI.Interval{T},V} where {T}
    for (inner_ci, pf) in affine_constraint_cache_inner
        delta_constant = _delta_parametric_constant(model, pf)
        if !iszero(delta_constant)
            pf.current_constant += delta_constant
            # new_set = S(pf.set_constant - pf.current_constant)
            set = affine_constraint_cache_set_inner[inner_ci]::S
            new_set = _set_with_new_constant(set, pf.current_constant)::S
            MOI.set(model.optimizer, MOI.ConstraintSet(), inner_ci, new_set)
        end
    end
    return
end

function _update_vector_affine_constraints!(model::Optimizer)
    for (F, S) in keys(model.vector_affine_constraint_cache.dict)
        vector_affine_constraint_cache_inner =
            model.vector_affine_constraint_cache[F, S]
        if !isempty(vector_affine_constraint_cache_inner)
            # barrier to avoid type instability of inner dicts
            _update_vector_affine_constraints!(
                model,
                vector_affine_constraint_cache_inner,
            )
        end
    end
    return
end

function _update_vector_affine_constraints!(
    model::Optimizer,
    vector_affine_constraint_cache_inner::DoubleDictInner{F,S,V},
) where {F<:MOI.VectorAffineFunction{T},S,V} where {T}
    for (inner_ci, pf) in vector_affine_constraint_cache_inner
        delta_constant = _delta_parametric_constant(model, pf)
        if !iszero(delta_constant)
            pf.current_constant .+= delta_constant
            MOI.modify(
                model.optimizer,
                inner_ci,
                MOI.VectorConstantChange(pf.current_constant),
            )
        end
    end
    return
end

function _update_quadratic_constraints!(model::Optimizer)
    for (F, S) in keys(model.quadratic_constraint_cache.dict)
        quadratic_constraint_cache_inner =
            model.quadratic_constraint_cache[F, S]
        quadratic_constraint_cache_set_inner =
            model.quadratic_constraint_cache_set[F, S]
        if !isempty(quadratic_constraint_cache_inner)
            # barrier to avoid type instability of inner dicts
            _update_quadratic_constraints!(
                model,
                quadratic_constraint_cache_inner,
                quadratic_constraint_cache_set_inner,
            )
        end
    end
    return
end

function _affine_build_change_and_up_param_func(
    pf::ParametricQuadraticFunction{T},
    delta_terms,
) where {T}
    changes = Vector{MOI.ScalarCoefficientChange}(undef, length(delta_terms))
    i = 1
    for (var, coef) in delta_terms
        base_coef = pf.current_terms_with_p[var]
        new_coef = base_coef + coef
        pf.current_terms_with_p[var] = new_coef
        changes[i] = MOI.ScalarCoefficientChange(var, new_coef)
        i += 1
    end
    return changes
end

function _update_quadratic_constraints!(
    model::Optimizer,
    quadratic_constraint_cache_inner::DoubleDictInner{F,S,V},
    quadratic_constraint_cache_set_inner::DoubleDictInner{
        F,
        S,
        MOI.AbstractScalarSet,
    },
) where {F,S<:SIMPLE_SCALAR_SETS{T},V} where {T}
    # cis = MOI.ConstraintIndex{F,S}[]
    # sets = S[]
    # sizehint!(cis, length(quadratic_constraint_cache_inner))
    # sizehint!(sets, length(quadratic_constraint_cache_inner))
    for (inner_ci, pf) in quadratic_constraint_cache_inner
        delta_constant = _delta_parametric_constant(model, pf)
        if !iszero(delta_constant)
            pf.current_constant += delta_constant
            new_set = S(pf.set_constant - pf.current_constant)
            # new_set = _set_with_new_constant(set, param_constant)
            MOI.set(model.optimizer, MOI.ConstraintSet(), inner_ci, new_set)
            # push!(cis, inner_ci)
            # push!(sets, new_set)
        end
        delta_terms = _delta_parametric_affine_terms(model, pf)
        if !isempty(delta_terms)
            changes = _affine_build_change_and_up_param_func(pf, delta_terms)
            cis = fill(inner_ci, length(changes))
            MOI.modify(model.optimizer, cis, changes)
        end
    end
    # if !isempty(cis)
    #     MOI.set(model.optimizer, MOI.ConstraintSet(), cis, sets)
    # end
    return
end

function _update_quadratic_constraints!(
    model::Optimizer,
    quadratic_constraint_cache_inner::DoubleDictInner{F,S,V},
    quadratic_constraint_cache_set_inner::DoubleDictInner{
        F,
        S,
        MOI.AbstractScalarSet,
    },
) where {F,S<:MOI.Interval{T},V} where {T}
    for (inner_ci, pf) in quadratic_constraint_cache_inner
        delta_constant = _delta_parametric_constant(model, pf)
        if !iszero(delta_constant)
            pf.current_constant += delta_constant
            # new_set = S(pf.set_constant - pf.current_constant)
            set = quadratic_constraint_cache_set_inner[inner_ci]::S
            new_set = _set_with_new_constant(set, pf.current_constant)::S
            MOI.set(model.optimizer, MOI.ConstraintSet(), inner_ci, new_set)
        end
        delta_terms = _delta_parametric_affine_terms(model, pf)
        if !isempty(delta_terms)
            changes = _affine_build_change_and_up_param_func(pf, delta_terms)
            cis = fill(inner_ci, length(changes))
            MOI.modify(model.optimizer, cis, changes)
        end
    end
    return
end

function _update_affine_objective!(model::Optimizer{T}) where {T}
    if model.affine_objective_cache === nothing
        return
    end
    pf = model.affine_objective_cache
    delta_constant = _delta_parametric_constant(model, pf)
    if !iszero(delta_constant)
        pf.current_constant += delta_constant
        # F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
        MOI.modify(
            model.optimizer,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
            MOI.ScalarConstantChange(pf.current_constant),
        )
    end
    return
end

function _update_quadratic_objective!(model::Optimizer{T}) where {T}
    if model.quadratic_objective_cache === nothing
        return
    end
    pf = model.quadratic_objective_cache
    delta_constant = _delta_parametric_constant(model, pf)
    if !iszero(delta_constant)
        pf.current_constant += delta_constant
        F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
        MOI.modify(
            model.optimizer,
            MOI.ObjectiveFunction{F}(),
            MOI.ScalarConstantChange(pf.current_constant),
        )
    end
    delta_terms = _delta_parametric_affine_terms(model, pf)
    if !isempty(delta_terms)
        F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
        changes = _affine_build_change_and_up_param_func(pf, delta_terms)
        MOI.modify(model.optimizer, MOI.ObjectiveFunction{F}(), changes)
    end
    return
end

function update_parameters!(model::Optimizer)
    _update_affine_constraints!(model)
    _update_vector_affine_constraints!(model)
    _update_quadratic_constraints!(model)
    _update_affine_objective!(model)
    _update_quadratic_objective!(model)
    _update_vector_quadratic_constraints!(model)

    # Update parameters and put NaN to indicate that the parameter has been
    # updated
    for (parameter_index, val) in model.updated_parameters
        if !isnan(val)
            model.parameters[parameter_index] = val
            model.updated_parameters[parameter_index] = NaN
        end
    end

    return
end

function _update_vector_quadratic_constraints!(model::Optimizer)
    for (F, S) in keys(model.vector_quadratic_constraint_cache.dict)
        vector_quadratic_constraint_cache_inner =
            model.vector_quadratic_constraint_cache[F, S]
        if !isempty(vector_quadratic_constraint_cache_inner)
            _update_vector_quadratic_constraints!(
                model,
                vector_quadratic_constraint_cache_inner,
            )
        end
    end
    return
end

function _update_vector_quadratic_constraints!(
    model::Optimizer,
    vector_quadratic_constraint_cache_inner::DoubleDictInner{F,S,V},
) where {F<:MOI.VectorQuadraticFunction{T},S,V} where {T}
    for (inner_ci, pf) in vector_quadratic_constraint_cache_inner
        delta_constant = _delta_parametric_constant(model, pf)
        if !iszero(sum(abs, delta_constant))
            pf.current_constant .+= delta_constant
            MOI.modify(
                model.optimizer,
                inner_ci,
                MOI.VectorConstantChange(pf.current_constant),
            )
        end
        # TODO: handle variable coefficients if needed
    end
    return
end