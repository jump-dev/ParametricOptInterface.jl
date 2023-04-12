# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function set_with_new_constant(s::MOI.LessThan{T}, val::T) where {T}
    return MOI.LessThan{T}(s.upper - val)
end

function set_with_new_constant(s::MOI.GreaterThan{T}, val::T) where {T}
    return MOI.GreaterThan{T}(s.lower - val)
end

function set_with_new_constant(s::MOI.EqualTo{T}, val::T) where {T}
    return MOI.EqualTo{T}(s.value - val)
end

function set_with_new_constant(s::MOI.Interval{T}, val::T) where {T}
    return MOI.Interval{T}(s.lower - val, s.upper - val)
end

# Affine
# change to use only inner_ci all around so tha tupdates are faster
# modifications should not be used any ways, afterall we have param all around
function update_parametric_affine_constraints!(model::Optimizer)
    for (F, S) in keys(model.affine_constraint_cache.dict)
        affine_constraint_cache_inner = model.affine_constraint_cache[F, S]
        affine_constraint_cache_set_inner =
            model.affine_constraint_cache_set[F, S]
        if !isempty(affine_constraint_cache_inner)
            # barrier to avoid type instability of inner dicts
            update_parametric_affine_constraints!(
                model,
                affine_constraint_cache_inner,
                affine_constraint_cache_set_inner,
            )
        end
    end
    return
end

# TODO: cache changes and then batch them instead

function update_parametric_affine_constraints!(
    model::Optimizer,
    affine_constraint_cache_inner::DoubleDictInner{F,S,V},
    affine_constraint_cache_set_inner::DoubleDictInner{F,S,MOI.AbstractScalarSet},
) where {F,S<:SIMPLE_SCALAR_SETS{T},V} where {T}
    # cis = MOI.ConstraintIndex{F,S}[]
    # sets = S[]
    # sizehint!(cis, length(affine_constraint_cache_inner))
    # sizehint!(sets, length(affine_constraint_cache_inner))
    for (inner_ci, pf) in affine_constraint_cache_inner
        delta_constant = delta_parametric_constant(model, pf)
        if !iszero(delta_constant)
            pf.current_constant += delta_constant
            new_set = S(pf.set_constant - pf.current_constant)
            # new_set = set_with_new_constant(set, param_constant)
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

function update_parametric_affine_constraints!(
    model::Optimizer,
    affine_constraint_cache_inner::DoubleDictInner{F,S,V},
    affine_constraint_cache_set_inner::DoubleDictInner{F,S,MOI.AbstractScalarSet},
) where {F,S<:MOI.Interval{T},V} where {T}
    for (inner_ci, pf) in affine_constraint_cache_inner
        set = affine_constraint_cache_set_inner[inner_ci]::S
        delta_constant = delta_parametric_constant(model, pf)
        if !iszero(delta_constant)
            pf.current_constant += delta_constant
            # new_set = S(pf.set_constant - pf.current_constant)
            new_set = set_with_new_constant(set, pf.current_constant)::S
            MOI.set(model.optimizer, MOI.ConstraintSet(), inner_ci, new_set)
        end
    end
    return
end

function update_parametric_quadratic_constraints!(model::Optimizer)
    for (F, S) in keys(model.quadratic_constraint_cache.dict)
        quadratic_constraint_cache_inner = model.quadratic_constraint_cache[F, S]
        quadratic_constraint_cache_set_inner =
            model.quadratic_constraint_cache_set[F, S]
        if !isempty(quadratic_constraint_cache_inner)
            # barrier to avoid type instability of inner dicts
            update_parametric_quadratic_constraints!(
                model,
                quadratic_constraint_cache_inner,
                quadratic_constraint_cache_set_inner,
            )
        end
    end
    return
end

function affine_build_change_and_up_param_func(
    pf::ParametricQuadraticFunction{T},
    delta_terms,
) where {T}
    changes = Vector{MOI.ScalarCoefficientChange}(
        undef,
        length(delta_terms),
    )
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

function update_parametric_quadratic_constraints!(
    model::Optimizer,
    quadratic_constraint_cache_inner::DoubleDictInner{F,S,V},
    quadratic_constraint_cache_set_inner::DoubleDictInner{F,S,MOI.AbstractScalarSet},
) where {F,S<:SIMPLE_SCALAR_SETS{T},V} where {T}
    # cis = MOI.ConstraintIndex{F,S}[]
    # sets = S[]
    # sizehint!(cis, length(quadratic_constraint_cache_inner))
    # sizehint!(sets, length(quadratic_constraint_cache_inner))
    for (inner_ci, pf) in quadratic_constraint_cache_inner
        delta_constant = delta_parametric_constant(model, pf)
        if !iszero(delta_constant)
            pf.current_constant += delta_constant
            new_set = S(pf.set_constant - pf.current_constant)
            # new_set = set_with_new_constant(set, param_constant)
            MOI.set(model.optimizer, MOI.ConstraintSet(), inner_ci, new_set)
            # push!(cis, inner_ci)
            # push!(sets, new_set)
        end
        delta_terms = delta_parametric_affine_terms(model, pf)
        if !isempty(delta_terms)
            changes = affine_build_change_and_up_param_func(pf, delta_terms)
            cis = fill(inner_ci, length(changes))
            MOI.modify(model.optimizer, cis, changes)
        end
    end
    # if !isempty(cis)
    #     MOI.set(model.optimizer, MOI.ConstraintSet(), cis, sets)
    # end
    return
end

function update_parametric_quadratic_constraints!(
    model::Optimizer,
    quadratic_constraint_cache_inner::DoubleDictInner{F,S,V},
    quadratic_constraint_cache_set_inner::DoubleDictInner{F,S,MOI.AbstractScalarSet},
) where {F,S<:MOI.Interval{T},V} where {T}
    for (inner_ci, pf) in quadratic_constraint_cache_inner
        set = quadratic_constraint_cache_set_inner[inner_ci]::S
        delta_constant = delta_parametric_constant(model, pf)
        if !iszero(delta_constant)
            pf.current_constant += delta_constant
            # new_set = S(pf.set_constant - pf.current_constant)
            new_set = set_with_new_constant(set, pf.current_constant)::S
            MOI.set(model.optimizer, MOI.ConstraintSet(), inner_ci, new_set)
        end
        delta_terms = delta_parametric_affine_terms(model, pf)
        if !isempty(delta_terms)
            changes = affine_build_change_and_up_param_func(pf, delta_terms)
            cis = fill(inner_ci, length(changes))
            MOI.modify(model.optimizer, cis, changes)
        end
    end
    return
end

function update_parametric_affine_objective!(model::Optimizer{T}) where {T}
    if model.affine_objective_cache === nothing
        return
    end
    pf = model.affine_objective_cache
    delta_constant = delta_parametric_constant(model, pf)
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

function update_parametric_quadratic_objective!(model::Optimizer{T}) where {T}
    if model.quadratic_objective_cache === nothing
        return
    end
    pf = model.quadratic_objective_cache
    delta_constant = delta_parametric_constant(model, pf)
    if !iszero(delta_constant)
        pf.current_constant += delta_constant
        F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
        MOI.modify(
            model.optimizer,
            MOI.ObjectiveFunction{F}(),
            MOI.ScalarConstantChange(pf.current_constant),
        )
    end
    delta_terms = delta_parametric_affine_terms(model, pf)
    if !isempty(delta_terms)
        F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
        changes = affine_build_change_and_up_param_func(pf, delta_terms)
        MOI.modify(model.optimizer, MOI.ObjectiveFunction{F}(), changes)
    end
    return
end

# TODO
# Vector Affine
function update_parameter_in_vector_affine_constraints!(model::Optimizer)
    for (F, S) in keys(model.vector_constraint_cache.dict)
        vector_constraint_cache_inner = model.vector_constraint_cache[F, S]
        if !isempty(vector_constraint_cache_inner)
            update_parameter_in_vector_affine_constraints!(
                model.optimizer,
                model.parameters,
                model.updated_parameters,
                vector_constraint_cache_inner,
            )
        end
    end
    return model
end

function update_parameter_in_vector_affine_constraints!(
    optimizer::OT,
    parameters::ParamTo{T},
    updated_parameters::ParamTo{T},
    vector_constraint_cache_inner::DoubleDictInner{F,S,V},
) where {OT,T,F,S,V}
    for (ci, param_array) in vector_constraint_cache_inner
        update_parameter_in_vector_affine_constraints!(
            optimizer,
            ci,
            param_array,
            parameters,
            updated_parameters,
        )
    end

    return optimizer
end

function update_parameter_in_vector_affine_constraints!(
    optimizer::OT,
    ci::CI,
    param_array::Vector{MOI.VectorAffineTerm{T}},
    parameters::ParamTo{T},
    updated_parameters::ParamTo{T},
) where {OT,T,CI}
    cf = MOI.get(optimizer, MOI.ConstraintFunction(), ci)

    n_dims = length(cf.constants)
    param_constants = zeros(T, n_dims)

    for term in param_array
        vi = term.scalar_term.variable

        if !isnan(updated_parameters[p_idx(vi)])
            param_constants[term.output_index] =
                term.scalar_term.coefficient *
                (updated_parameters[p_idx(vi)] - parameters[p_idx(vi)])
        end
    end

    if param_constants != zeros(T, n_dims)
        MOI.modify(
            optimizer,
            ci,
            MOI.VectorConstantChange(cf.constants + param_constants),
        )
    end

    return ci
end

function update_parameters!(model::Optimizer)
    update_parametric_affine_constraints!(model)
    update_parametric_quadratic_constraints!(model)
    update_parametric_affine_objective!(model)
    update_parametric_quadratic_objective!(model)
    # update_parameter_in_vector_affine_constraints!(model)

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
