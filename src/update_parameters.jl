function update_constant!(s::MOI.LessThan{T}, val) where {T}
    return MOI.LessThan{T}(s.upper - val)
end

function update_constant!(s::MOI.GreaterThan{T}, val) where {T}
    return MOI.GreaterThan{T}(s.lower - val)
end

function update_constant!(s::MOI.EqualTo{T}, val) where {T}
    return MOI.EqualTo{T}(s.value - val)
end

# Affine
function update_parameter_in_affine_constraints!(model::Optimizer)
    for S in SUPPORTED_SETS
        affine_constraint_cache_inner =
            model.affine_constraint_cache[MOI.ScalarAffineFunction{Float64}, S]
        if !isempty(affine_constraint_cache_inner)
            update_parameter_in_affine_constraints!(
                model.optimizer,
                model.parameters,
                model.updated_parameters,
                affine_constraint_cache_inner,
            )
        end
    end
    return model
end

function update_parameter_in_affine_constraints!(
    optimizer::OT,
    parameters::MOI.Utilities.CleverDicts.CleverDict{
        ParameterIndex,
        T,
        typeof(MOI.Utilities.CleverDicts.key_to_index),
        typeof(MOI.Utilities.CleverDicts.index_to_key),
    },
    updated_parameters::MOI.Utilities.CleverDicts.CleverDict{
        ParameterIndex,
        T,
        typeof(MOI.Utilities.CleverDicts.key_to_index),
        typeof(MOI.Utilities.CleverDicts.index_to_key),
    },
    affine_constraint_cache_inner::MOI.Utilities.DoubleDicts.DoubleDictInner{
        F,
        S,
        V,
    },
) where {OT,T,F,S,V}
    for (ci, param_array) in affine_constraint_cache_inner
        update_parameter_in_affine_constraints!(
            optimizer,
            ci,
            param_array,
            parameters,
            updated_parameters,
        )
    end
    return optimizer
end

function update_parameter_in_affine_constraints!(
    optimizer::OT,
    ci::CI,
    param_array::Vector{MOI.ScalarAffineTerm{T}},
    parameters::MOI.Utilities.CleverDicts.CleverDict{
        ParameterIndex,
        T,
        typeof(MOI.Utilities.CleverDicts.key_to_index),
        typeof(MOI.Utilities.CleverDicts.index_to_key),
    },
    updated_parameters::MOI.Utilities.CleverDicts.CleverDict{
        ParameterIndex,
        T,
        typeof(MOI.Utilities.CleverDicts.key_to_index),
        typeof(MOI.Utilities.CleverDicts.index_to_key),
    },
) where {OT,T,CI}
    param_constant = zero(T)
    for term in param_array
        if !isnan(updated_parameters[p_idx(term.variable)])
            param_constant +=
                term.coefficient *
                (updated_parameters[p_idx(term.variable)] - parameters[p_idx(term.variable)])
        end
    end
    if param_constant != zero(Float64)
        set = MOI.get(optimizer, MOI.ConstraintSet(), ci)
        set = update_constant!(set, param_constant)
        MOI.set(optimizer, MOI.ConstraintSet(), ci, set)
    end
    return ci
end

function update_parameter_in_affine_objective!(model::Optimizer)
    if !isempty(model.affine_objective_cache)
        objective_constant = zero(Float64)
        for term in model.affine_objective_cache
            if !isnan(model.updated_parameters[p_idx(term.variable)])
                param_old = model.parameters[p_idx(term.variable)]
                param_new = model.updated_parameters[p_idx(term.variable)]
                aux = param_new - param_old
                objective_constant += term.coefficient * aux
            end
        end
        if objective_constant != zero(Float64)
            F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
            f = MOI.get(model.optimizer, MOI.ObjectiveFunction{F}())
            fvar = MOI.ScalarAffineFunction(
                f.terms,
                f.constant + objective_constant,
            )
            MOI.set(model.optimizer, MOI.ObjectiveFunction{F}(), fvar)
        end
    end
    return model
end

function update_parameter_in_quadratic_constraints_pc!(model::Optimizer)
    for (ci, fparam) in model.quadratic_constraint_cache_pc
        param_constant = zero(Float64)
        for term in fparam
            if !isnan(model.updated_parameters[p_idx(term.variable)])
                param_old = model.parameters[p_idx(term.variable)]
                param_new = model.updated_parameters[p_idx(term.variable)]
                aux = param_new - param_old
                param_constant += term.coefficient * aux
            end
        end
        if param_constant != zero(Float64)
            set = MOI.get(
                model.optimizer,
                MOI.ConstraintSet(),
                model.quadratic_added_cache[ci],
            )
            set = update_constant!(set, param_constant)
            MOI.set(
                model.optimizer,
                MOI.ConstraintSet(),
                model.quadratic_added_cache[ci],
                set,
            )
        end
    end
end

function update_parameter_in_quadratic_objective_pc!(model::Optimizer)
    if !isempty(model.quadratic_objective_cache_pc)
        objective_constant = zero(Float64)
        for term in model.quadratic_objective_cache_pc
            if !isnan(model.updated_parameters[p_idx(term.variable)])
                param_old = model.parameters[p_idx(term.variable)]
                param_new = model.updated_parameters[p_idx(term.variable)]
                aux = param_new - param_old
                objective_constant += term.coefficient * aux
            end
        end
        if objective_constant != zero(Float64)
            F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
            f = MOI.get(model.optimizer, MOI.ObjectiveFunction{F}())

            # TODO
            # Is there another way to verify the Type of F without expliciting {Float64}?
            # Something like isa(F, MOI.ScalarAffineFunction)
            fvar = if F == MathOptInterface.ScalarAffineFunction{Float64}
                MOI.ScalarAffineFunction(f.terms, f.constant + objective_constant)
            else
                MOI.ScalarQuadraticFunction(
                    f.quadratic_terms,
                    f.affine_terms,
                    f.constant + objective_constant,
                )
            end
            MOI.set(model.optimizer, MOI.ObjectiveFunction{F}(), fvar)
        end
    end
    return model
end

function update_parameter_in_quadratic_constraints_pp!(model::Optimizer)
    for (ci, fparam) in model.quadratic_constraint_cache_pp
        param_constant = zero(Float64)
        for term in fparam
            if !isnan(model.updated_parameters[p_idx(term.variable_1)]) &&
               !isnan(model.updated_parameters[p_idx(term.variable_2)])
                param_constant += term.coefficient *
                         (
                             (model.updated_parameters[p_idx(term.variable_1)] * model.updated_parameters[p_idx(term.variable_2)]) - 
                             (model.parameters[p_idx(term.variable_1)] * model.parameters[p_idx(term.variable_2)])
                         )
             elseif !isnan(model.updated_parameters[p_idx(term.variable_1)])
                param_constant += term.coefficient * 
                         model.parameters[p_idx(term.variable_2)] * 
                         (model.updated_parameters[p_idx(term.variable_1)] - model.parameters[p_idx(term.variable_1)])
             elseif !isnan(model.updated_parameters[p_idx(term.variable_2)])
                param_constant +=
                     term.coefficient * model.parameters[p_idx(term.variable_1)] * (model.updated_parameters[p_idx(term.variable_2)] - model.parameters[p_idx(term.variable_2)])
             end
        end
        if param_constant != zero(Float64)
            set = MOI.get(
                model.optimizer,
                MOI.ConstraintSet(),
                model.quadratic_added_cache[ci],
            )
            set = update_constant!(set, param_constant)
            MOI.set(
                model.optimizer,
                MOI.ConstraintSet(),
                model.quadratic_added_cache[ci],
                set,
            )
        end
    end
end

function update_parameter_in_quadratic_objective_pp!(model::Optimizer)
    if !isempty(model.quadratic_objective_cache_pp)
        objective_constant = zero(Float64)
        for term in model.quadratic_objective_cache_pp
            if !isnan(model.updated_parameters[p_idx(term.variable_1)]) &&
               !isnan(model.updated_parameters[p_idx(term.variable_2)])
                objective_constant += term.coefficient *
                        (
                            (model.updated_parameters[p_idx(term.variable_1)] * model.updated_parameters[p_idx(term.variable_2)]) - 
                            (model.parameters[p_idx(term.variable_1)] * model.parameters[p_idx(term.variable_2)])
                        )
            elseif !isnan(model.updated_parameters[p_idx(term.variable_1)])
                objective_constant += term.coefficient * 
                        model.parameters[p_idx(term.variable_2)] * 
                        (model.updated_parameters[p_idx(term.variable_1)] - model.parameters[p_idx(term.variable_1)])
            elseif !isnan(model.updated_parameters[p_idx(term.variable_2)])
                objective_constant +=
                    term.coefficient * model.parameters[p_idx(term.variable_1)] * (model.updated_parameters[p_idx(term.variable_2)] - model.parameters[p_idx(term.variable_2)])
            end
        end
        if objective_constant != zero(Float64)
            F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
            f = MOI.get(model.optimizer, MOI.ObjectiveFunction{F}())

            # TODO
            # Is there another way to verify the Type of F without expliciting {Float64}?
            # Something like isa(F, MOI.ScalarAffineFunction)
            fvar = if F == MathOptInterface.ScalarAffineFunction{Float64}
                MOI.ScalarAffineFunction(f.terms, f.constant + objective_constant)
            else
                MOI.ScalarQuadraticFunction(
                    f.quadratic_terms,
                    f.affine_terms,
                    f.constant + objective_constant,
                )
            end
            MOI.set(model.optimizer, MOI.ObjectiveFunction{F}(), fvar)
        end
    end
end

function update_parameter_in_quadratic_constraints_pv!(model::Optimizer)
    for (ci, quad_aff_vars) in model.quadratic_constraint_cache_pv
        # We need this dictionary because we could have terms like
        # p_1 * v_1 + p_2 * v_1 and we should add p_1 and p_2 on the update
        new_coeff_per_variable = Dict{MOI.VariableIndex,Float64}()
        for term in quad_aff_vars
            # Here we use the convention that the parameter always comes as the first variables
            # in the caches
            if !isnan(model.updated_parameters[p_idx(term.variable_1)])
                coef = term.coefficient
                param_new = model.updated_parameters[p_idx(term.variable_1)]
                if haskey(new_coeff_per_variable, term.variable_2)
                    new_coeff_per_variable[term.variable_2] += param_new * coef
                else
                    new_coeff_per_variable[term.variable_2] = param_new * coef
                end
            end
        end
        if haskey(
            model.quadratic_constraint_variables_associated_to_parameters_cache,
            ci,
        )
            for aff_term in
                model.quadratic_constraint_variables_associated_to_parameters_cache[ci]
                coef = aff_term.coefficient
                if haskey(new_coeff_per_variable, aff_term.variable)
                    new_coeff_per_variable[aff_term.variable] += coef
                else
                    new_coeff_per_variable[aff_term.variable] = coef
                end
            end
        end
        old_ci = model.quadratic_added_cache[ci]
        for (vi, value) in new_coeff_per_variable
            MOI.modify(
                model.optimizer,
                old_ci,
                MOI.ScalarCoefficientChange(vi, value),
            )
        end
    end
    return model
end

function update_parameter_in_quadratic_objective_pv!(model::Optimizer)
    # We need this dictionary because we could have terms like
    # p_1 * v_1 + p_2 * v_1 and we should add p_1 and p_2 on the update
    new_coeff_per_variable = Dict{MOI.VariableIndex,Float64}()
    for term in model.quadratic_objective_cache_pv
        # Here we use the convention that the parameter always comes as the first variables
        # in the caches
        if !isnan(model.updated_parameters[p_idx(term.variable_1)])
            coef = term.coefficient
            param_new = model.updated_parameters[p_idx(term.variable_1)]
            if haskey(new_coeff_per_variable, term.variable_2)
                new_coeff_per_variable[term.variable_2] += param_new * coef
            else
                new_coeff_per_variable[term.variable_2] = param_new * coef
            end
        end
    end

    for aff_term in
        model.quadratic_objective_variables_associated_to_parameters_cache
        coef = aff_term.coefficient
        if haskey(new_coeff_per_variable, aff_term.variable)
            new_coeff_per_variable[aff_term.variable] += coef
        else
            new_coeff_per_variable[aff_term.variable] = coef
        end
    end

    F_pv = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
    for (vi, value) in new_coeff_per_variable
        MOI.modify(
            model.optimizer,
            MOI.ObjectiveFunction{F_pv}(),
            MOI.ScalarCoefficientChange(vi, value),
        )
    end
    return model
end

# Vector Affine
function update_parameter_in_vector_affine_constraints!(model::Optimizer)
    for S in SUPPORTED_VECTOR_SETS
        vector_constraint_cache_inner =
            model.vector_constraint_cache[MOI.VectorAffineFunction{Float64}, S]
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
    parameters::MOI.Utilities.CleverDicts.CleverDict{
        ParameterIndex,
        T,
        typeof(MOI.Utilities.CleverDicts.key_to_index),
        typeof(MOI.Utilities.CleverDicts.index_to_key),
    },
    updated_parameters::MOI.Utilities.CleverDicts.CleverDict{
        ParameterIndex,
        T,
        typeof(MOI.Utilities.CleverDicts.key_to_index),
        typeof(MOI.Utilities.CleverDicts.index_to_key),
    },
    vector_constraint_cache_inner::MOI.Utilities.DoubleDicts.DoubleDictInner{
        F,
        S,
        V,
    },
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
    parameters::MOI.Utilities.CleverDicts.CleverDict{
        ParameterIndex,
        T,
        typeof(MOI.Utilities.CleverDicts.key_to_index),
        typeof(MOI.Utilities.CleverDicts.index_to_key),
    },
    updated_parameters::MOI.Utilities.CleverDicts.CleverDict{
        ParameterIndex,
        T,
        typeof(MOI.Utilities.CleverDicts.key_to_index),
        typeof(MOI.Utilities.CleverDicts.index_to_key),
    },
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
    update_parameter_in_affine_constraints!(model)
    update_parameter_in_affine_objective!(model)
    update_parameter_in_quadratic_constraints_pc!(model)
    update_parameter_in_quadratic_objective_pc!(model)
    update_parameter_in_quadratic_constraints_pv!(model)
    update_parameter_in_quadratic_objective_pv!(model)
    update_parameter_in_quadratic_constraints_pp!(model)
    update_parameter_in_quadratic_objective_pp!(model)
    update_parameter_in_vector_affine_constraints!(model)

    # Update parameters and put NaN to indicate that the parameter has been updated
    for (parameter_index, val) in model.updated_parameters
        if !isnan(val) 
            model.parameters[parameter_index] = val
            model.updated_parameters[parameter_index] = NaN
        end
    end

    return model
end
