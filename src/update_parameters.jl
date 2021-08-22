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
function update_parameter_in_affine_constraints!(model::ParametricOptimizer)
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
    parameters::Dict{MOI.VariableIndex,T},
    updated_parameters::Dict{MOI.VariableIndex,T},
    affine_constraint_cache_inner::MOI.Utilities.DoubleDicts.WithType{F,S},
) where {OT,T,F,S}
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
    parameters::Dict{MOI.VariableIndex,T},
    updated_parameters::Dict{MOI.VariableIndex,T},
) where {OT,T,CI}
    param_constant = zero(T)
    for term in param_array
        if haskey(updated_parameters, term.variable_index) # TODO This haskey can be slow
            param_constant +=
                term.coefficient * (
                    updated_parameters[term.variable_index] -
                    parameters[term.variable_index]
                )
        end
    end
    if param_constant != 0
        set = MOI.get(optimizer, MOI.ConstraintSet(), ci)
        set = update_constant!(set, param_constant)
        MOI.set(optimizer, MOI.ConstraintSet(), ci, set)
    end
    return ci
end

function update_parameters_in_affine_objective!(model::ParametricOptimizer)
    if !isempty(model.affine_objective_cache)
        objective_constant = 0
        for j in model.affine_objective_cache
            if haskey(model.updated_parameters, j.variable_index)
                param_old = model.parameters[j.variable_index]
                param_new = model.updated_parameters[j.variable_index]
                aux = param_new - param_old
                objective_constant += j.coefficient * aux
            end
        end
        if objective_constant != 0
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

function update_parameter_in_quadratic_constraints_pc!(
    model::ParametricOptimizer,
)
    for (ci, fparam) in model.quadratic_constraint_cache_pc
        param_constant = 0
        for j in fparam
            if haskey(model.updated_parameters, j.variable_index)
                param_old = model.parameters[j.variable_index]
                param_new = model.updated_parameters[j.variable_index]
                aux = param_new - param_old
                param_constant += j.coefficient * aux
            end
        end
        if param_constant != 0
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

function update_parameter_in_quadratic_objective_pc!(model::ParametricOptimizer)
    if !isempty(model.quadratic_objective_cache_pc)
        objective_constant = 0
        for j in model.quadratic_objective_cache_pc
            if haskey(model.updated_parameters, j.variable_index)
                param_old = model.parameters[j.variable_index]
                param_new = model.updated_parameters[j.variable_index]
                aux = param_new - param_old
                objective_constant += j.coefficient * aux
            end
        end
        if objective_constant != 0
            F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
            f = MOI.get(model.optimizer, MOI.ObjectiveFunction{F}())

            # TODO
            # Is there another way to verify the Type of F without expliciting {Float64}?
            # Something like isa(F, MOI.ScalarAffineFunction)
            fvar = if F == MathOptInterface.ScalarAffineFunction{Float64}
                MOI.ScalarAffineFunction(f.terms, f.constant + objective_constant)
            else
                MOI.ScalarQuadraticFunction(
                    f.affine_terms,
                    f.quadratic_terms,
                    f.constant + objective_constant,
                )
            end
            MOI.set(model.optimizer, MOI.ObjectiveFunction{F}(), fvar)
        end
    end
    return model
end

function update_parameter_in_quadratic_constraints_pp!(
    model::ParametricOptimizer,
)
    for (ci, fparam) in model.quadratic_constraint_cache_pp
        param_constant = 0
        for j in fparam
            if haskey(model.updated_parameters, j.variable_index_1) &&
               haskey(model.updated_parameters, j.variable_index_2)
                param_new_1 = model.updated_parameters[j.variable_index_1]
                param_new_2 = model.updated_parameters[j.variable_index_2]
                param_old_1 = model.parameters[j.variable_index_1]
                param_old_2 = model.parameters[j.variable_index_2]
                param_constant +=
                    j.coefficient *
                    ((param_new_1 * param_new_2) - (param_old_1 * param_old_2))
            elseif haskey(model.updated_parameters, j.variable_index_1)
                param_new_1 = model.updated_parameters[j.variable_index_1]
                param_old_1 = model.parameters[j.variable_index_1]
                param_old_2 = model.parameters[j.variable_index_2]
                param_constant +=
                    j.coefficient * param_old_2 * (param_new_1 - param_old_1)
            elseif haskey(model.updated_parameters, j.variable_index_2)
                param_new_2 = model.updated_parameters[j.variable_index_2]
                param_old_1 = model.parameters[j.variable_index_1]
                param_old_2 = model.parameters[j.variable_index_2]
                param_constant +=
                    j.coefficient * param_old_1 * (param_new_2 - param_old_2)
            end
        end
        if param_constant != 0
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

function update_parameter_in_quadratic_objective_pp!(model::ParametricOptimizer)
    if !isempty(model.quadratic_objective_cache_pp)
        objective_constant = 0
        for j in model.quadratic_objective_cache_pp
            if haskey(model.updated_parameters, j.variable_index_1) &&
               haskey(model.updated_parameters, j.variable_index_2)
                param_new_1 = model.updated_parameters[j.variable_index_1]
                param_new_2 = model.updated_parameters[j.variable_index_2]
                param_old_1 = model.parameters[j.variable_index_1]
                param_old_2 = model.parameters[j.variable_index_2]
                objective_constant +=
                    j.coefficient *
                    ((param_new_1 * param_new_2) - (param_old_1 * param_old_2))
            elseif haskey(model.updated_parameters, j.variable_index_1)
                param_new_1 = model.updated_parameters[j.variable_index_1]
                param_old_1 = model.parameters[j.variable_index_1]
                param_old_2 = model.parameters[j.variable_index_2]
                objective_constant +=
                    j.coefficient * param_old_2 * (param_new_1 - param_old_1)
            elseif haskey(model.updated_parameters, j.variable_index_2)
                param_new_2 = model.updated_parameters[j.variable_index_2]
                param_old_1 = model.parameters[j.variable_index_1]
                param_old_2 = model.parameters[j.variable_index_2]
                objective_constant +=
                    j.coefficient * param_old_1 * (param_new_2 - param_old_2)
            end
        end
        if objective_constant != 0
            F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
            f = MOI.get(model.optimizer, MOI.ObjectiveFunction{F}())

            # TODO
            # Is there another way to verify the Type of F without expliciting {Float64}?
            # Something like isa(F, MOI.ScalarAffineFunction)
            fvar = if F == MathOptInterface.ScalarAffineFunction{Float64}
                MOI.ScalarAffineFunction(f.terms, f.constant + objective_constant)
            else
                MOI.ScalarQuadraticFunction(
                    f.affine_terms,
                    f.quadratic_terms,
                    f.constant + objective_constant,
                )
            end
            MOI.set(model.optimizer, MOI.ObjectiveFunction{F}(), fvar)
        end
    end
end

# Vector Affine
function update_parameter_in_vector_affine_constraints!(
    model::ParametricOptimizer,
)
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
    parameters::Dict{MOI.VariableIndex,T},
    updated_parameters::Dict{MOI.VariableIndex,T},
    vector_constraint_cache_inner::MOI.Utilities.DoubleDicts.WithType{F,S},
) where {OT,T,F,S}
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
    parameters::Dict{MOI.VariableIndex,T},
    updated_parameters::Dict{MOI.VariableIndex,T},
) where {OT,T,CI}
    cf = MOI.get(optimizer, MOI.ConstraintFunction(), ci)

    n_dims = length(cf.constants)
    param_constants = zeros(T, n_dims)

    for term in param_array
        vi = term.scalar_term.variable_index

        if haskey(updated_parameters, vi) # TODO This haskey can be slow
            param_constants[term.output_index] =
                term.scalar_term.coefficient *
                (updated_parameters[vi] - parameters[vi])
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

function update_parameters!(model::ParametricOptimizer)
    update_parameter_in_affine_constraints!(model)
    update_parameters_in_affine_objective!(model)
    update_parameter_in_quadratic_constraints_pc!(model)
    update_parameter_in_quadratic_objective_pc!(model)
    update_parameter_in_quadratic_constraints_pp!(model)
    update_parameter_in_quadratic_objective_pp!(model)
    update_parameter_in_vector_affine_constraints!(model)

    # TODO make this part better
    constraint_aux_dict = Dict{Any,Any}()

    for (ci, fparam) in model.quadratic_constraint_cache_pv
        for j in fparam
            if haskey(model.updated_parameters, j.variable_index_1)
                coef = j.coefficient
                param_new = model.updated_parameters[j.variable_index_1]
                if haskey(constraint_aux_dict, (ci, j.variable_index_2))
                    constraint_aux_dict[(ci, j.variable_index_2)] +=
                        param_new * coef
                else
                    constraint_aux_dict[(ci, j.variable_index_2)] =
                        param_new * coef
                end
            end
        end
    end

    for (ci, fparam) in
        model.quadratic_constraint_variables_associated_to_parameters_cache
        for j in fparam
            coef = j.coefficient
            if haskey(constraint_aux_dict, (ci, j.variable_index))#
                constraint_aux_dict[(ci, j.variable_index)] += coef
            else
                constraint_aux_dict[(ci, j.variable_index)] = coef
            end
        end
    end

    for ((ci, vi), value) in constraint_aux_dict
        old_ci = model.quadratic_added_cache[ci]
        MOI.modify(
            model.optimizer,
            old_ci,
            MOI.ScalarCoefficientChange(vi, value),
        )
    end

    objective_aux_dict = Dict{Any,Any}()

    if !isempty(model.quadratic_objective_cache_pv)
        for j in model.quadratic_objective_cache_pv
            if haskey(model.updated_parameters, j.variable_index_1)
                coef = j.coefficient
                param_new = model.updated_parameters[j.variable_index_1]
                if haskey(objective_aux_dict, (j.variable_index_2))
                    objective_aux_dict[(j.variable_index_2)] += param_new * coef
                else
                    objective_aux_dict[(j.variable_index_2)] = param_new * coef
                end
            end
        end
    end

    for j in model.quadratic_objective_variables_associated_to_parameters_cache
        coef = j.coefficient
        if haskey(objective_aux_dict, j.variable_index)
            objective_aux_dict[j.variable_index] += coef
        else
            objective_aux_dict[j.variable_index] = coef
        end
    end

    F_pv = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())

    for (key, value) in objective_aux_dict
        MOI.modify(
            model.optimizer,
            MOI.ObjectiveFunction{F_pv}(),
            MOI.ScalarCoefficientChange(key, value),
        )
    end

    for (i, val) in model.updated_parameters
        model.parameters[i] = val
    end

    empty!(model.updated_parameters)

    return model
end
