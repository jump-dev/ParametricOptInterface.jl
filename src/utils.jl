function next_variable_index!(model::Optimizer)
    return model.last_variable_index_added += 1
end

function next_parameter_index!(model::Optimizer)
    return model.last_parameter_index_added += 1
end

function update_number_of_parameters!(model::Optimizer)
    return model.number_of_parameters_in_model += 1
end

function is_parameter_in_model(model::Optimizer, v::MOI.VariableIndex)
    return PARAMETER_INDEX_THRESHOLD <
           v.value <=
           model.last_parameter_index_added
end

function is_variable_in_model(model::Optimizer, v::MOI.VariableIndex)
    return 0 < v.value <= model.last_variable_index_added
end

function has_quadratic_constraint_caches(model::Optimizer)
    return !isempty(model.quadratic_added_cache)
end

function function_has_parameters(
    model::Optimizer,
    f::MOI.ScalarAffineFunction{T},
) where {T}
    for term in f.terms
        if is_parameter_in_model(model, term.variable)
            return true
        end
    end
    return false
end

function function_has_parameters(model::Optimizer, f::MOI.VectorOfVariables)
    for variable in f.variables
        if is_parameter_in_model(model, variable)
            return true
        end
    end
    return false
end

function function_has_parameters(
    model::Optimizer,
    f::MOI.VectorAffineFunction{T},
) where {T}
    for term in f.terms
        if is_parameter_in_model(model, term.scalar_term.variable)
            return true
        end
    end
    return false
end

function function_has_parameters(
    model::Optimizer,
    f::MOI.ScalarQuadraticFunction{T},
) where {T}
    return function_affine_terms_has_parameters(model, f.affine_terms) ||
           function_quadratic_terms_has_parameters(model, f.quadratic_terms)
end

function function_affine_terms_has_parameters(
    model::Optimizer,
    affine_terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    for term in affine_terms
        if is_parameter_in_model(model, term.variable)
            return true
        end
    end
    return false
end

function function_quadratic_terms_has_parameters(
    model::Optimizer,
    quadratic_terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}
    for term in quadratic_terms
        if is_parameter_in_model(model, term.variable_1) ||
           is_parameter_in_model(model, term.variable_2)
            return true
        end
    end
    return false
end

function count_scalar_affine_terms_types(
    model::Optimizer,
    terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    num_vars = count(x -> is_variable_in_model(model, x.variable), terms)
    num_params = length(terms) - num_vars
    return num_vars, num_params
end

function count_scalar_affine_terms_types(
    model::Optimizer,
    terms::Vector{MOI.ScalarAffineTerm{T}},
    variables_associated_to_parameters::Vector{MOI.VariableIndex},
) where {T}
    num_vars = 0
    num_params = 0
    num_vars_associated_to_params = 0
    for term in terms
        if is_variable_in_model(model, term.variable)
            num_vars += 1
            if term.variable in variables_associated_to_parameters
                num_vars_associated_to_params += 1
            end
        else
            num_params += 1
        end
    end
    return num_vars, num_params, num_vars_associated_to_params
end

function separate_possible_terms_and_calculate_parameter_constant(
    model::Optimizer,
    terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    num_vars, num_params = count_scalar_affine_terms_types(model, terms)
    vars = Vector{MOI.ScalarAffineTerm{T}}(undef, num_vars)
    params = Vector{MOI.ScalarAffineTerm{T}}(undef, num_params)
    param_constant = zero(T)
    i_vars = 1
    i_params = 1

    for term in terms
        if is_variable_in_model(model, term.variable)
            vars[i_vars] = term
            i_vars += 1
        elseif is_parameter_in_model(model, term.variable)
            params[i_params] = term
            param_constant += term.coefficient * model.parameters[p_idx(term.variable)]
            i_params += 1
        else
            error("Constraint uses a variable that is not in the model")
        end
    end
    return vars, params, param_constant
end

# This version is used on SQFs
function separate_possible_terms_and_calculate_parameter_constant(
    model::Optimizer,
    terms::Vector{MOI.ScalarAffineTerm{T}},
    variables_associated_to_parameters::Vector{MOI.VariableIndex},
) where {T}
    num_vars, num_params, num_vars_associated_to_params =
        count_scalar_affine_terms_types(
            model,
            terms,
            variables_associated_to_parameters,
        )
    vars = Vector{MOI.ScalarAffineTerm{T}}(undef, num_vars)
    params = Vector{MOI.ScalarAffineTerm{T}}(undef, num_params)
    terms_with_variables_associated_to_parameters =
        Vector{MOI.ScalarAffineTerm{T}}(undef, num_vars_associated_to_params)
    param_constant = zero(T)
    i_vars = 1
    i_params = 1
    i_vars_associated_to_params = 1

    for term in terms
        if is_variable_in_model(model, term.variable)
            vars[i_vars] = term
            if term.variable in variables_associated_to_parameters
                terms_with_variables_associated_to_parameters[i_vars_associated_to_params] =
                    term
                i_vars_associated_to_params += 1
            end
            i_vars += 1
        elseif is_parameter_in_model(model, term.variable)
            params[i_params] = term
            param_constant += term.coefficient * model.parameters[p_idx(term.variable)]
            i_params += 1
        else
            error("Constraint uses a variable that is not in the model")
        end
    end
    return vars,
    params,
    terms_with_variables_associated_to_parameters,
    param_constant
end

function count_scalar_quadratic_terms_types(
    model::Optimizer,
    terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}
    num_quad_vars = 0
    num_quad_params = 0
    num_quad_aff_vars = 0
    for term in terms
        if is_variable_in_model(model, term.variable_1) &&
           is_variable_in_model(model, term.variable_2)
            num_quad_vars += 1
        elseif is_variable_in_model(model, term.variable_1) &&
               is_parameter_in_model(model, term.variable_2)
            num_quad_aff_vars += 1
        elseif is_parameter_in_model(model, term.variable_1) &&
               is_variable_in_model(model, term.variable_2)
            num_quad_aff_vars += 1
        else
            num_quad_params += 1
        end
    end
    return num_quad_vars, num_quad_params, num_quad_aff_vars
end

function separate_possible_terms_and_calculate_parameter_constant(
    model::Optimizer,
    terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}
    num_quad_vars, num_quad_params, num_quad_aff_vars =
        count_scalar_quadratic_terms_types(model, terms)

    quad_params = Vector{MOI.ScalarQuadraticTerm{T}}(undef, num_quad_params) # parameter x parameter
    quad_aff_vars = Vector{MOI.ScalarQuadraticTerm{T}}(undef, num_quad_aff_vars) # parameter (as a variable) x variable
    quad_vars = Vector{MOI.ScalarQuadraticTerm{T}}(undef, num_quad_vars) # variable x variable
    aff_terms = Vector{MOI.ScalarAffineTerm{T}}(undef, num_quad_aff_vars) # parameter (as a number) x variable
    variables_associated_to_parameters =
        Vector{MOI.VariableIndex}(undef, num_quad_aff_vars)
    quad_param_constant = zero(T)

    i_quad_vars = 1
    i_quad_params = 1
    i_quad_aff_vars = 1

    # When we have a parameter x variable or a variable x parameter the convention is to rewrite
    # the SQT with parameter as variable_index_1 and variable as variable_index_2
    for term in terms
        if (
            is_variable_in_model(model, term.variable_1) &&
            is_variable_in_model(model, term.variable_2)
        )
            quad_vars[i_quad_vars] = term # if there are only variables, it remains a quadratic term
            i_quad_vars += 1
        elseif (
            is_parameter_in_model(model, term.variable_1) &&
            is_variable_in_model(model, term.variable_2)
        )
            quad_aff_vars[i_quad_aff_vars] = term
            variables_associated_to_parameters[i_quad_aff_vars] =
                term.variable_2
            aff_terms[i_quad_aff_vars] = MOI.ScalarAffineTerm(
                term.coefficient * model.parameters[p_idx(term.variable_1)],
                term.variable_2,
            )
            model.evaluate_duals &&
                push!(model.multiplicative_parameters, term.variable_1.value)
            i_quad_aff_vars += 1
        elseif (
            is_variable_in_model(model, term.variable_1) &&
            is_parameter_in_model(model, term.variable_2)
        )
            # Check convention defined above. We use the convention to know decide who is a variable and who is
            # a parameter withou having to recheck which is which.
            quad_aff_vars[i_quad_aff_vars] = MOI.ScalarQuadraticTerm(
                term.coefficient,
                term.variable_2,
                term.variable_1,
            )
            variables_associated_to_parameters[i_quad_aff_vars] =
                term.variable_1
            aff_terms[i_quad_aff_vars] = MOI.ScalarAffineTerm(
                term.coefficient * model.parameters[p_idx(term.variable_2)],
                term.variable_1,
            )
            model.evaluate_duals &&
                push!(model.multiplicative_parameters, term.variable_2.value)
            i_quad_aff_vars += 1
        elseif (
            is_parameter_in_model(model, term.variable_1) &&
            is_parameter_in_model(model, term.variable_2)
        )
            quad_params[i_quad_params] = term
            model.evaluate_duals &&
                push!(model.multiplicative_parameters, term.variable_1.value)
            model.evaluate_duals &&
                push!(model.multiplicative_parameters, term.variable_2.value)
            quad_param_constant +=
                term.coefficient *
                model.parameters[p_idx(term.variable_1)] *
                model.parameters[p_idx(term.variable_2)]
            i_quad_params += 1
        else
            throw(
                ErrorException(
                    "Constraint uses a variable or parameter that is not in the model",
                ),
            )
        end
    end
    return quad_vars,
    quad_aff_vars,
    quad_params,
    aff_terms,
    variables_associated_to_parameters,
    quad_param_constant
end

function fill_quadratic_constraint_caches!(
    model::Optimizer,
    new_ci::MOI.ConstraintIndex,
    quad_aff_vars::Vector{MOI.ScalarQuadraticTerm{T}},
    quad_params::Vector{MOI.ScalarQuadraticTerm{T}},
    aff_params::Vector{MOI.ScalarAffineTerm{T}},
    terms_with_variables_associated_to_parameters::Vector{
        MOI.ScalarAffineTerm{T},
    },
) where {T}
    if !isempty(quad_aff_vars)
        model.quadratic_constraint_cache_pv[new_ci] = quad_aff_vars
    end
    if !isempty(quad_params)
        model.quadratic_constraint_cache_pp[new_ci] = quad_params
    end
    if !isempty(aff_params)
        model.quadratic_constraint_cache_pc[new_ci] = aff_params
    end
    if !isempty(terms_with_variables_associated_to_parameters)
        model.quadratic_constraint_variables_associated_to_parameters_cache[new_ci] =
            terms_with_variables_associated_to_parameters
    end
    return nothing
end

function quadratic_constraint_cache_map_check(
    model::Optimizer,
    idx::MOI.ConstraintIndex{F,S},
) where {F,S}
    cached_constraints = values(model.quadratic_added_cache)
    # Using this becuase some custom brodcast method throws errors if
    # inner_idex .∈ cached_constraints is used
    return idx ∈ cached_constraints
end

# Vector Affine
function separate_possible_terms_and_calculate_parameter_constant(
    model::Optimizer,
    f::MOI.VectorAffineFunction{T},
    set::S,
) where {T,S<:MOI.AbstractVectorSet}
    vars = MOI.VectorAffineTerm{T}[]
    params = MOI.VectorAffineTerm{T}[]
    n_dims = length(f.constants)
    param_constants = zeros(T, n_dims)
    for term in f.terms
        oi = term.output_index

        if is_variable_in_model(model, term.scalar_term.variable)
            push!(vars, term)
        elseif is_parameter_in_model(model, term.scalar_term.variable)
            push!(params, term)
            param_constants[oi] +=
                term.scalar_term.coefficient *
                model.parameters[p_idx(term.scalar_term.variable)]
        else
            error("Constraint uses a variable that is not in the model")
        end
    end
    return vars, params, param_constants
end
