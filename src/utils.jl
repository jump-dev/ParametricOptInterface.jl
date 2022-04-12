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

function separate_possible_terms_and_calculate_parameter_constant(
    model::Optimizer,
    terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    param_constant = zero(T)
    vars = filter(x -> is_variable_in_model(model, x.variable), terms)
    params = filter(x -> is_parameter_in_model(model, x.variable), terms)

    if (length(vars) + length(params)) != length(terms)
        error("Function uses a variable index that is not in the model")
    end

    for param in params
        param_constant += param.coefficient * model.parameters[param.variable]
    end

    return vars, params, param_constant
end

# This version is used on SQFs
function separate_possible_terms_and_calculate_parameter_constant(
    model::Optimizer,
    terms::Vector{MOI.ScalarAffineTerm{T}},
    variables_associated_to_parameters::Vector{MOI.VariableIndex},
) where {T}
    vars = MOI.ScalarAffineTerm{T}[]
    params = MOI.ScalarAffineTerm{T}[]
    terms_with_variables_associated_to_parameters = MOI.ScalarAffineTerm{T}[]
    param_constant = zero(T)

    if function_affine_terms_has_parameters(model, terms)
        for term in terms
            if is_variable_in_model(model, term.variable)
                push!(vars, term)
                if term.variable in variables_associated_to_parameters
                    push!(terms_with_variables_associated_to_parameters, term)
                end
            elseif is_parameter_in_model(model, term.variable)
                push!(params, term)
                param_constant +=
                    term.coefficient * model.parameters[term.variable]
            else
                error("Constraint uses a variable that is not in the model")
            end
        end
    else
        vars = terms
    end
    return vars,
    params,
    terms_with_variables_associated_to_parameters,
    param_constant
end

function separate_possible_terms_and_calculate_parameter_constant(
    model::Optimizer,
    terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}

    quad_param_constant = zero(T)

    quad_vars = filter(x -> (is_variable_in_model(model, x.variable_1) && is_variable_in_model(model, x.variable_2)), terms)
    quad_params = filter(x -> (is_parameter_in_model(model, x.variable_1) && is_parameter_in_model(model, x.variable_2)), terms)

    # When we have a parameter x variable or a variable x parameter the convention is to rewrite
    # the SQT with parameter as variable_index_1 and variable as variable_index_2
    quad_aff_vars_p_v = filter(x -> (is_parameter_in_model(model, x.variable_1) && is_variable_in_model(model, x.variable_2)), terms)
    quad_aff_vars_v_p = filter(x -> (is_variable_in_model(model, x.variable_1) && is_parameter_in_model(model, x.variable_2)), terms)

    num_terms_p_v = length(quad_aff_vars_p_v)
    num_terms_v_p = length(quad_aff_vars_v_p)

    quad_aff_vars = Vector{MOI.ScalarQuadraticTerm{T}}(undef, num_terms_p_v + num_terms_v_p)
    aff_terms = Vector{MOI.ScalarAffineTerm{T}}(undef, num_terms_p_v + num_terms_v_p)
    variables_associated_to_parameters = Vector{MOI.VariableIndex}(undef, num_terms_p_v + num_terms_v_p)
    for (i, term_p_v) in enumerate(quad_aff_vars_p_v)
        quad_aff_vars[i] = term_p_v
        variables_associated_to_parameters[i] = term_p_v.variable_2
        aff_terms[i] = MOI.ScalarAffineTerm(
            term_p_v.coefficient * model.parameters[term_p_v.variable_1],
            term_p_v.variable_2,
        )
        model.evaluate_duals && push!(model.multiplicative_parameters, term_p_v.variable_1.value)
    end
    for (i, term_v_p) in enumerate(quad_aff_vars_v_p)
        idx = i + num_terms_p_v
        quad_aff_vars[idx] = MOI.ScalarQuadraticTerm(
            term_v_p.coefficient,
            term_v_p.variable_2,
            term_v_p.variable_1,
        )
        variables_associated_to_parameters[idx] = term_v_p.variable_1
        aff_terms[idx] = MOI.ScalarAffineTerm(
            term_v_p.coefficient * model.parameters[term_v_p.variable_2],
            term_v_p.variable_1,
        )
        model.evaluate_duals && push!(model.multiplicative_parameters, term_v_p.variable_1.value)
    end
    
    for term_p_p in quad_params
        model.evaluate_duals && push!(model.multiplicative_parameters, term_p_p.variable_1.value)
        model.evaluate_duals && push!(model.multiplicative_parameters, term_p_p.variable_2.value)
        quad_param_constant +=
                    term_p_p.coefficient *
                    model.parameters[term_p_p.variable_1] *
                    model.parameters[term_p_p.variable_2]
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
                model.parameters[term.scalar_term.variable]
        else
            error("Constraint uses a variable that is not in the model")
        end
    end
    return vars, params, param_constants
end
