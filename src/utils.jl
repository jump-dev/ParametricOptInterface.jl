function next_variable_index!(model::ParametricOptimizer)
    return model.last_variable_index_added += 1
end
function next_parameter_index!(model::ParametricOptimizer)
    return model.last_parameter_index_added += 1
end
function update_number_of_parameters!(model::ParametricOptimizer)
    return model.number_of_parameters_in_model += 1
end

function is_parameter_in_model(model::ParametricOptimizer, v::MOI.VariableIndex)
    return PARAMETER_INDEX_THRESHOLD <
           v.value <=
           model.last_parameter_index_added
end

function is_variable_in_model(model::ParametricOptimizer, v::MOI.VariableIndex)
    return 0 < v.value <= model.last_variable_index_added
end

function function_has_parameters(
    model::ParametricOptimizer,
    f::MOI.ScalarAffineFunction{T},
) where {T}
    for term in f.terms
        if is_parameter_in_model(model, term.variable_index)
            return true
        end
    end
    return false
end
function function_has_parameters(
    model::ParametricOptimizer,
    f::MOI.VectorOfVariables,
)
    for variable in f.variables
        if is_parameter_in_model(model, variable)
            return true
        end
    end
    return false
end
function function_has_parameters(
    model::ParametricOptimizer,
    f::MOI.VectorAffineFunction{T},
) where {T}
    for term in f.terms
        if is_parameter_in_model(model, term.variable_index)
            return true
        end
    end
    return false
end
function function_has_parameters(
    model::ParametricOptimizer,
    f::MOI.ScalarQuadraticFunction{T},
) where {T}
    return function_affine_terms_has_parameters(model, f.affine_terms) ||
           function_quadratic_terms_has_parameters(model, f.quadratic_terms)
end
function function_affine_terms_has_parameters(
    model::ParametricOptimizer,
    affine_terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    for term in affine_terms
        if is_parameter_in_model(model, term.variable_index)
            return true
        end
    end
    return false
end
function function_quadratic_terms_has_parameters(
    model::ParametricOptimizer,
    quadratic_terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}
    for term in quadratic_terms
        if is_parameter_in_model(model, term.variable_index_1) ||
           is_parameter_in_model(model, term.variable_index_2)
            return true
        end
    end
    return false
end

function separate_possible_terms_and_calculate_parameter_constant(
    model::ParametricOptimizer,
    terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    vars = MOI.ScalarAffineTerm{T}[]
    params = MOI.ScalarAffineTerm{T}[]
    param_constant = zero(T)

    for term in terms
        if is_variable_in_model(model, term.variable_index)
            push!(vars, term)
        elseif is_parameter_in_model(model, term.variable_index)
            push!(params, term)
            param_constant +=
                term.coefficient * model.parameters[term.variable_index]
        else
            error("Constraint uses a variable that is not in the model")
        end
    end
    return vars, params, param_constant
end
# This version is used on SQFs
function separate_possible_terms_and_calculate_parameter_constant(
    model::ParametricOptimizer,
    terms::Vector{MOI.ScalarAffineTerm{T}},
    variables_associated_to_parameters::Vector{MOI.VariableIndex},
) where {T}
    vars = MOI.ScalarAffineTerm{T}[]
    params = MOI.ScalarAffineTerm{T}[]
    terms_with_variables_associated_to_parameters = MOI.ScalarAffineTerm{T}[]
    param_constant = zero(T)

    for term in terms
        if is_variable_in_model(model, term.variable_index)
            push!(vars, term)
            if term.variable_index in variables_associated_to_parameters
                push!(terms_with_variables_associated_to_parameters, term)
            end
        elseif is_parameter_in_model(model, term.variable_index)
            push!(params, term)
            param_constant +=
                term.coefficient * model.parameters[term.variable_index]
        else
            error("Constraint uses a variable that is not in the model")
        end
    end
    return vars,
    params,
    terms_with_variables_associated_to_parameters,
    param_constant
end
function separate_possible_terms_and_calculate_parameter_constant(
    model::ParametricOptimizer,
    terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}
    quad_params = MOI.ScalarQuadraticTerm{T}[] # parameter x parameter
    quad_aff_vars = MOI.ScalarQuadraticTerm{T}[] # parameter x variable 
    quad_vars = MOI.ScalarQuadraticTerm{T}[] # variable x variable
    aff_terms = MOI.ScalarAffineTerm{T}[]
    variables_associated_to_parameters = MOI.VariableIndex[]
    quad_param_constant = zero(T)

    # TODO (guilherme) do we really need the convention ?
    # When we have a parameter x variable or a variable x parameter the convention is to rewrite
    # the SQT with parameter as variable_index_1 and variable as variable_index_2
    for term in terms
        if (
            is_variable_in_model(model, term.variable_index_1) &&
            is_variable_in_model(model, term.variable_index_2)
        )
            push!(quad_vars, term) # if there are only variables, it remains a quadratic term

        elseif (
            is_parameter_in_model(model, term.variable_index_1) &&
            is_variable_in_model(model, term.variable_index_2)
        )
            push!(quad_aff_vars, term)
            push!(variables_associated_to_parameters, term.variable_index_2)
            push!(model.multiplicative_parameters, term.variable_index_1.value)
            push!(
                aff_terms,
                MOI.ScalarAffineTerm(
                    term.coefficient * model.parameters[term.variable_index_1],
                    term.variable_index_2,
                ),
            )

        elseif (
            is_variable_in_model(model, term.variable_index_1) &&
            is_parameter_in_model(model, term.variable_index_2)
        )
            # Check convention defined above
            push!(
                quad_aff_vars,
                MOI.ScalarQuadraticTerm(
                    term.coefficient,
                    term.variable_index_2,
                    term.variable_index_1,
                ),
            )
            push!(variables_associated_to_parameters, term.variable_index_2)
            push!(
                aff_terms,
                MOI.ScalarAffineTerm(
                    term.coefficient * model.parameters[term.variable_index_2],
                    term.variable_index_1,
                ),
            )
            push!(model.multiplicative_parameters, term.variable_index_2.value)

        elseif (
            is_parameter_in_model(model, term.variable_index_1) &&
            is_parameter_in_model(model, term.variable_index_2)
        )
            push!(quad_params, term)
            push!(model.multiplicative_parameters, term.variable_index_1.value)
            push!(model.multiplicative_parameters, term.variable_index_2.value)
            quad_param_constant +=
                term.coefficient *
                model.parameters[term.variable_index_1] *
                model.parameters[term.variable_index_2]
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
    model::ParametricOptimizer,
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
