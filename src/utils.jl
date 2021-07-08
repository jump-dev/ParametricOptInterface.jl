next_variable_index!(model::ParametricOptimizer) = model.last_variable_index_added += 1
next_parameter_index!(model::ParametricOptimizer) = model.last_parameter_index_added += 1

function is_parameter_in_model(model::ParametricOptimizer, v::MOI.VariableIndex)
    return PARAMETER_INDEX_THRESHOLD < v.value <= model.last_parameter_index_added 
end

function is_variable_in_model(model::ParametricOptimizer, v::MOI.VariableIndex)
    return 0 < v.value <= model.last_variable_index_added 
end

function function_has_parameters(model::ParametricOptimizer, f::MOI.ScalarAffineFunction{T}) where T
    for term in f.terms
        if is_parameter_in_model(model, term.variable_index)
            return true
        end
    end
    return false
end
function function_has_parameters(model::ParametricOptimizer, f::MOI.VectorOfVariables)
    for variable in f.variables
        if is_parameter_in_model(model, variable)
            return true
        end
    end
    return false
end
function function_has_parameters(model::ParametricOptimizer, f::MOI.VectorAffineFunction{T}) where T
    for term in f.terms
        if is_parameter_in_model(model, term.variable_index)
            return true
        end
    end
    return false
end
function function_has_parameters(model::ParametricOptimizer, f::MOI.ScalarQuadraticFunction{T}) where T
    return function_affine_terms_has_parameters(model, f.affine_terms) ||
           function_quadratic_terms_has_parameters(model, f.quadratic_terms)
end
function function_affine_terms_has_parameters(model::ParametricOptimizer, affine_terms::Vector{MOI.ScalarAffineTerm{T}}) where T
    for term in affine_terms
        if is_parameter_in_model(model, term.variable_index)
            return true
        end
    end
    return false
end
function function_quadratic_terms_has_parameters(model::ParametricOptimizer, quadratic_terms::Vector{MOI.ScalarQuadraticTerm{T}}) where T
    for term in quadratic_terms
        if is_parameter_in_model(model, term.variable_index_1) || is_parameter_in_model(model, term.variable_index_2)
            return true
        end
    end
    return false
end

function separate_variables_from_parameters(model::ParametricOptimizer, terms::Vector{MOI.ScalarAffineTerm{T}}) where T
    vars = MOI.ScalarAffineTerm{T}[]
    params = MOI.ScalarAffineTerm{T}[]

    for term in terms
        if is_variable_in_model(model, term.variable_index)
            push!(vars, term)
        elseif is_parameter_in_model(model, term.variable_index)
            push!(params, term)
        else
            error("Constraint uses a variable that is not in the model")
        end
    end
    return vars, params
end

function calculate_param_constant(model::ParametricOptimizer, params::Vector{MOI.ScalarAffineTerm{T}}) where T
    param_constant = zero(T)
    for param in params
        param_constant += param.coefficient * model.parameters[param.variable_index]
    end
    return param_constant
end