function MOI.add_variable(model::Optimizer)
    return MOI.add_variable(inner_optimizer(model))
end

function MOI.add_constrained_variable(model::Optimizer, set::MOI.AbstractScalarSet)
    return MOI.add_constrained_variable(inner_optimizer(model), set)
end

function MOI.add_constrained_variable(model::Optimizer, set::MOI.Parameter)
    parameters_cache = inner_parameters_cache(model)
    new_number_of_parameters = length(parameters_cache.parameter_indexes) + 1
    push!(parameters_cache.parameter_indexes, ParameterIndex(new_number_of_parameters))
    push!(parameters_cache.parameter_values, MOI.constant(set))
    parameter_constraint_index = MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter}(
        new_number_of_parameters
    )
    return MOI.VariableIndex(new_number_of_parameters + VARIABLE_INDEX_THRESHOLD), parameter_constraint_index
end

function MOI.delete(optimizer::Optimizer, index::MOI.VariableIndex)
    if isvariable(index)
        MOI.delete(inner_optimizer(optimizer), index)
    else
        delete_parameter!(inner_parameters_cache(optimizer), index)
    end
    return nothing
end