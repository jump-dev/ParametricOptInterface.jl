function MOI.add_constraint(
    optimizer::Optimizer,
    vi::MOI.VariableIndex,
    set::MOI.AbstractScalarSet,
)
    if isparameter(vi)
        error("Cannot constrain a parameter")
    end
    return MOI.add_constraint(inner_optimizer(optimizer), vi, set)
end

function MOI.add_constraint(
    optimizer::Optimizer,
    f::MOI.ScalarAffineFunction{T},
    set::MOI.AbstractScalarSet,
) where {T}
    if !function_has_parameters(f)
        return MOI.add_constraint(inner_optimizer(optimizer), f, set)
    elseif has_at_lest_two_variables(f)
        
    else # Only has one variable and will be added as a VariableIndex - in - Set

    end
end