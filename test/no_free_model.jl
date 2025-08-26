MOI.Utilities.@model(
    NoFreeVariablesModel,
    (),
    (),
    (MOI.Nonnegatives,),
    (),
    (),
    (),
    (MOI.VectorOfVariables,),
    (),
)

function MOI.supports_constraint(
    ::NoFreeVariablesModel,
    ::Type{MOI.VectorOfVariables},
    ::Type{MOI.Reals},
)
    return false
end

function MOI.supports_add_constrained_variable(
    ::NoFreeVariablesModel{T},
    ::Type{MOI.LessThan{T}},
) where {T}
    return true
end

function MOI.supports_add_constrained_variables(
    ::NoFreeVariablesModel,
    ::Type{MOI.Nonnegatives},
)
    return true
end

function MOI.supports_add_constrained_variables(
    ::NoFreeVariablesModel,
    ::Type{MOI.Reals},
)
    return false
end
