mutable struct ParametricOptInterfaceConfigurations
    evaluate_duals::Bool
    number_of_parameters_hint::Int
end

struct ParameterIndex
    value::Int
end

# All variables in parameters will be added with an index that is bigger than this number.
# In practice this means that if the MOI.VariableIndex value is bigger than this number 
# it is a parameter.
const VARIABLE_INDEX_THRESHOLD = 1_000_000_000_000_000_000

struct ParametersCache
    parameter_indexes::Vector{ParameterIndex}
    parameter_values::Vector{Float64}
    function ParametersCache(configs::ParametricOptInterfaceConfigurations)
        parameter_indexes = sizehint!(ParameterIndex[], configs.number_of_parameters_hint)
        parameter_values = sizehint!(Float64[], configs.number_of_parameters_hint)
        return new(
            parameter_indexes,
            parameter_values
        )
    end
end

function Base.isless(idx1::ParameterIndex, idx2::ParameterIndex)::Bool
    return idx1.value < idx2.value
end

function delete_parameter!(parameters_cache::ParametersCache, index::MOI.VariableIndex)
    idx_on_parameters = searchsortedfirst(parameters_cache.parameter_indexes, ParameterIndex(index.value - VARIABLE_INDEX_THRESHOLD))
    deleteat!(parameters_cache.parameter_indexes, idx_on_parameters)
    deleteat!(parameters_cache.parameter_values, idx_on_parameters)
    return nothing
end

"""
    Optimizer{T, OT <: MOI.ModelLike} <: MOI.AbstractOptimizer

Declares a `Optimizer`, which allows the handling of parameters in a
optimization model.

## Keyword arguments

- `evaluate_duals::Bool`: If `true`, evaluates the dual of parameters. Users might want to set it to `false`
  to increase performance when the duals of parameters are not necessary. Defaults to `true`.

- `number_of_parameters_hint`: Gives a hint to create caches of parameters in appropriate sizes. Defaults to `1`.

## Example

```julia-repl
julia> ParametricOptInterface.Optimizer(GLPK.Optimizer())
ParametricOptInterface.Optimizer{Float64,GLPK.Optimizer}
```
"""
mutable struct Optimizer{T,OT<:MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT
    configs::ParametricOptInterfaceConfigurations
    parameters_cache::ParametersCache
    function Optimizer(
        optimizer::OT;
        evaluate_duals::Bool = false,
        number_of_parameters_hint::Int = 1
    ) where {OT}
        configs = ParametricOptInterfaceConfigurations(
            evaluate_duals,
            number_of_parameters_hint
        )
        return new{Float64,OT}(
            optimizer,
            configs,
            ParametersCache(configs)
        )
    end
end

function inner_optimizer(optimizer::Optimizer)
    return optimizer.optimizer
end

function inner_parameters_cache(optimizer::Optimizer)
    return optimizer.parameters_cache
end

function MOI.get(optimizer::Optimizer, attr::MOI.AbstractModelAttribute)
    return MOI.get(inner_optimizer(optimizer), attr)
end

function MOI.supports_constraint(
    optimizer::Optimizer,
    F::Union{
        Type{MOI.VariableIndex},
        Type{MOI.ScalarAffineFunction{T}},
        Type{MOI.VectorOfVariables},
        Type{MOI.VectorAffineFunction{T}},
    },
    S::Type{<:MOI.AbstractSet},
) where {T}
    return MOI.supports_constraint(inner_optimizer(optimizer), F, S)
end

function MOI.supports_incremental_interface(optimizer::Optimizer)
    return MOI.supports_incremental_interface(inner_optimizer(optimizer))
end

struct NumberOfParameters <: MOI.AbstractModelAttribute end

function MOI.get(optimizer::Optimizer, ::NumberOfParameters)
    return length(inner_parameters_cache(optimizer).parameter_indexes)
end