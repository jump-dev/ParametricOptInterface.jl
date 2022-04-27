module ParametricOptInterface

using MathOptInterface

const MOI = MathOptInterface

const PARAMETER_INDEX_THRESHOLD = 1_000_000_000_000_000_000

"""
    Parameter(val::Float64)

The `Parameter` structure stores the numerical value associated to a given
parameter.

## Example

```julia-repl
julia> ParametricOptInterface.Parameter(5)
ParametricOptInterface.Parameter(5)
```
"""
struct Parameter <: MOI.AbstractScalarSet
    val::Float64
end

# Utilities for using a CleverDict in Parameters
struct ParameterIndex
    index::Int64
end
function MOI.Utilities.CleverDicts.index_to_key(
    ::Type{ParameterIndex},
    index::Int64,
)
    return ParameterIndex(index)
end
function MOI.Utilities.CleverDicts.key_to_index(key::ParameterIndex)
    return key.index
end
function p_idx(vi::MOI.VariableIndex)
    return ParameterIndex(vi.value - PARAMETER_INDEX_THRESHOLD)
end

"""
    Optimizer{T, OT <: MOI.ModelLike} <: MOI.AbstractOptimizer

Declares a `Optimizer`, which allows the handling of parameters in a
optimization model.

## Keyword arguments

- `evaluate_duals::Bool`: If `true`, evaluates the dual of parameters. Users might want to set it to false 
  to increase performance when the duals of parameters are not necessary. Defaults to `true`.

## Example

```julia-repl
julia> ParametricOptInterface.Optimizer(GLPK.Optimizer())
ParametricOptInterface.Optimizer{Float64,GLPK.Optimizer}
```
"""
mutable struct Optimizer{T,OT<:MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT
    parameters::MOI.Utilities.CleverDicts.CleverDict{
        ParameterIndex,
        T,
        typeof(MOI.Utilities.CleverDicts.key_to_index),
        typeof(MOI.Utilities.CleverDicts.index_to_key),
    }
    parameters_name::Dict{MOI.VariableIndex,String}
    # The updated_parameters dictionary has the same dimension of the 
    # parameters dictionary and if the value stored is a NaN is means
    # that the parameter has not been updated.
    updated_parameters::MOI.Utilities.CleverDicts.CleverDict{
        ParameterIndex,
        T,
        typeof(MOI.Utilities.CleverDicts.key_to_index),
        typeof(MOI.Utilities.CleverDicts.index_to_key),
    }
    variables::MOI.Utilities.CleverDicts.CleverDict{
        MOI.VariableIndex,
        MOI.VariableIndex,
        typeof(MOI.Utilities.CleverDicts.key_to_index),
        typeof(MOI.Utilities.CleverDicts.index_to_key),
    }
    last_variable_index_added::Int64
    last_parameter_index_added::Int64
    # Store the constraint function and set passed to POI by MOI
    original_constraint_function_and_set_cache::Dict{
        MOI.ConstraintIndex,
        Tuple{MOI.AbstractFunction,MOI.AbstractSet},
    }
    # Store reference to parameters of affine constraints with parameters: v + p
    affine_constraint_cache::MOI.Utilities.DoubleDicts.DoubleDict{
        Vector{MOI.ScalarAffineTerm{Float64}},
    }
    # Store reference quadratic constraints with parameter * variable constraints: p * v
    quadratic_constraint_cache_pv::MOI.Utilities.DoubleDicts.DoubleDict{
        Vector{MOI.ScalarQuadraticTerm{Float64}},
    }
    # Store reference quadratic constraints with parameter * variable constraints: p * v
    quadratic_constraint_cache_pp::MOI.Utilities.DoubleDicts.DoubleDict{
        Vector{MOI.ScalarQuadraticTerm{Float64}},
    }
    # Store reference to constraints with quad_variable_term + affine_with_parameters: v * v + p
    quadratic_constraint_cache_pc::MOI.Utilities.DoubleDicts.DoubleDict{
        Vector{MOI.ScalarAffineTerm{Float64}},
    }
    # Store the reference to variables in the scalar affine part that are 
    # multiplied by parameters in the scalar quadratic terms.
    # i.e.
    # If we have a constraint function with both scalar quadratic terms and 
    # scalar affine terms such as p_1 * v_1 + 2.0 * v_1
    # When we need to update the constraint coefficient after updating the parameter 
    # we must do (new_p_1 + 2.0) * v_1
    # This cache is storing the 2.0 * v_1 part.
    quadratic_constraint_variables_associated_to_parameters_cache::MOI.Utilities.DoubleDicts.DoubleDict{
        Vector{MOI.ScalarAffineTerm{T}},
    }
    # Store the map for SQFs that were transformed into SAF
    # for instance p*p + var -> ScalarAffine(var)
    quadratic_added_cache::Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}
    last_quad_add_added::Int64
    vector_constraint_cache::MOI.Utilities.DoubleDicts.DoubleDict{
        Vector{MOI.VectorAffineTerm{Float64}},
    }
    # Ditto from the constraint caches but for the objective function
    original_objective_function::MOI.AbstractFunction
    affine_objective_cache::Vector{MOI.ScalarAffineTerm{T}}
    quadratic_objective_cache_pv::Vector{MOI.ScalarQuadraticTerm{T}}
    quadratic_objective_cache_pp::Vector{MOI.ScalarQuadraticTerm{T}}
    quadratic_objective_cache_pc::Vector{MOI.ScalarAffineTerm{T}}
    quadratic_objective_variables_associated_to_parameters_cache::Vector{
        MOI.ScalarAffineTerm{T},
    }
    multiplicative_parameters::Set{Int64}
    dual_value_of_parameters::Vector{Float64}
    evaluate_duals::Bool
    number_of_parameters_in_model::Int64
    function Optimizer(optimizer::OT; evaluate_duals::Bool = true) where {OT}
        return new{Float64,OT}(
            optimizer,
            MOI.Utilities.CleverDicts.CleverDict{ParameterIndex,Float64}(
                MOI.Utilities.CleverDicts.key_to_index,
                MOI.Utilities.CleverDicts.index_to_key,
            ),
            Dict{MOI.VariableIndex,String}(),
            MOI.Utilities.CleverDicts.CleverDict{ParameterIndex,Float64}(
                MOI.Utilities.CleverDicts.key_to_index,
                MOI.Utilities.CleverDicts.index_to_key,
            ),
            MOI.Utilities.CleverDicts.CleverDict{
                MOI.VariableIndex,
                MOI.VariableIndex,
            }(
                MOI.Utilities.CleverDicts.key_to_index,
                MOI.Utilities.CleverDicts.index_to_key,
            ),
            0,
            PARAMETER_INDEX_THRESHOLD,
            Dict{
                MOI.ConstraintIndex,
                Tuple{MOI.AbstractFunction,MOI.AbstractSet},
            }(),
            MOI.Utilities.DoubleDicts.DoubleDict{
                Vector{MOI.ScalarAffineTerm{Float64}},
            }(),
            MOI.Utilities.DoubleDicts.DoubleDict{
                Vector{MOI.ScalarQuadraticTerm{Float64}},
            }(),
            MOI.Utilities.DoubleDicts.DoubleDict{
                Vector{MOI.ScalarQuadraticTerm{Float64}},
            }(),
            MOI.Utilities.DoubleDicts.DoubleDict{
                Vector{MOI.ScalarAffineTerm{Float64}},
            }(),
            MOI.Utilities.DoubleDicts.DoubleDict{
                Vector{MOI.ScalarAffineTerm{Float64}},
            }(),
            Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}(),
            0,
            MOI.Utilities.DoubleDicts.DoubleDict{
                Vector{MOI.VectorAffineTerm{Float64}},
            }(),
            MOI.VariableIndex(-1),
            Vector{MOI.ScalarAffineTerm{Float64}}(),
            Vector{MOI.ScalarQuadraticTerm{Float64}}(),
            Vector{MOI.ScalarQuadraticTerm{Float64}}(),
            Vector{MOI.ScalarAffineTerm{Float64}}(),
            Vector{MOI.ScalarAffineTerm{Float64}}(),
            Set{Int64}(),
            Vector{Float64}(),
            evaluate_duals,
            0,
        )
    end
end

include("utils.jl")
include("duals.jl")
include("update_parameters.jl")

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.optimizer) &&
           isempty(model.parameters) &&
           isempty(model.parameters_name) &&
           isempty(model.variables) &&
           isempty(model.updated_parameters) &&
           isempty(model.variables) &&
           model.last_variable_index_added == 0 &&
           model.last_parameter_index_added == PARAMETER_INDEX_THRESHOLD &&
           isempty(model.original_constraint_function_and_set_cache) &&
           isempty(model.affine_constraint_cache) &&
           isempty(model.quadratic_constraint_cache_pv) &&
           isempty(model.quadratic_constraint_cache_pp) &&
           isempty(model.quadratic_constraint_cache_pc) &&
           isempty(
               model.quadratic_constraint_variables_associated_to_parameters_cache,
           ) &&
           isempty(model.quadratic_added_cache) &&
           model.last_quad_add_added == 0 &&
           model.original_objective_function == MOI.VariableIndex(-1) &&
           isempty(model.affine_objective_cache) &&
           isempty(model.quadratic_objective_cache_pv) &&
           isempty(model.quadratic_objective_cache_pp) &&
           isempty(model.quadratic_objective_cache_pc) &&
           isempty(
               model.quadratic_objective_variables_associated_to_parameters_cache,
           ) &&
           isempty(model.dual_value_of_parameters) &&
           model.number_of_parameters_in_model == 0
end

function MOI.supports_constraint(
    model::Optimizer,
    F::Union{
        Type{MOI.VariableIndex},
        Type{MOI.ScalarAffineFunction{T}},
        Type{MOI.VectorOfVariables},
        Type{MOI.VectorAffineFunction{T}},
    },
    S::Type{<:MOI.AbstractSet},
) where {T}
    return MOI.supports_constraint(model.optimizer, F, S)
end

function MOI.supports_constraint(
    model::Optimizer,
    ::Type{MOI.ScalarQuadraticFunction{T}},
    S::Type{<:MOI.AbstractSet},
) where {T}
    return MOI.supports_constraint(
        model.optimizer,
        MOI.ScalarAffineFunction{T},
        S,
    )
end

function MOI.supports_constraint(
    model::Optimizer,
    ::Type{MOI.VectorQuadraticFunction{T}},
    S::Type{<:MOI.AbstractSet},
) where {T}
    return MOI.supports_constraint(
        model.optimizer,
        MOI.VectorAffineFunction{T},
        S,
    )
end

function MOI.supports(
    model::Optimizer,
    attr::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{MOI.VariableIndex},
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}},
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}},
    },
) where {T}
    return MOI.supports(model.optimizer, attr)
end

function MOI.supports_incremental_interface(model::Optimizer)
    return MOI.supports_incremental_interface(model.optimizer)
end
function MOI.supports(model::Optimizer, ::MOI.Name)
    return MOI.supports(model.optimizer, MOI.Name())
end
function MOI.get(model::Optimizer, ::MOI.ListOfModelAttributesSet)
    return MOI.get(model.optimizer, MOI.ListOfModelAttributesSet())
end
MOI.get(model::Optimizer, ::MOI.Name) = MOI.get(model.optimizer, MOI.Name())
function MOI.set(model::Optimizer, ::MOI.Name, name::String)
    return MOI.set(model.optimizer, MOI.Name(), name)
end
function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return MOI.get(model.optimizer, MOI.ListOfVariableIndices())
end
function MOI.get(model::Optimizer, ::MOI.ListOfVariableAttributesSet)
    return MOI.get(model.optimizer, MOI.ListOfVariableAttributesSet())
end
function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintAttributesSet{F,S},
) where {F,S}
    if F === MOI.ScalarQuadraticFunction
        error(
            "MOI.ListOfConstraintAttributesSet is not implemented for ScalarQuadraticFunction.",
        )
    end
    return MOI.get(model.optimizer, MOI.ListOfConstraintAttributesSet{F,S}())
end
function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{F,S},
    f::F,
) where {F,S}
    MOI.set(model.optimizer, MOI.ConstraintFunction(), c, f)
    return
end
function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,S},
    s::S,
) where {F,S}
    MOI.set(model.optimizer, MOI.ConstraintSet(), c, s)
    return
end
function MOI.modify(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
    chg::MOI.ScalarCoefficientChange{Float64},
) where {F,S}
    MOI.modify(model.optimizer, c, chg)
    return
end
function MOI.modify(
    model::Optimizer,
    c::MOI.ObjectiveFunction{F},
    chg::MOI.ScalarCoefficientChange{Float64},
) where {F<:MathOptInterface.AbstractScalarFunction}
    MOI.modify(model.optimizer, c, chg)
    return
end
function MOI.modify(
    model::Optimizer,
    c::MOI.ObjectiveFunction{F},
    chg::MOI.ScalarConstantChange{Float64},
) where {F<:MathOptInterface.AbstractScalarFunction}
    MOI.modify(model.optimizer, c, chg)
    return
end

function MOI.empty!(model::Optimizer{T}) where {T}
    MOI.empty!(model.optimizer)
    empty!(model.parameters)
    empty!(model.parameters_name)
    empty!(model.updated_parameters)
    empty!(model.variables)
    model.last_variable_index_added = 0
    model.last_parameter_index_added = PARAMETER_INDEX_THRESHOLD
    empty!(model.original_constraint_function_and_set_cache)
    empty!(model.affine_constraint_cache)
    empty!(model.quadratic_constraint_cache_pv)
    empty!(model.quadratic_constraint_cache_pp)
    empty!(model.quadratic_constraint_cache_pc)
    empty!(model.quadratic_constraint_variables_associated_to_parameters_cache)
    empty!(model.quadratic_added_cache)
    model.last_quad_add_added = 0
    model.original_objective_function = MOI.VariableIndex(-1)
    empty!(model.vector_constraint_cache)
    empty!(model.affine_objective_cache)
    empty!(model.quadratic_objective_cache_pv)
    empty!(model.quadratic_objective_cache_pp)
    empty!(model.quadratic_objective_cache_pc)
    empty!(model.quadratic_objective_variables_associated_to_parameters_cache)
    empty!(model.dual_value_of_parameters)
    model.number_of_parameters_in_model = 0
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.VariableName,
    v::MOI.VariableIndex,
    name::String,
)
    if is_parameter_in_model(model, v)
        model.parameters_name[v] = name
    else
        MOI.set(model.optimizer, attr, v, name)
    end
    return
end

function MOI.get(model::Optimizer, attr::MOI.VariableName, v::MOI.VariableIndex)
    if is_parameter_in_model(model, v)
        return get(model.parameters_name, v, "")
    else
        return MOI.get(model.optimizer, attr, v)
    end
end

function MOI.get(model::Optimizer, tp::Type{MOI.VariableIndex}, attr::String)
    return MOI.get(model.optimizer, tp, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveBound)
    return MOI.get(model.optimizer, attr)
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.VariableName,
    tp::Type{MOI.VariableIndex},
)
    return MOI.supports(model.optimizer, attr, tp)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex,
    name::String,
)
    MOI.set(model.optimizer, attr, c, name)
    return
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex,
)
    return MOI.get(model.optimizer, attr, c)
end

function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    return length(model.parameters) + length(model.variables)
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.ConstraintName,
    tp::Type{<:MOI.ConstraintIndex},
)
    return MOI.supports(model.optimizer, attr, tp)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintFunction,
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    if haskey(model.original_constraint_function_and_set_cache, ci)
        return model.original_constraint_function_and_set_cache[ci][1]
    else
        return MOI.get(model.optimizer, attr, ci)
    end
end

function MOI.get(
    model::Optimizer,
    tp::Type{MOI.ConstraintIndex{F,S}},
    attr::String,
) where {F,S}
    return MOI.get(model.optimizer, tp, attr)
end

function MOI.get(model::Optimizer, tp::Type{MOI.ConstraintIndex}, attr::String)
    return MOI.get(model.optimizer, tp, attr)
end

function MOI.is_valid(model::Optimizer, vi::MOI.VariableIndex)
    return MOI.is_valid(model.optimizer, vi)
end

function MOI.supports(model::Optimizer, ::MOI.NumberOfThreads)
    return MOI.supports(model.optimizer, MOI.NumberOfThreads())
end

function MOI.supports(model::Optimizer, ::MOI.TimeLimitSec)
    return MOI.supports(model.optimizer, MOI.TimeLimitSec())
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value)
    MOI.set(model.optimizer, MOI.TimeLimitSec(), value)
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return MOI.get(model.optimizer, MOI.TimeLimitSec())
end

function MOI.get(model::Optimizer, ::MOI.SolveTimeSec)
    return MOI.get(model.optimizer, MOI.SolveTimeSec())
end

function MOI.supports(model::Optimizer, ::MOI.Silent)
    return MOI.supports(model.optimizer, MOI.Silent())
end

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    MOI.set(model.optimizer, MOI.Silent(), value)
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = MOI.get(model.optimizer, MOI.Silent())

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    return MOI.get(model.optimizer, MOI.RawStatusString())
end

function MOI.get(model::Optimizer, ::MOI.NumberOfConstraints{F,S}) where {F,S}
    return length(MOI.get(model, MOI.ListOfConstraintIndices{F,S}()))
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    if haskey(model.original_constraint_function_and_set_cache, ci)
        return model.original_constraint_function_and_set_cache[ci][2]
    else
        MOI.throw_if_not_valid(model, ci)
        return MOI.get(model.optimizer, attr, ci)
    end
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveSense)
    return MOI.get(model.optimizer, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveFunctionType)
    return typeof(model.original_objective_function)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{F},
) where {
    F<:Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{T},
        MOI.ScalarQuadraticFunction{T},
    },
} where {T}
    if !function_has_parameters(model, model.original_objective_function)
        return MOI.get(model.optimizer, attr)
    else
        if F === typeof(model.original_objective_function)
            return model.original_objective_function
        else
            throw(InexactError)
        end
    end
end

function MOI.get(model::Optimizer, attr::MOI.ResultCount)
    return MOI.get(model.optimizer, attr)
end

# In the AbstractBridgeOptimizer, we collect all the possible constraint types and them filter with NumberOfConstraints.
# If NumberOfConstraints is zero then we remove it from the list.
# Here, you can look over keys(quadratic_added_cache) and add the F-S types of all the keys in constraints.
# To implement NumberOfConstraints, you call NumberOfConstraints to the inner optimizer.
# Then you remove the number of constraints of that that in values(quadratic_added_cache)
function MOI.get(model::Optimizer, ::MOI.ListOfConstraintTypesPresent)
    inner_ctrs = MOI.get(model.optimizer, MOI.ListOfConstraintTypesPresent())
    if !has_quadratic_constraint_caches(model)
        return inner_ctrs
    end

    cache_keys = collect(keys(model.quadratic_added_cache))
    constraints = Set{Tuple{DataType,DataType}}()

    for (F, S) in inner_ctrs
        inner_index =
            MOI.get(model.optimizer, MOI.ListOfConstraintIndices{F,S}())
        cache_map_check =
            quadratic_constraint_cache_map_check.(model, inner_index)
        for type in typeof.(cache_keys[cache_map_check])
            push!(constraints, (type.parameters[1], type.parameters[2]))
        end
        # If not all the constraints are chached then also push the original type
        if !all(cache_map_check)
            push!(constraints, (F, S))
        end
    end

    return collect(constraints)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ListOfConstraintIndices{F,S},
) where {S,F<:Union{MOI.VectorOfVariables,MOI.VariableIndex}}
    return MOI.get(model.optimizer, attr)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{F,S},
) where {
    S<:MOI.AbstractSet,
    F<:Union{MOI.ScalarAffineFunction{T},MOI.VectorAffineFunction{T}},
} where {T}
    inner_index = MOI.get(model.optimizer, MOI.ListOfConstraintIndices{F,S}())
    if !has_quadratic_constraint_caches(model)
        return inner_index
    end

    cache_map_check = quadratic_constraint_cache_map_check(mode, inner_index)
    return inner_index[cache_map_check]
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{F,S},
) where {S<:MOI.AbstractSet,F<:MOI.ScalarQuadraticFunction{T}} where {T}
    inner_index = MOI.ConstraintIndex{F,S}[]
    if MOI.supports_constraint(model.optimizer, F, S)
        inner_index =
            MOI.get(model.optimizer, MOI.ListOfConstraintIndices{F,S}())
        if !has_quadratic_constraint_caches(model)
            return inner_index
        end
    end

    quadratic_caches = [
        :quadratic_constraint_cache_pc,
        :quadratic_constraint_cache_pp,
        :quadratic_constraint_cache_pv,
        # JD: Check if this applies here
        # :quadratic_constraint_variables_associated_to_parameters_cache
    ]

    for field in quadratic_caches
        cache = getfield(model, field)
        push!(inner_index, keys(cache)...)
    end
    return inner_index
end

function MOI.supports_add_constrained_variable(::Optimizer, ::Type{Parameter})
    return true
end

function MOI.supports_add_constrained_variables(
    model::Optimizer,
    ::Type{MOI.Reals},
)
    return MOI.supports_add_constrained_variables(model.optimizer, MOI.Reals)
end

function MOI.add_variable(model::Optimizer)
    next_variable_index!(model)
    return MOI.Utilities.CleverDicts.add_item(
        model.variables,
        MOI.add_variable(model.optimizer),
    )
end

function MOI.add_constrained_variable(model::Optimizer, set::Parameter)
    next_parameter_index!(model)
    p = MOI.VariableIndex(model.last_parameter_index_added)
    MOI.Utilities.CleverDicts.add_item(model.parameters, set.val)
    cp = MOI.ConstraintIndex{MOI.VariableIndex,Parameter}(
        model.last_parameter_index_added,
    )
    MOI.Utilities.CleverDicts.add_item(model.updated_parameters, NaN)
    update_number_of_parameters!(model)
    return p, cp
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.VariableIndex,
    set::MOI.AbstractScalarSet,
)
    if is_parameter_in_model(model, f)
        error("Cannot constrain a parameter")
    elseif !is_variable_in_model(model, f)
        error("Variable not in the model")
    end
    return MOI.add_constraint(model.optimizer, f, set)
end

function add_constraint_with_parameters_on_function(
    model::Optimizer,
    f::MOI.ScalarAffineFunction{T},
    set::MOI.AbstractScalarSet,
) where {T}
    vars, params, param_constant =
        separate_possible_terms_and_calculate_parameter_constant(model, f.terms)
    ci = MOI.Utilities.normalize_and_add_constraint(
        model.optimizer,
        MOI.ScalarAffineFunction(vars, f.constant + param_constant),
        set,
    )
    model.affine_constraint_cache[ci] = params
    model.original_constraint_function_and_set_cache[ci] = (f, set)
    return ci
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.ScalarAffineFunction{T},
    set::MOI.AbstractScalarSet,
) where {T}
    if !function_has_parameters(model, f)
        return MOI.add_constraint(model.optimizer, f, set)
    else
        return add_constraint_with_parameters_on_function(model, f, set)
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    v::MOI.VariableIndex,
)
    if is_parameter_in_model(model, v)
        return model.parameters[p_idx(v)]
    elseif is_variable_in_model(model, v)
        return MOI.get(model.optimizer, attr, model.variables[v])
    else
        error("Variable not in the model")
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariableBasisStatus,
    v::MOI.VariableIndex,
)
    if is_variable_in_model(model, v)
        return MOI.get(model.optimizer, attr, model.variables[v])
    else
        error("VariableBasisStatus is not supported for parameters.")
    end
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    cp::MOI.ConstraintIndex{MOI.VariableIndex,Parameter},
    set::Parameter,
)
    p = MOI.VariableIndex(cp.value)
    if !is_parameter_in_model(model, p)
        error("Parameter not in the model")
    end
    return model.updated_parameters[p_idx(p)] = set.val
end

"""
    ParameterValue <: MOI.AbstractVariableAttribute

Attribute defined to set and get parameter values

# Example

```julia
MOI.set(model, POI.ParameterValue(), p, 2.0)
MOI.get(model, POI.ParameterValue(), p)
```
"""
struct ParameterValue <: MOI.AbstractVariableAttribute end

function MOI.set(
    model::Optimizer,
    ::ParameterValue,
    vi::MOI.VariableIndex,
    val::Float64,
)
    if !is_parameter_in_model(model, vi)
        error("Parameter not in the model")
    end
    return model.updated_parameters[p_idx(vi)] = val
end

function MOI.get(model::Optimizer, ::ParameterValue, vi::MOI.VariableIndex)
    if !is_parameter_in_model(model, vi)
        error("Parameter not in the model")
    end
    return model.updated_parameters[p_idx(vi)]
end

function MOI.set(
    model::Optimizer,
    ::ParameterValue,
    vi::MOI.VariableIndex,
    val::Real,
)
    return MOI.set(model, ParameterValue(), vi, convert(Float64, val))
end

function empty_objective_function_caches!(model::Optimizer)
    empty!(model.affine_objective_cache)
    empty!(model.quadratic_objective_cache_pv)
    empty!(model.quadratic_objective_cache_pp)
    empty!(model.quadratic_objective_cache_pc)
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction,
    f::MOI.ScalarAffineFunction{T},
) where {T}
    # clear previously defined objetive function cache
    empty_objective_function_caches!(model)
    if !function_has_parameters(model, f)
        MOI.set(model.optimizer, attr, f)
    else
        vars, params, param_constant =
            separate_possible_terms_and_calculate_parameter_constant(
                model,
                f.terms,
            )
        MOI.set(
            model.optimizer,
            attr,
            MOI.ScalarAffineFunction(vars, f.constant + param_constant),
        )
        model.affine_objective_cache = params
    end
    model.original_objective_function = f
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction,
    v::MOI.VariableIndex,
)
    if is_parameter_in_model(model, v)
        error("Cannot use a parameter as objective function alone")
    elseif !is_variable_in_model(model, v)
        error("Variable not in the model")
    end
    model.original_objective_function = v
    return MOI.set(model.optimizer, attr, model.variables[v])
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    MOI.set(model.optimizer, attr, sense)
    return
end

function MOI.get(
    model::Optimizer,
    attr::T,
) where {
    T<:Union{
        MOI.TerminationStatus,
        MOI.ObjectiveValue,
        MOI.DualObjectiveValue,
        MOI.PrimalStatus,
        MOI.DualStatus,
    },
}
    return MOI.get(model.optimizer, attr)
end

function MOI.get(
    model::Optimizer,
    attr::T,
    c::MOI.ConstraintIndex,
) where {
    T<:Union{MOI.ConstraintPrimal,MOI.ConstraintDual,MOI.ConstraintBasisStatus},
}
    return MOI.get(model.optimizer, attr, c)
end

function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, val::Any)
    MOI.set(model.optimizer, attr, val)
    return
end

function MOI.get(model::Optimizer, ::MOI.SolverName)
    name = MOI.get(model.optimizer, MOI.SolverName())
    return "Parametric Optimizer with $(name) attached"
end

function MOI.get(model::Optimizer, ::MOI.SolverVersion)
    return MOI.get(model.optimizer, MOI.SolverVersion())
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.VectorOfVariables,
    set::MOI.AbstractVectorSet,
) where {T}
    if function_has_parameters(model, f)
        error("VectorOfVariables does not allow parameters")
    end
    return MOI.add_constraint(model.optimizer, f, set)
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.VectorAffineFunction{T},
    set::MOI.AbstractVectorSet,
) where {T}
    if !function_has_parameters(model, f)
        return MOI.add_constraint(model.optimizer, f, set)
    else
        return add_constraint_with_parameters_on_function(model, f, set)
    end
end

function add_constraint_with_parameters_on_function(
    model::Optimizer,
    f::MOI.VectorAffineFunction{T},
    set::MOI.AbstractVectorSet,
) where {T}
    vars, params, param_constants =
        separate_possible_terms_and_calculate_parameter_constant(model, f, set)
    ci = MOI.add_constraint(
        model.optimizer,
        MOI.VectorAffineFunction(vars, f.constants + param_constants),
        set,
    )
    model.vector_constraint_cache[ci] = params
    model.original_constraint_function_and_set_cache[ci] = (f, set)
    return ci
end

function add_constraint_with_parameters_on_function(
    model::Optimizer,
    f::MOI.ScalarQuadraticFunction{T},
    set::S,
) where {T,S<:MOI.AbstractScalarSet}
    (
        quad_vars,
        quad_aff_vars,
        quad_params,
        aff_terms,
        variables_associated_to_parameters,
        quad_param_constant,
    ) = separate_possible_terms_and_calculate_parameter_constant(
        model,
        f.quadratic_terms,
    )

    (
        aff_vars,
        aff_params,
        terms_with_variables_associated_to_parameters,
        aff_param_constant,
    ) = separate_possible_terms_and_calculate_parameter_constant(
        model,
        f.affine_terms,
        variables_associated_to_parameters,
    )

    aff_terms = vcat(aff_terms, aff_vars)
    const_term = f.constant + aff_param_constant + quad_param_constant
    quad_terms = quad_vars
    f_quad = if !isempty(quad_vars)
        MOI.ScalarQuadraticFunction(quad_terms, aff_terms, const_term)
    else
        MOI.ScalarAffineFunction(aff_terms, const_term)
    end
    model.last_quad_add_added += 1
    ci =
        MOI.Utilities.normalize_and_add_constraint(model.optimizer, f_quad, set)
    # This part is used to remember that ci came from a quadratic function
    # It is particularly useful because sometimes the constraint mutates
    new_ci = MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S}(
        model.last_quad_add_added,
    )
    model.quadratic_added_cache[new_ci] = ci
    model.original_constraint_function_and_set_cache[new_ci] = (f, set)
    fill_quadratic_constraint_caches!(
        model,
        new_ci,
        quad_aff_vars,
        quad_params,
        aff_params,
        terms_with_variables_associated_to_parameters,
    )
    return new_ci
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.ScalarQuadraticFunction{T},
    set::MOI.AbstractScalarSet,
) where {T}
    if !function_has_parameters(model, f)
        return MOI.add_constraint(model.optimizer, f, set)
    else
        return add_constraint_with_parameters_on_function(model, f, set)
    end
end

function MOI.delete(model::Optimizer, v::MOI.VariableIndex)
    delete!(model.variables, v)
    MOI.delete(model.optimizer, v)
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {F<:MOI.ScalarAffineFunction,S<:MOI.AbstractSet}
    if haskey(model.affine_constraint_cache, c)
        delete!(model.affine_constraint_cache, c)
    end
    MOI.delete(model.optimizer, c)
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{MOI.VariableIndex,MOI.VectorOfVariables,MOI.VectorAffineFunction},
    S<:MOI.AbstractSet,
}
    MOI.delete(model.optimizer, c)
    return
end

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{MOI.VariableIndex,MOI.VectorOfVariables,MOI.VectorAffineFunction},
    S<:MOI.AbstractSet,
}
    return MOI.is_valid(model.optimizer, c)
end

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {F<:MOI.ScalarAffineFunction,S<:MOI.AbstractSet}
    return MOI.is_valid(model.optimizer, c)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction,
    f::MOI.ScalarQuadraticFunction{T},
) where {T}
    # clear previously defined objetive function cache
    empty_objective_function_caches!(model)
    if !function_has_parameters(model, f)
        MOI.set(model.optimizer, attr, f)
        return
    end
    (
        quad_vars,
        quad_aff_vars,
        quad_params,
        aff_terms,
        variables_associated_to_parameters,
        quad_param_constant,
    ) = separate_possible_terms_and_calculate_parameter_constant(
        model,
        f.quadratic_terms,
    )

    (
        aff_vars,
        aff_params,
        terms_with_variables_associated_to_parameters,
        aff_param_constant,
    ) = separate_possible_terms_and_calculate_parameter_constant(
        model,
        f.affine_terms,
        variables_associated_to_parameters,
    )

    aff_terms = vcat(aff_terms, aff_vars)
    const_term = f.constant + aff_param_constant + quad_param_constant
    quad_terms = quad_vars

    if !isempty(quad_vars)
        MOI.set(
            model.optimizer,
            attr,
            MOI.ScalarQuadraticFunction(quad_terms, aff_terms, const_term),
        )
    else
        MOI.set(
            model.optimizer,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
            MOI.ScalarAffineFunction(aff_terms, const_term),
        )
    end

    if !isempty(quad_terms)
        f_quad = MOI.ScalarQuadraticFunction(quad_terms, aff_terms, const_term)
        MOI.set(model.optimizer, attr, f_quad)
    else
        f_quad = MOI.ScalarAffineFunction(aff_terms, const_term)
        MOI.set(
            model.optimizer,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
            f_quad,
        )
    end

    model.original_objective_function = f
    model.quadratic_objective_cache_pv = quad_aff_vars
    model.quadratic_objective_cache_pp = quad_params
    model.quadratic_objective_cache_pc = aff_params
    model.quadratic_objective_variables_associated_to_parameters_cache =
        terms_with_variables_associated_to_parameters

    return
end

function MOI.optimize!(model::Optimizer)
    if !isempty(model.updated_parameters)
        update_parameters!(model)
    end
    MOI.optimize!(model.optimizer)
    if model.evaluate_duals
        calculate_dual_of_parameters(model)
    end
    return
end

end # module
