# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module ParametricOptInterface

using MathOptInterface

const MOI = MathOptInterface

const PARAMETER_INDEX_THRESHOLD = Int64(4_611_686_018_427_387_904) # div(typemax(Int64),2)+1

@enum ConstraintsInterpretationCode ONLY_CONSTRAINTS ONLY_BOUNDS BOUNDS_AND_CONSTRAINTS

const SIMPLE_SCALAR_SETS{T} = Union{MOI.LessThan{T},MOI.GreaterThan{T},MOI.EqualTo{T}}

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
function p_idx(vi::MOI.VariableIndex)::ParameterIndex
    return ParameterIndex(vi.value - PARAMETER_INDEX_THRESHOLD)
end
function p_val(vi::MOI.VariableIndex)::Int64
    return vi.value - PARAMETER_INDEX_THRESHOLD
end
function p_val(ci::MOI.ConstraintIndex)::Int64
    return ci.value - PARAMETER_INDEX_THRESHOLD
end

const ParamTo{T} = MOI.Utilities.CleverDicts.CleverDict{
    ParameterIndex,
    T,
    typeof(MOI.Utilities.CleverDicts.key_to_index),
    typeof(MOI.Utilities.CleverDicts.index_to_key),
}
const VariableMap = MOI.Utilities.CleverDicts.CleverDict{
    MOI.VariableIndex,
    MOI.VariableIndex,
    typeof(MOI.Utilities.CleverDicts.key_to_index),
    typeof(MOI.Utilities.CleverDicts.index_to_key),
}

const DoubleDict{T} = MOI.Utilities.DoubleDicts.DoubleDict{T}
const DoubleDictInner{F,S,T} = MOI.Utilities.DoubleDicts.DoubleDictInner{F,S,T}

mutable struct ParametricQuadraticFunction{T}
    # helper to efficiently update affine terms 
    affine_data::Dict{MOI.VariableIndex,T}
    affine_data_np::Dict{MOI.VariableIndex,T}
    # constant * parameter * variable (in this order)
    pv::Vector{MOI.ScalarQuadraticTerm{T}}
    # constant * parameter * parameter
    pp::Vector{MOI.ScalarQuadraticTerm{T}}
    # constant * variable * variable
    vv::Vector{MOI.ScalarQuadraticTerm{T}}
    # constant * parameter
    p::Vector{MOI.ScalarAffineTerm{T}}
    # constant * variable
    v::Vector{MOI.ScalarAffineTerm{T}}
    # constant (does not include the set constant)
    c::T
    # to avoid unnecessary lookups in updates
    set_constant::T
    # cache data that is inside the solver to avoid slow getters
    current_terms_with_p::Dict{MOI.VariableIndex,T}
    current_constant::T
    # computed on runtime
    # updated_terms_with_p::Dict{MOI.VariableIndex,T}
    # updated_constant::T
end

mutable struct ParametricAffineFunction{T}
    # constant * parameter
    p::Vector{MOI.ScalarAffineTerm{T}}
    # constant * variable
    v::Vector{MOI.ScalarAffineTerm{T}}
    # constant
    c::T
    # to avoid unnecessary lookups in updates
    set_constant::T
    # cache to avoid slow getters
    current_constant::T
end

mutable struct ParametricVectorAffineFunction{T}
    # constant * parameter
    p::Vector{MOI.VectorAffineTerm{T}}
    # constant * variable
    v::Vector{MOI.VectorAffineTerm{T}}
    # constant
    c::Vector{T}
    # to avoid unnecessary lookups in updates
    set_constant::Vector{T}
    # cache to avoid slow getters
    current_constant::Vector{T}
end

"""
    Optimizer{T, OT <: MOI.ModelLike} <: MOI.AbstractOptimizer

Declares a `Optimizer`, which allows the handling of parameters in a
optimization model.

## Keyword arguments

- `evaluate_duals::Bool`: If `true`, evaluates the dual of parameters. Users might want to set it to `false`
  to increase performance when the duals of parameters are not necessary. Defaults to `true`.

- `save_original_objective_and_constraints`: If `true` saves the orginal function and set of the constraints
  as well as the original objective function inside [`POI.Optimizer`](@ref). This is useful for printing the model
  but greatly increases the memory footprint. Users might want to set it to `false` to increase performance
  in applications where you don't need to query the original expressions provided to the model in constraints
  or in the objective. Note that this might break printing or queries such as `MOI.get(model, MOI.ConstraintFunction(), c)`.
  Defaults to `true`.

## Example

```julia-repl
julia> ParametricOptInterface.Optimizer(GLPK.Optimizer())
ParametricOptInterface.Optimizer{Float64,GLPK.Optimizer}
```
"""
mutable struct Optimizer{T,OT<:MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT
    parameters::ParamTo{T}
    parameters_name::Dict{MOI.VariableIndex,String}
    # The updated_parameters dictionary has the same dimension of the
    # parameters dictionary and if the value stored is a NaN is means
    # that the parameter has not been updated.
    updated_parameters::ParamTo{T}
    variables::VariableMap
    last_variable_index_added::Int64
    last_parameter_index_added::Int64

    # affine constraint data
    last_affine_added::Int64
    # Store the map for SAFs (some might be transformed into VI)
    affine_outer_to_inner::DoubleDict{MOI.ConstraintIndex}
    # Clever cache of data (inner key)
    affine_constraint_cache::DoubleDict{ParametricAffineFunction{T}}
    # Store original constraint set (inner key)
    affine_constraint_cache_set::DoubleDict{MOI.AbstractScalarSet}

    # quadratic constraitn data
    last_quad_add_added::Int64
    # Store the map for SQFs (some might be transformed into SAF)
    # for instance p*p + var -> ScalarAffine(var)
    quadratic_outer_to_inner::DoubleDict{MOI.ConstraintIndex}
    # Clever cache of data (inner key)
    quadratic_constraint_cache::DoubleDict{ParametricQuadraticFunction{T}}
    # Store original constraint set (inner key)
    quadratic_constraint_cache_set::DoubleDict{MOI.AbstractScalarSet}

    # objective function data
    # Clever cache of data (at most one can be !== nothing)
    affine_objective_cache::Union{Nothing,ParametricAffineFunction{T}}
    quadratic_objective_cache::Union{Nothing,ParametricQuadraticFunction{T}}
    original_objective_cache::MOI.Utilities.ObjectiveContainer{T}
    # Store parametric expressions for product of variables
    quadratic_objective_cache_product::Dict{
        Tuple{MOI.VariableIndex,MOI.VariableIndex},
        MOI.AbstractFunction,
    }

    # vector affine function data
    # vector_constraint_cache::DoubleDict{Vector{MOI.VectorAffineTerm{T}}}
    # Clever cache of data (inner key)
    vector_affine_constraint_cache::DoubleDict{ParametricVectorAffineFunction{T}}

    #
    multiplicative_parameters::Set{Int64}
    dual_value_of_parameters::Vector{T}

    # params
    evaluate_duals::Bool
    number_of_parameters_in_model::Int64
    constraints_interpretation::ConstraintsInterpretationCode
    save_original_objective_and_constraints::Bool
    function Optimizer(
        optimizer::OT;
        evaluate_duals::Bool = true,
        save_original_objective_and_constraints::Bool = true,
    ) where {OT}
        T = Float64
        return new{T,OT}(
            optimizer,
            MOI.Utilities.CleverDicts.CleverDict{ParameterIndex,T}(
                MOI.Utilities.CleverDicts.key_to_index,
                MOI.Utilities.CleverDicts.index_to_key,
            ),
            Dict{MOI.VariableIndex,String}(),
            MOI.Utilities.CleverDicts.CleverDict{ParameterIndex,T}(
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
            # affine constraint
            0,
            DoubleDict{MOI.ConstraintIndex}(),
            DoubleDict{ParametricAffineFunction{T}}(),
            DoubleDict{MOI.AbstractScalarSet}(),
            # quadratic constraint
            0,
            DoubleDict{MOI.ConstraintIndex}(),
            DoubleDict{ParametricQuadraticFunction{T}}(),
            DoubleDict{MOI.AbstractScalarSet}(),
            # objective
            nothing,
            nothing,
            # nothing,
            MOI.Utilities.ObjectiveContainer{T}(),
            Dict{
                Tuple{MOI.VariableIndex,MOI.VariableIndex},
                MOI.AbstractFunction,
            }(),
            # vec affine
            # DoubleDict{Vector{MOI.VectorAffineTerm{T}}}(),
            DoubleDict{ParametricVectorAffineFunction{T}}(),
            # other
            Set{Int64}(),
            Vector{T}(),
            evaluate_duals,
            0,
            ONLY_CONSTRAINTS,
            save_original_objective_and_constraints,
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
        isempty(model.updated_parameters) &&
        isempty(model.variables) &&
        model.last_variable_index_added == 0 &&
        model.last_parameter_index_added == PARAMETER_INDEX_THRESHOLD &&
        # affine ctr
        model.last_affine_added == 0 &&
        isempty(model.affine_outer_to_inner) &&
        isempty(model.affine_constraint_cache) &&
        isempty(model.affine_constraint_cache_set) &&
        # quad ctr
        model.last_quad_add_added == 0 &&
        isempty(model.quadratic_outer_to_inner) &&
        isempty(model.quadratic_constraint_cache) &&
        isempty(model.quadratic_constraint_cache_set) &&
        # obj
        model.affine_objective_cache === nothing &&
        model.quadratic_objective_cache === nothing &&
        MOI.is_empty(model.original_objective_cache) &&
        isempty(model.quadratic_objective_cache_product) &&
        #
        isempty(model.vector_affine_constraint_cache) &&
        #
        isempty(model.multiplicative_parameters) &&
        isempty(model.dual_value_of_parameters) &&
        model.number_of_parameters_in_model == 0
end

function MOI.empty!(model::Optimizer{T}) where {T}
    MOI.empty!(model.optimizer)
    empty!(model.parameters)
    empty!(model.parameters_name)
    empty!(model.updated_parameters)
    empty!(model.variables)
    model.last_variable_index_added = 0
    model.last_parameter_index_added = PARAMETER_INDEX_THRESHOLD
    # affine ctr
    model.last_affine_added = 0
    empty!(model.affine_outer_to_inner)
    empty!(model.affine_constraint_cache)
    empty!(model.affine_constraint_cache_set)
    # quad ctr
    model.last_quad_add_added = 0
    empty!(model.quadratic_outer_to_inner)
    empty!(model.quadratic_constraint_cache)
    empty!(model.quadratic_constraint_cache_set)
    # obj
    model.affine_objective_cache = nothing
    model.quadratic_objective_cache = nothing
    MOI.empty!(model.original_objective_cache)
    empty!(model.quadratic_objective_cache_product)
    #
    empty!(model.vector_affine_constraint_cache)
    #
    empty!(model.multiplicative_parameters)
    empty!(model.dual_value_of_parameters)
    #
    model.number_of_parameters_in_model = 0
    return
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
    },
) where {T}
    return MOI.supports(model.optimizer, attr)
end

function MOI.supports(
    model::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}},
) where {T}
    return MOI.supports(
        model.optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
    )
end

function MOI.supports(model::Optimizer, ::MOI.NLPBlock)
    return MOI.supports(model.optimizer, MOI.NLPBlock())
end

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    return MOI.set(model.optimizer, MOI.NLPBlock(), nlp_data)
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

struct ListOfPureVariableIndices <: MOI.AbstractModelAttribute end
struct ListOfParameterIndices <: MOI.AbstractModelAttribute end

function MOI.get(model::Optimizer, ::ListOfPureVariableIndices)
    return collect(keys(model.variables))
end
function MOI.get(model::Optimizer, ::ListOfParameterIndices)
    return collect(keys(model.parameters))
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
    chg::MOI.ScalarCoefficientChange{T},
) where {F,S,T}
    if haskey(model.quadratic_constraint_cache, c) ||
        haskey(model.affine_constraint_cache, c)
        error("Parametric constraint cannot be modified")
    end
    MOI.modify(model.optimizer, c, chg)
    return
end

function MOI.modify(
    model::Optimizer,
    c::MOI.ObjectiveFunction{F},
    chg::Union{
        MOI.ScalarConstantChange{T},
        MOI.ScalarCoefficientChange{T},
    },
) where {F<:MathOptInterface.AbstractScalarFunction,T}
    if model.quadratic_objective_cache !== nothing ||
        model.affine_objective_cache !== nothing ||
        !isempty(model.quadratic_objective_cache_product)
        error("Parametric objective cannot be modified")
    end
    MOI.modify(model.optimizer, c, chg)
    MOI.modify(model.original_objective_cache, c, chg)
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
    c::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S},
    name::String,
) where {T,S<:MOI.AbstractSet}
    if haskey(model.quadratic_outer_to_inner, c)
        MOI.set(model.optimizer, attr, model.quadratic_outer_to_inner[c], name)
    else
        MOI.set(model.optimizer, attr, c, name)
    end
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S},
    name::String,
) where {T,S<:MOI.AbstractSet}
    if haskey(model.affine_outer_to_inner, c)
        MOI.set(model.optimizer, attr, model.affine_outer_to_inner[c], name)
    else
        MOI.set(model.optimizer, attr, c, name)
    end
    return
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
    c::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S},
) where {T,S<:MOI.AbstractSet}
    if haskey(model.quadratic_outer_to_inner, c)
        return MOI.get(model.optimizer, attr, model.quadratic_outer_to_inner[c])
    else
        return MOI.get(model.optimizer, attr, c)
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex,
)
    return MOI.get(model.optimizer, attr, c)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S},
) where {T,S}

    if haskey(model.affine_outer_to_inner, c)
        inner_ci = model.affine_outer_to_inner[c]
        # This SAF constraint was transformed into variable bound
        if typeof(inner_ci) === MOI.ConstraintIndex{MOI.VariableIndex,S}
            v = MOI.get(model.optimizer, MOI.ConstraintFunction(), inner_ci)
            variable_name = MOI.get(model.optimizer, MOI.VariableName(), v)
            return "ParametricBound_$(S)_$(variable_name)"
        end
        return MOI.get(model.optimizer, attr, inner_ci)
    else
        return MOI.get(model.optimizer, attr, c)
    end
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
    if haskey(model.quadratic_outer_to_inner, ci)
        inner_ci = model.quadratic_outer_to_inner[ci]
        return original_function(model.quadratic_constraint_cache[inner_ci])
    elseif haskey(model.affine_outer_to_inner, ci)
        inner_ci = model.affine_outer_to_inner[ci]
        return original_function(model.affine_constraint_cache[inner_ci])
    else
        MOI.throw_if_not_valid(model, ci)
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
    if haskey(model.quadratic_outer_to_inner, ci)
        inner_ci = model.quadratic_outer_to_inner[ci]
        return model.quadratic_constraint_cache_set[inner_ci]
    elseif haskey(model.affine_outer_to_inner, ci)
        inner_ci = model.affine_outer_to_inner[ci]
        return model.affine_constraint_cache_set[inner_ci]
    else
        MOI.throw_if_not_valid(model, ci)
        return MOI.get(model.optimizer, attr, ci)
    end
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveSense)
    return MOI.get(model.optimizer, attr)
end

function MOI.get(model::Optimizer{T}, attr::MOI.ObjectiveFunctionType) where {T}
    return MOI.get(model.original_objective_cache, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveFunction)
    return MOI.get(model.original_objective_cache, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ResultCount)
    return MOI.get(model.optimizer, attr)
end

# TODO: cleanup
# In the AbstractBridgeOptimizer, we collect all the possible constraint types and them filter with NumberOfConstraints.
# If NumberOfConstraints is zero then we remove it from the list.
# Here, you can look over keys(quadratic_outer_to_inner) and add the F-S types of all the keys in constraints.
# To implement NumberOfConstraints, you call NumberOfConstraints to the inner optimizer.
# Then you remove the number of constraints of that that in values(quadratic_outer_to_inner)
function MOI.get(model::Optimizer, ::MOI.ListOfConstraintTypesPresent)
    inner_ctrs = MOI.get(model.optimizer, MOI.ListOfConstraintTypesPresent())
    if !has_quadratic_constraint_caches(model)
        return inner_ctrs
    end

    cache_keys = collect(keys(model.quadratic_outer_to_inner))
    constraints = Set{Tuple{DataType,DataType}}()

    for (F, S) in inner_ctrs
        inner_index =
            MOI.get(model.optimizer, MOI.ListOfConstraintIndices{F,S}())
        cache_map_check =
            quadratic_constraint_cache_map_check.(model, inner_index)
        for type in typeof.(cache_keys[cache_map_check])
            push!(constraints, (type.parameters[1], type.parameters[2]))
        end
        # If not all the constraints are cached then also push the original type
        # since there was a function with no parameters of that type
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
    end
    if !has_quadratic_constraint_caches(model)
        return inner_index
    end

    for key in keys(model.quadratic_outer_to_inner)
        push!(inner_index, key)
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
    if !is_variable(f)
        error("Cannot constrain a parameter")
    elseif !is_variable_in_model(model, f)
        error("Variable not in the model")
    end
    return MOI.add_constraint(model.optimizer, f, set)
end

function add_constraint_with_parameters_on_function(
    model::Optimizer,
    f::MOI.ScalarAffineFunction{T},
    set::S,
) where {T,S}
    pf = ParametricAffineFunction(f)
    cache_set_constant!(pf, set)
    if model.constraints_interpretation == ONLY_BOUNDS
        if length(pf.v) == 1 && isone(MOI.coefficient(pf.v[]))
            poi_ci = add_vi_constraint(model, pf, set)
        else
            error(
                "It was not possible to interpret this constraint as a variable bound.",
            )
        end
    elseif model.constraints_interpretation == ONLY_CONSTRAINTS
        poi_ci = add_saf_constraint(model, pf, set)
    elseif model.constraints_interpretation == BOUNDS_AND_CONSTRAINTS
        if length(pf.v) == 1 && isone(MOI.coefficient(pf.v[]))
            poi_ci = add_vi_constraint(model, pf, set)
        else
            poi_ci = add_saf_constraint(model, pf, set)
        end
    end
    return poi_ci
end

function add_saf_constraint(
    model::Optimizer,
    pf::ParametricAffineFunction{T},
    set::S,
) where {T,S}
    update_cache!(pf, model)
    inner_ci = MOI.Utilities.normalize_and_add_constraint(
        model.optimizer,
        MOI.ScalarAffineFunction{T}(pf.v, 0.0),
        set_with_new_constant(set, pf.current_constant),
    )
    model.last_affine_added += 1
    outer_ci = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S}(
        model.last_affine_added,
    )
    model.affine_outer_to_inner[outer_ci] = inner_ci
    # model.outer_to_inner_map[outer_ci] = inner_ci
    model.affine_constraint_cache[inner_ci] = pf
    model.affine_constraint_cache_set[inner_ci] = set
    return outer_ci
end

function add_vi_constraint(
    model::Optimizer,
    pf::ParametricAffineFunction{T},
    set::S,
) where {T,S}
    update_cache!(pf, model)
    inner_ci = MOI.Utilities.normalize_and_add_constraint(
        model.optimizer,
        pf.v[].variable,
        set_with_new_constant(set, pf.current_constant),
    )
    model.last_affine_added += 1
    outer_ci = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S}(
        model.last_affine_added,
    )
    model.affine_outer_to_inner[outer_ci] = inner_ci
    # model.outer_to_inner_map[outer_ci] = inner_ci
    model.affine_constraint_cache[inner_ci] = pf
    model.affine_constraint_cache_set[inner_ci] = set
    return outer_ci
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.ScalarAffineFunction{T},
    set::MOI.AbstractScalarSet,
) where {T}
    if !function_has_parameters(f)
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

function MOI.supports(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    tp::Type{MOI.VariableIndex},
)
    return MOI.supports(model.optimizer, attr, tp)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    v::MOI.VariableIndex,
    val,
)
    if is_variable_in_model(model, v)
        MOI.set(model.optimizer, attr, v, val)
    else
        error("$attr is not supported for parameters")
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    v::MOI.VariableIndex,
)
    if is_variable_in_model(model, v)
        return MOI.get(model.optimizer, attr, model.variables[v])
    else
        error("$attr is not supported for parameters")
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

# We need a CachingOptimizer fallback to
# get ParameterValue working correctly on JuMP
# TODO: Think of a better solution for this

function MOI.set(
    opt::MOI.Utilities.CachingOptimizer,
    ::ParameterValue,
    var::MOI.VariableIndex,
    val::Float64,
)
    ci = MOI.ConstraintIndex{MOI.VariableIndex,Parameter}(var.value)
    set = MOI.set(opt, MOI.ConstraintSet(), ci, Parameter(val))
    return nothing
end

function MOI.set(
    model::Optimizer,
    ::ParameterValue,
    var::MOI.VariableIndex,
    val::Float64,
)
    ci = MOI.ConstraintIndex{MOI.VariableIndex,Parameter}(var.value)
    set = MOI.set(model, MOI.ConstraintSet(), ci, Parameter(val))
    return nothing
end

function MOI.set(
    opt::MOI.Utilities.CachingOptimizer,
    ::ParameterValue,
    vi::MOI.VariableIndex,
    val::Real,
)
    return MOI.set(opt, ParameterValue(), vi, convert(Float64, val))
end

function MOI.set(
    model::Optimizer,
    ::ParameterValue,
    vi::MOI.VariableIndex,
    val::Real,
)
    return MOI.set(model, ParameterValue(), vi, convert(Float64, val))
end

function MOI.get(
    opt::MOI.Utilities.CachingOptimizer,
    ::ParameterValue,
    var::MOI.VariableIndex,
)
    ci = MOI.ConstraintIndex{MOI.VariableIndex,Parameter}(var.value)
    set = MOI.get(opt, MOI.ConstraintSet(), ci)
    return set.val
end

function MOI.get(model::Optimizer, ::ParameterValue, var::MOI.VariableIndex)
    return model.parameters[p_idx(var)]
end

"""
    ConstraintsInterpretation <: MOI.AbstractOptimizerAttribute

Attribute to define how [`POI.Optimizer`](@ref) should interpret constraints.

- `POI.ONLY_CONSTRAINTS`: Only interpret `ScalarAffineFunction` constraints as linear constraints
  If an expression such as `x >= p1 + p2` appears it will be trated like a new constraint.
  **This is the default behaviour of [`POI.Optimizer`](@ref)**

- `POI.ONLY_BOUNDS`: Only interpret `ScalarAffineFunction` constraints as a variable bound.
  This is valid for constraints such as `x >= p` or `x >= p1 + p2`. If a constraint `x1 + x2 >= p` appears,
  which is not a valid variable bound it will throw an error.

- `POI.BOUNDS_AND_CONSTRAINTS`: Interpret `ScalarAffineFunction` constraints as a variable bound if they
  are a valid variable bound, i.e., `x >= p` or `x >= p1 + p2` and interpret them as linear constraints
  otherwise.

# Example

```julia
MOI.set(model, POI.InterpretConstraintsAsBounds(), POI.ONLY_BOUNDS)
MOI.set(model, POI.InterpretConstraintsAsBounds(), POI.ONLY_CONSTRAINTS)
MOI.set(model, POI.InterpretConstraintsAsBounds(), POI.BOUNDS_AND_CONSTRAINTS)
```
"""
struct ConstraintsInterpretation <: MOI.AbstractOptimizerAttribute end

function MOI.set(
    model::Optimizer,
    ::ConstraintsInterpretation,
    value::ConstraintsInterpretationCode,
)
    return model.constraints_interpretation = value
end

function empty_objective_function_caches!(model::Optimizer{T}) where {T}
    model.affine_objective_cache = nothing
    model.quadratic_objective_cache = nothing
    model.original_objective_cache = MOI.Utilities.ObjectiveContainer{T}()
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction,
    f::MOI.ScalarAffineFunction{T},
) where {T}
    # clear previously defined objetive function cache
    empty_objective_function_caches!(model)
    if !function_has_parameters(f)
        MOI.set(model.optimizer, attr, f)
    else
        pf = ParametricAffineFunction(f)
        update_cache!(pf, model)
        MOI.set(
            model.optimizer,
            attr,
            current_function(pf),
        )
        model.affine_objective_cache = pf
    end
    MOI.set(model.original_objective_cache, attr, f)
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{F},
    f::F,
) where {F<:MOI.ScalarQuadraticFunction{T}} where {T}
    # clear previously defined objetive function cache
    empty_objective_function_caches!(model)
    if !function_has_parameters(f)
        MOI.set(model.optimizer, attr, f)
    else
        pf = ParametricQuadraticFunction(f)
        cache_multiplicative_params!(model, pf)
        update_cache!(pf, model)
        func = current_function(pf)
        MOI.set(
            model.optimizer,
            MOI.ObjectiveFunction{
                (is_affine(func) ?
                    MOI.ScalarAffineFunction{T} :
                    MOI.ScalarQuadraticFunction{T})
            }(),
            # func,
            (is_affine(func) ?
                MOI.ScalarAffineFunction(func.affine_terms, func.constant) :
                func),
        )
        model.quadratic_objective_cache = pf
    end
    MOI.set(model.original_objective_cache, attr, f)
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction,
    v::MOI.VariableIndex,
)
    if is_parameter(v)
        error("Cannot use a parameter as objective function alone")
    elseif !is_variable_in_model(model, v)
        error("Variable not in the model")
    end
    MOI.set(model.optimizer, attr, model.variables[v])
    MOI.set(model.original_objective_cache, attr, v)
    return
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

function MOI.get(
    model::Optimizer,
    attr::AT,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S},
) where {
    AT<:Union{
        MOI.ConstraintPrimal,
        MOI.ConstraintDual,
        MOI.ConstraintBasisStatus,
    },
    T,
    S<:MOI.AbstractScalarSet,
}
    moi_ci = get(model.affine_outer_to_inner, c, c)
    return MOI.get(model.optimizer, attr, moi_ci)
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
)
    if function_has_parameters(f)
        error("VectorOfVariables does not allow parameters")
    end
    return MOI.add_constraint(model.optimizer, f, set)
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.VectorAffineFunction{T},
    set::MOI.AbstractVectorSet,
) where {T}
    if !function_has_parameters(f)
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
    pf = ParametricVectorAffineFunction(f)
    # cache_set_constant!(pf, set) # there is no constant is vector sets
    update_cache!(pf, model)
    inner_ci = MOI.add_constraint(
        model.optimizer,
        current_function(pf),
        set,
    )
    model.vector_affine_constraint_cache[inner_ci] = pf
    return inner_ci
end

function add_constraint_with_parameters_on_function(
    model::Optimizer,
    f::MOI.ScalarQuadraticFunction{T},
    s::S,
) where {T,S<:MOI.AbstractScalarSet}
    pf = ParametricQuadraticFunction(f)
    cache_multiplicative_params!(model, pf)
    cache_set_constant!(pf, s)
    update_cache!(pf, model)

    func = current_function(pf)
    f_quad = if !is_affine(func)
        fq = func
        inner_ci = MOI.Utilities.normalize_and_add_constraint(model.optimizer, fq, s)
        model.last_quad_add_added += 1
        outer_ci = MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S}(
            model.last_quad_add_added,
        )
        model.quadratic_outer_to_inner[outer_ci] = inner_ci
        # model.outer_to_inner_map[outer_ci] = inner_ci
    else
        fa = MOI.ScalarAffineFunction(func.affine_terms, func.constant)
        inner_ci = MOI.Utilities.normalize_and_add_constraint(model.optimizer, fa, s)
        model.last_quad_add_added += 1
        outer_ci = MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S}(
            model.last_quad_add_added,
        )
        # This part is used to remember that ci came from a quadratic function
        # It is particularly useful because sometimes the constraint mutates
        model.quadratic_outer_to_inner[outer_ci] = inner_ci
        # model.outer_to_inner_map[outer_ci] = inner_ci
    end
    model.quadratic_constraint_cache[inner_ci] = pf
    model.quadratic_constraint_cache_set[inner_ci] = s
    return outer_ci
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.ScalarQuadraticFunction{T},
    set::MOI.AbstractScalarSet,
) where {T}
    if !function_has_parameters(f)
        return MOI.add_constraint(model.optimizer, f, set)
    else
        return add_constraint_with_parameters_on_function(model, f, set)
    end
end

function MOI.delete(model::Optimizer, v::MOI.VariableIndex)
    delete!(model.variables, v)
    MOI.delete(model.optimizer, v)
    MOI.delete(model.original_objective_cache, v)
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
    F<:Union{MOI.VariableIndex,MOI.VectorOfVariables},
    S<:MOI.AbstractSet,
}
    MOI.delete(model.optimizer, c)
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{F,S},
) where {
    F<:MOI.VectorAffineFunction,
    S<:MOI.AbstractSet,
}
    MOI.delete(model.optimizer, c)
    deleteat!(model.vector_affine_constraint_cache, c)
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

struct QuadraticObjectiveCoef <: MOI.AbstractModelAttribute end

function _evaluate_parametric_expression(model::Optimizer, p::MOI.VariableIndex)
    return model.parameters[p_idx(p)]
end

function _evaluate_parametric_expression(
    model::Optimizer,
    fparam::MOI.ScalarAffineFunction{T},
) where {T}
    constant = fparam.constant
    terms = fparam.terms
    evaluated_parameter_expression = zero(T)
    for term in terms
        coef = term.coefficient
        p = term.variable
        evaluated_parameter_expression += coef * model.parameters[p_idx(p)]
        evaluated_parameter_expression += constant
    end
    return evaluated_parameter_expression
end

function set_quadratic_product_in_obj!(model::Optimizer{T}) where {T}
    n = length(model.quadratic_objective_cache_product)
    if n == 0
        return
    end

    f = if model.affine_objective_cache !== nothing
        current_function(model.affine_objective_cache)
    elseif model.quadratic_objective_cache !== nothing
        current_function(model.quadratic_objective_cache)
    else
        F = MOI.get(model.original_objective_cache, MOI.ObjectiveFunctionType())
        MOI.get(model.original_objective_cache, MOI.ObjectiveFunction{F}())
    end
    F = typeof(f)

    quadratic_prods_vector = MOI.ScalarQuadraticTerm{T}[]
    sizehint!(quadratic_prods_vector, n)

    for ((x, y), fparam) in model.quadratic_objective_cache_product
        # x, y = prod_var
        evaluated_fparam = _evaluate_parametric_expression(model, fparam)
        push!(quadratic_prods_vector,
            MOI.ScalarQuadraticTerm(evaluated_fparam, x, y))
    end

    f_new = if F <: MOI.VariableIndex
        MOI.ScalarQuadraticFunction(
            quadratic_prods_vector,
            MOI.ScalarAffineTerm{T}[MOI.ScalarAffineTerm{T}(1.0, f)],
            0.0,
        )
    elseif F <: MOI.ScalarAffineFunction{T}
        MOI.ScalarQuadraticFunction(
            quadratic_prods_vector,
            f.terms,
            f.constant,
        )
    elseif  F <: MOI.ScalarQuadraticFunction{T}
        quadratic_terms = vcat(f.quadratic_terms, quadratic_prods_vector)
        MOI.ScalarQuadraticFunction(
            quadratic_terms,
            f.affine_terms,
            f.constant,
        )
    end

    MOI.set(
        model.optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}(),
        f_new,
    )

    return
end

function MOI.set(
    model::Optimizer,
    ::QuadraticObjectiveCoef,
    (x1, x2)::Tuple{MOI.VariableIndex,MOI.VariableIndex},
    ::Nothing,
)
    if x1.value > x2.value
        aux = x1
        x1 = x2
        x2 = aux
    end
    delete!(model.quadratic_objective_cache_product, (x1, x2))
    return
end

function MOI.set(
    model::Optimizer,
    ::QuadraticObjectiveCoef,
    (x1, x2)::Tuple{MOI.VariableIndex,MOI.VariableIndex},
    f_param::Union{MOI.VariableIndex,MOI.ScalarAffineFunction{T}},
) where {T}
    if x1.value > x2.value
        aux = x1
        x1 = x2
        x2 = aux
    end
    model.quadratic_objective_cache_product[(x1, x2)] = f_param
    return
end

function MOI.get(
    model::Optimizer,
    ::QuadraticObjectiveCoef,
    (x1, x2)::Tuple{MOI.VariableIndex,MOI.VariableIndex},
)
    if x1.value > x2.value
        aux = x1
        x1 = x2
        x2 = aux
    end
    if haskey(model.quadratic_objective_cache_product, (x1, x2))
        return model.quadratic_objective_cache_product[(x1, x2)]
    else
        throw(
            ErrorException(
                "Parameter not set in product of variables ($x1,$x2)",
            ),
        )
    end
end

function _poi_default_copy_to(dest::T, src::MOI.ModelLike) where {T}
    if !MOI.supports_incremental_interface(dest)
        error("Model $(typeof(dest)) does not support copy_to.")
    end
    MOI.empty!(dest)
    vis_src = MOI.get(src, MOI.ListOfVariableIndices())
    index_map = MOI.IndexMap()
    # The `NLPBlock` assumes that the order of variables does not change (#849)
    # Therefore, all VariableIndex and VectorOfVariable constraints are added
    # seprately, and no variables constrained-on-creation are added.

    # This is not valid for NLPs with Parameters, they should enter
    has_nlp = MOI.NLPBlock() in MOI.get(src, MOI.ListOfModelAttributesSet())
    constraints_not_added = if has_nlp
        vcat(
            Any[
                MOI.get(src, MOI.ListOfConstraintIndices{F,S}()) for
                (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent()) if
                MOI.Utilities._is_variable_function(F) && S != Parameter
            ],
            Any[MOI.Utilities._try_constrain_variables_on_creation(
                dest,
                src,
                index_map,
                Parameter,
            )],
        )
    else
        Any[
            MOI.Utilities._try_constrain_variables_on_creation(
                dest,
                src,
                index_map,
                S,
            ) for S in MOI.Utilities.sorted_variable_sets_by_cost(dest, src)
        ]
    end
    MOI.Utilities._copy_free_variables(dest, index_map, vis_src)
    # Copy variable attributes
    MOI.Utilities.pass_attributes(dest, src, index_map, vis_src)
    # Copy model attributes
    MOI.Utilities.pass_attributes(dest, src, index_map)
    # Copy constraints
    MOI.Utilities._pass_constraints(dest, src, index_map, constraints_not_added)
    MOI.Utilities.final_touch(dest, index_map)
    return index_map
end

function MOI.Utilities.default_copy_to(
    dest::MOI.Bridges.LazyBridgeOptimizer{Optimizer{T,OT}},
    src::MOI.ModelLike,
) where {T,OT}
    return _poi_default_copy_to(dest, src)
end

function MOI.Utilities.default_copy_to(
    dest::Optimizer{T,OT},
    src::MOI.ModelLike,
) where {T,OT}
    return _poi_default_copy_to(dest, src)
end

function MOI.optimize!(model::Optimizer)
    if !isempty(model.updated_parameters)
        update_parameters!(model)
    end
    if !isempty(model.quadratic_objective_cache_product)
        set_quadratic_product_in_obj!(model)
    end
    MOI.optimize!(model.optimizer)
    if MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION &&
       model.evaluate_duals
        @warn "Dual solution not available, ignoring `evaluate_duals`"
    elseif model.evaluate_duals
        compute_dual_of_parameters!(model)
    end
    return
end

end # module
