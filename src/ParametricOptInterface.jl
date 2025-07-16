# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module ParametricOptInterface

using MathOptInterface

const MOI = MathOptInterface

@enum ConstraintsInterpretationCode ONLY_CONSTRAINTS ONLY_BOUNDS BOUNDS_AND_CONSTRAINTS

#
# Parameter Index
#

const SIMPLE_SCALAR_SETS{T} =
    Union{MOI.LessThan{T},MOI.GreaterThan{T},MOI.EqualTo{T}}

const PARAMETER_INDEX_THRESHOLD = Int64(4_611_686_018_427_387_904) # div(typemax(Int64),2)+1

struct ParameterIndex
    index::Int64
end

function p_idx(vi::MOI.VariableIndex)::ParameterIndex
    return ParameterIndex(vi.value - PARAMETER_INDEX_THRESHOLD)
end

function v_idx(pi::ParameterIndex)::MOI.VariableIndex
    return MOI.VariableIndex(pi.index + PARAMETER_INDEX_THRESHOLD)
end

function p_val(vi::MOI.VariableIndex)::Int64
    return vi.value - PARAMETER_INDEX_THRESHOLD
end

function p_val(ci::MOI.ConstraintIndex)::Int64
    return ci.value - PARAMETER_INDEX_THRESHOLD
end

#
# MOI Special structure helpers
#

# Utilities for using a CleverDict in Parameters
function MOI.Utilities.CleverDicts.index_to_key(
    ::Type{ParameterIndex},
    index::Int64,
)
    return ParameterIndex(index)
end

function MOI.Utilities.CleverDicts.key_to_index(key::ParameterIndex)
    return key.index
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

#
# parametric functions
#

include("parametric_functions.jl")

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

    # mapping of all constraints: necessary for getters
    constraint_outer_to_inner::DoubleDict{MOI.ConstraintIndex}

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
    # Vector quadratic function data
    vector_quadratic_constraint_cache::DoubleDict{ParametricVectorQuadraticFunction{T}}

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
    quadratic_objective_cache_product_changed::Bool

    # vector affine function data
    # vector_constraint_cache::DoubleDict{Vector{MOI.VectorAffineTerm{T}}}
    # Clever cache of data (inner key)
    vector_affine_constraint_cache::DoubleDict{
        ParametricVectorAffineFunction{T},
    }

    #
    multiplicative_parameters_pv::Set{Int64}
    multiplicative_parameters_pp::Set{Int64}
    dual_value_of_parameters::Vector{T}

    # params
    evaluate_duals::Bool
    number_of_parameters_in_model::Int64
    constraints_interpretation::ConstraintsInterpretationCode
    save_original_objective_and_constraints::Bool

    parameters_in_conflict::Set{MOI.VariableIndex}

    # extension data
    ext::Dict{Symbol,Any}
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
            DoubleDict{MOI.ConstraintIndex}(),
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
            DoubleDict{ParametricVectorQuadraticFunction{T}}(),
            # objective
            nothing,
            nothing,
            # nothing,
            MOI.Utilities.ObjectiveContainer{T}(),
            Dict{
                Tuple{MOI.VariableIndex,MOI.VariableIndex},
                MOI.AbstractFunction,
            }(),
            false,
            # vec affine
            # DoubleDict{Vector{MOI.VectorAffineTerm{T}}}(),
            DoubleDict{ParametricVectorAffineFunction{T}}(),
            # other
            Set{Int64}(),
            Set{Int64}(),
            Vector{T}(),
            evaluate_duals,
            0,
            ONLY_CONSTRAINTS,
            save_original_objective_and_constraints,
            Set{MOI.VariableIndex}(),
            Dict{Symbol,Any}(),
        )
    end
end

function _next_variable_index!(model::Optimizer)
    return model.last_variable_index_added += 1
end

function _next_parameter_index!(model::Optimizer)
    return model.last_parameter_index_added += 1
end

function _update_number_of_parameters!(model::Optimizer)
    return model.number_of_parameters_in_model += 1
end

function _parameter_in_model(model::Optimizer, v::MOI.VariableIndex)
    return PARAMETER_INDEX_THRESHOLD <
           v.value <=
           model.last_parameter_index_added
end

function _variable_in_model(model::Optimizer, v::MOI.VariableIndex)
    return 0 < v.value <= model.last_variable_index_added
end

include("duals.jl")
include("update_parameters.jl")
include("MOI_wrapper.jl")

end # module
