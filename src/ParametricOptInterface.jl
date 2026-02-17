# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module ParametricOptInterface

import MathOptInterface as MOI
import MathOptInterface.Utilities: CleverDicts
import MathOptInterface.Utilities: DoubleDicts

@enum(
    ConstraintsInterpretationCode,
    ONLY_CONSTRAINTS,
    ONLY_BOUNDS,
    BOUNDS_AND_CONSTRAINTS,
)

const PARAMETER_INDEX_THRESHOLD_MAX = typemax(Int64)

const PARAMETER_INDEX_THRESHOLD = div(PARAMETER_INDEX_THRESHOLD_MAX, 2) + 1

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

function CleverDicts.index_to_key(::Type{ParameterIndex}, index::Int64)
    return ParameterIndex(index)
end

CleverDicts.key_to_index(key::ParameterIndex) = key.index

include("cubic_types.jl")
include("cubic_parser.jl")
include("parametric_functions.jl")
include("parametric_cubic_function.jl")

"""
    Optimizer{T}(
        optimizer::Union{MOI.ModelLike,Any};
        evaluate_duals::Bool = true,
        save_original_objective_and_constraints::Bool = true,
        with_bridge_type = nothing,
    )

Create an `Optimizer`, which allows the handling of parameters in an
optimization model.

If `optimizer` is not a `MOI.ModelLike,` the inner optimizer is constructed
using `MOI.instantiate(optimizer; with_bridge_type)`.

The `{T}` type parameter is optional; it defaults to `Float64`.

## Keyword arguments

- `evaluate_duals::Bool`: If `true`, evaluates the dual of parameters. Set it to
  `false` to increase performance when the duals of parameters are not
  necessary. Defaults to `true`.

- `save_original_objective_and_constraints`: If `true` saves the orginal
  function and set of the constraints as well as the original objective function
  inside [`Optimizer`](@ref). This is useful for printing the model but greatly
  increases the memory footprint. Users might want to set it to `false` to
  increase performance in applications where you don't need to query the
  original expressions provided to the model in constraints or in the objective.
  Note that this might break printing or queries such as
  `MOI.get(model, MOI.ConstraintFunction(), c)`. Defaults to `true`.

- `with_bridge_type`: this is ignroed if `optimizer::MOI.ModelLike`, otherwise
  it is passed to `MOI.instantiate`.

## Example

```julia-repl
julia> import ParametricOptInterface as POI

julia> import HiGHS

julia> POI.Optimizer(HiGHS.Optimizer(); evaluate_duals = true)
ParametricOptInterface.Optimizer{Float64, HiGHS.Optimizer}
├ ObjectiveSense: FEASIBILITY_SENSE
├ ObjectiveFunctionType: MOI.ScalarAffineFunction{Float64}
├ NumberOfVariables: 0
└ NumberOfConstraints: 0

julia> POI.Optimizer(
           HiGHS.Optimizer;
           with_bridge_type = Float64,
           evaluate_duals = false,
       )
ParametricOptInterface.Optimizer{Float64, MOIB.LazyBridgeOptimizer{HiGHS.Optimizer}}
├ ObjectiveSense: FEASIBILITY_SENSE
├ ObjectiveFunctionType: MOI.ScalarAffineFunction{Float64}
├ NumberOfVariables: 0
└ NumberOfConstraints: 0
```
"""
mutable struct Optimizer{T,OT<:MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT
    parameters::CleverDicts.CleverDict{
        ParameterIndex,
        T,
        typeof(CleverDicts.key_to_index),
        typeof(CleverDicts.index_to_key),
    }
    parameters_name::Dict{MOI.VariableIndex,String}
    # The updated_parameters dictionary has the same dimension of the
    # parameters dictionary and if the value stored is a NaN is means
    # that the parameter has not been updated.
    updated_parameters::CleverDicts.CleverDict{
        ParameterIndex,
        T,
        typeof(CleverDicts.key_to_index),
        typeof(CleverDicts.index_to_key),
    }
    variables::CleverDicts.CleverDict{
        MOI.VariableIndex,
        MOI.VariableIndex,
        typeof(CleverDicts.key_to_index),
        typeof(CleverDicts.index_to_key),
    }
    last_variable_index_added::Int64
    last_parameter_index_added::Int64
    # mapping of all constraints: necessary for getters
    constraint_outer_to_inner::DoubleDicts.DoubleDict{MOI.ConstraintIndex}
    # affine constraint data
    last_affine_added::Int64
    # Store the map for SAFs (some might be transformed into VI)
    affine_outer_to_inner::DoubleDicts.DoubleDict{MOI.ConstraintIndex}
    # Clever cache of data (inner key)
    affine_constraint_cache::DoubleDicts.DoubleDict{ParametricAffineFunction{T}}
    # Store original constraint set (inner key)
    affine_constraint_cache_set::DoubleDicts.DoubleDict{MOI.AbstractScalarSet}
    # quadratic constraint data
    last_quad_add_added::Int64
    last_vec_quad_add_added::Int64
    # Store the map for SQFs (some might be transformed into SAF)
    # for instance p*p + var -> ScalarAffine(var)
    quadratic_outer_to_inner::DoubleDicts.DoubleDict{MOI.ConstraintIndex}
    vector_quadratic_outer_to_inner::DoubleDicts.DoubleDict{MOI.ConstraintIndex}
    # Clever cache of data (inner key)
    quadratic_constraint_cache::DoubleDicts.DoubleDict{
        ParametricQuadraticFunction{T},
    }
    # Store original constraint set (inner key)
    quadratic_constraint_cache_set::DoubleDicts.DoubleDict{
        MOI.AbstractScalarSet,
    }
    # Vector quadratic function data
    vector_quadratic_constraint_cache::DoubleDicts.DoubleDict{
        ParametricVectorQuadraticFunction{T},
    }
    # Store original constraint set (inner key)
    vector_quadratic_constraint_cache_set::DoubleDicts.DoubleDict{
        MOI.AbstractVectorSet,
    }
    # objective function data
    # Clever cache of data (at most one can be !== nothing)
    affine_objective_cache::Union{Nothing,ParametricAffineFunction{T}}
    quadratic_objective_cache::Union{Nothing,ParametricQuadraticFunction{T}}
    cubic_objective_cache::Union{Nothing,ParametricCubicFunction{T}}
    original_objective_cache::MOI.Utilities.ObjectiveContainer{T}
    # Clever cache of data (inner key)
    vector_affine_constraint_cache::DoubleDicts.DoubleDict{
        ParametricVectorAffineFunction{T},
    }
    multiplicative_parameters_pv::Set{Int64}
    multiplicative_parameters_pp::Set{Int64}
    dual_value_of_parameters::Vector{T}
    evaluate_duals::Bool
    number_of_parameters_in_model::Int64
    constraints_interpretation::ConstraintsInterpretationCode
    save_original_objective_and_constraints::Bool
    parameters_in_conflict::Set{MOI.VariableIndex}
    warn_quad_affine_ambiguous::Bool
    ext::Dict{Symbol,Any}

    function Optimizer{T}(
        optimizer::OT;
        evaluate_duals::Bool = true,
        save_original_objective_and_constraints::Bool = true,
    ) where {T,OT<:MOI.ModelLike}
        return new{T,OT}(
            # optimizer
            optimizer,
            # parameters
            CleverDicts.CleverDict{ParameterIndex,T}(),
            # parameters_name
            Dict{MOI.VariableIndex,String}(),
            # updated_parameters
            CleverDicts.CleverDict{ParameterIndex,T}(),
            # variables
            CleverDicts.CleverDict{MOI.VariableIndex,MOI.VariableIndex}(),
            # last_variable_index_added
            0,
            # last_parameter_index_added
            PARAMETER_INDEX_THRESHOLD,
            # constraint_outer_to_inner
            DoubleDicts.DoubleDict{MOI.ConstraintIndex}(),
            # last_affine_added
            0,
            # affine_outer_to_inner
            DoubleDicts.DoubleDict{MOI.ConstraintIndex}(),
            # affine_constraint_cache
            DoubleDicts.DoubleDict{ParametricAffineFunction{T}}(),
            # affine_constraint_cache_set
            DoubleDicts.DoubleDict{MOI.AbstractScalarSet}(),
            # last_quad_add_added
            0,
            # last_vec_quad_add_added
            0,
            # quadratic_outer_to_inner
            DoubleDicts.DoubleDict{MOI.ConstraintIndex}(),
            # vector_quadratic_outer_to_inner
            DoubleDicts.DoubleDict{MOI.ConstraintIndex}(),
            # quadratic_constraint_cache
            DoubleDicts.DoubleDict{ParametricQuadraticFunction{T}}(),
            # quadratic_constraint_cache_set
            DoubleDicts.DoubleDict{MOI.AbstractScalarSet}(),
            # vector_quadratic_constraint_cache
            DoubleDicts.DoubleDict{ParametricVectorQuadraticFunction{T}}(),
            # vector_quadratic_constraint_cache_set
            DoubleDicts.DoubleDict{MOI.AbstractVectorSet}(),
            # affine_objective_cache
            nothing,
            # quadratic_objective_cache
            nothing,
            # cubic_objective_cache
            nothing,
            # original_objective_cache
            MOI.Utilities.ObjectiveContainer{T}(),
            # vector_affine_constraint_cache
            DoubleDicts.DoubleDict{ParametricVectorAffineFunction{T}}(),
            # multiplicative_parameters_pv
            Set{Int64}(),
            # multiplicative_parameters_pp
            Set{Int64}(),
            # dual_value_of_parameters
            T[],
            # evaluate_duals
            evaluate_duals,
            # number_of_parameters_in_model
            0,
            # constraints_interpretation
            ONLY_CONSTRAINTS,
            # save_original_objective_and_constraints
            save_original_objective_and_constraints,
            # parameters_in_conflict
            Set{MOI.VariableIndex}(),
            # warn_quad_affine_ambiguous
            true,
            # ext
            Dict{Symbol,Any}(),
        )
    end
end

Optimizer(arg; kwargs...) = Optimizer{Float64}(arg; kwargs...)

function Optimizer{T}(
    optimizer_fn;
    with_bridge_type = nothing,
    kwargs...,
) where {T}
    inner = MOI.instantiate(optimizer_fn; with_bridge_type)
    if !MOI.supports_incremental_interface(inner)
        cache = MOI.default_cache(inner, T)
        inner = MOI.Utilities.CachingOptimizer(cache, inner)
    end
    return Optimizer{T}(inner; kwargs...)
end

function _parameter_in_model(model::Optimizer, v::MOI.VariableIndex)
    return _is_parameter(v) && haskey(model.parameters, p_idx(v))
end

include("duals.jl")
include("update_parameters.jl")
include("MOI_wrapper.jl")
include("cubic_objective.jl")

end # module
