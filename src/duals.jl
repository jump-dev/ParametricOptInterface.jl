# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function create_param_dual_cum_sum(model::Optimizer)
    return zeros(model.number_of_parameters_in_model)
end

function calculate_dual_of_parameters(model::Optimizer)
    param_dual_cum_sum = create_param_dual_cum_sum(model)
    update_duals_with_affine_constraint_cache!(param_dual_cum_sum, model)
    update_duals_with_quadratic_constraint_cache!(param_dual_cum_sum, model)
    update_duals_in_affine_objective!(
        param_dual_cum_sum,
        model.affine_objective_cache,
    )
    update_duals_in_quadratic_objective!(
        param_dual_cum_sum,
        model.quadratic_objective_cache_pc,
    )
    empty_and_feed_duals_to_model(model, param_dual_cum_sum)
    return model
end

function update_duals_with_affine_constraint_cache!(
    param_dual_cum_sum::Vector{Float64},
    model::Optimizer,
)
    for (F, S) in keys(model.affine_constraint_cache.dict)
        affine_constraint_cache_inner = model.affine_constraint_cache[F, S]
        affine_added_cache_inner = model.affine_added_cache[F, S]
        if !isempty(affine_constraint_cache_inner)
            update_duals_with_affine_constraint_cache!(
                param_dual_cum_sum,
                model.optimizer,
                affine_constraint_cache_inner,
                affine_added_cache_inner,
            )
        end
    end
    return
end

function update_duals_with_affine_constraint_cache!(
    param_dual_cum_sum::Vector{Float64},
    optimizer::OT,
    affine_constraint_cache_inner::DoubleDictInner{F,S,V1},
    affine_added_cache_inner::DoubleDictInner{F,S,V2},
) where {OT,F,S,V1,V2}
    for (ci, param_array) in affine_constraint_cache_inner
        calculate_parameters_in_ci!(
            param_dual_cum_sum,
            optimizer,
            param_array,
            affine_added_cache_inner[ci],
        )
    end
    return
end

function update_duals_with_quadratic_constraint_cache!(
    param_dual_cum_sum::Vector{Float64},
    model::Optimizer,
)
    for (F, S) in keys(model.quadratic_constraint_cache_pc.dict)
        quadratic_constraint_cache_pc_inner =
            model.quadratic_constraint_cache_pc[F, S]
        if !isempty(quadratic_constraint_cache_pc_inner)
            update_duals_with_quadratic_constraint_cache!(
                param_dual_cum_sum,
                model,
                quadratic_constraint_cache_pc_inner,
            )
        end
    end
    return
end

function update_duals_with_quadratic_constraint_cache!(
    param_dual_cum_sum::Vector{Float64},
    model::Optimizer,
    quadratic_constraint_cache_pc_inner::DoubleDictInner{F,S,V},
) where {F,S,V}
    for (poi_ci, param_array) in quadratic_constraint_cache_pc_inner
        moi_ci = model.moi_quadratic_to_poi_affine_map[poi_ci]
        calculate_parameters_in_ci!(
            param_dual_cum_sum,
            model.optimizer,
            param_array,
            moi_ci,
        )
    end
    return
end

function calculate_parameters_in_ci!(
    param_dual_cum_sum::Vector{Float64},
    optimizer::OT,
    param_array::Vector{MOI.ScalarAffineTerm{T}},
    ci::CI,
) where {OT,CI,T}
    cons_dual = MOI.get(optimizer, MOI.ConstraintDual(), ci)

    for param in param_array
        param_dual_cum_sum[param.variable.value-PARAMETER_INDEX_THRESHOLD] -=
            cons_dual * param.coefficient
    end
    return
end

# this one seem to be the same as the next
function update_duals_in_affine_objective!(
    param_dual_cum_sum::Vector{Float64},
    affine_objective_cache::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    for param in affine_objective_cache
        param_dual_cum_sum[param.variable.value-PARAMETER_INDEX_THRESHOLD] -=
            param.coefficient
    end
    return
end

function update_duals_in_quadratic_objective!(
    param_dual_cum_sum::Vector{Float64},
    quadratic_objective_cache_pc::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    for param in quadratic_objective_cache_pc
        param_dual_cum_sum[param.variable.value-PARAMETER_INDEX_THRESHOLD] -=
            param.coefficient
    end
    return
end

function empty_and_feed_duals_to_model(
    model::Optimizer,
    param_dual_cum_sum::Vector{Float64},
)
    empty!(model.dual_value_of_parameters)
    model.dual_value_of_parameters = zeros(model.number_of_parameters_in_model)
    for (vi_val_minus_threshold, param_dual) in enumerate(param_dual_cum_sum)
        model.dual_value_of_parameters[vi_val_minus_threshold] = param_dual
    end
    return
end

"""
    ParameterDual <: MOI.AbstractVariableAttribute

Attribute defined to get the dual values associated to parameters

# Example

```julia
MOI.get(model, POI.ParameterValue(), p)
```
"""
struct ParameterDual <: MOI.AbstractVariableAttribute end

MOI.is_set_by_optimize(::ParametricOptInterface.ParameterDual) = true

function MOI.get(model::Optimizer, ::ParameterDual, v::MOI.VariableIndex)
    if !is_additive(
        model,
        MOI.ConstraintIndex{MOI.VariableIndex,Parameter}(v.value),
    )
        error("Cannot calculate the dual of a multiplicative parameter")
    end
    return model.dual_value_of_parameters[v.value-PARAMETER_INDEX_THRESHOLD]
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDual,
    cp::MOI.ConstraintIndex{MOI.VariableIndex,Parameter},
)
    if !is_additive(model, cp)
        error("Cannot calculate the dual of a multiplicative parameter")
    end
    return model.dual_value_of_parameters[cp.value-PARAMETER_INDEX_THRESHOLD]
end

function is_additive(model::Optimizer, cp::MOI.ConstraintIndex)
    if cp.value in model.multiplicative_parameters
        return false
    end
    return true
end
