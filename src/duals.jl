function create_param_dual_cum_sum(model::ParametricOptimizer)
    return zeros(model.number_of_parameters_in_model)
end

function calculate_dual_of_parameters(model::ParametricOptimizer)
    param_dual_cum_sum = create_param_dual_cum_sum(model)
    update_duals_with_affine_constraint_cache!(param_dual_cum_sum, model)
    update_duals_with_quadratic_constraint_cache!(param_dual_cum_sum, model)
    update_duals_in_affine_objective!(param_dual_cum_sum, model.affine_objective_cache)
    update_duals_in_quadratic_objective!(param_dual_cum_sum, model.quadratic_objective_cache_pc)
    empty_and_feed_duals_to_model(model, param_dual_cum_sum)
    return model
end

function update_duals_with_affine_constraint_cache!(param_dual_cum_sum::Vector{Float64}, model::POI.ParametricOptimizer)
    for S in SUPPORTED_SETS
        affine_constraint_cache_inner = model.affine_constraint_cache[MOI.ScalarAffineFunction{Float64}, S]
        if !isempty(affine_constraint_cache_inner)
            update_duals_with_affine_constraint_cache!(param_dual_cum_sum, model.optimizer, affine_constraint_cache_inner)
        end
    end
    return
end

function update_duals_with_affine_constraint_cache!(param_dual_cum_sum::Vector{Float64}, optimizer::OT, affine_constraint_cache_inner::DD.WithType{F, S}) where {OT,F,S}
    for (ci, param_array) in affine_constraint_cache_inner
        calculate_parameters_in_ci!(param_dual_cum_sum, optimizer, param_array, ci)
    end
    return
end

function update_duals_with_quadratic_constraint_cache!(param_dual_cum_sum::Vector{Float64}, model::POI.ParametricOptimizer)
    for S in SUPPORTED_SETS
        quadratic_constraint_cache_pc_inner = model.quadratic_constraint_cache_pc[MOI.ScalarQuadraticFunction{Float64}, S]
        if !isempty(quadratic_constraint_cache_pc_inner)
            update_duals_with_quadratic_constraint_cache!(param_dual_cum_sum, model, quadratic_constraint_cache_pc_inner)
        end
    end
    return
end

function update_duals_with_quadratic_constraint_cache!(param_dual_cum_sum::Vector{Float64}, model::ParametricOptimizer, quadratic_constraint_cache_pc_inner::DD.WithType{F, S}) where {F,S}
    for (poi_ci, param_array) in quadratic_constraint_cache_pc_inner
        moi_ci = model.quadratic_added_cache[poi_ci]
        calculate_parameters_in_ci!(param_dual_cum_sum, model.optimizer, param_array, moi_ci)
    end
    return
end

function calculate_parameters_in_ci!(param_dual_cum_sum::Vector{Float64}, optimizer::OT, param_array::Vector{MOI.ScalarAffineTerm{T}}, ci::CI) where {OT, CI, T}
    cons_dual = MOI.get(optimizer, MOI.ConstraintDual(), ci)

    for param in param_array
        param_dual_cum_sum[param.variable_index.value - PARAMETER_INDEX_THRESHOLD] += cons_dual*param.coefficient
    end
    return
end

function update_duals_in_affine_objective!(param_dual_cum_sum::Vector{Float64}, affine_objective_cache::Vector{MOI.ScalarAffineTerm{T}}) where T
    for param in affine_objective_cache
        param_dual_cum_sum[param.variable_index.value - PARAMETER_INDEX_THRESHOLD] += param.coefficient
    end
    return
end

function update_duals_in_quadratic_objective!(param_dual_cum_sum::Vector{Float64}, quadratic_objective_cache_pc::Vector{MOI.ScalarAffineTerm{T}}) where T
    for param in quadratic_objective_cache_pc
        param_dual_cum_sum[param.variable_index.value - PARAMETER_INDEX_THRESHOLD] += param.coefficient
    end
    return
end

function empty_and_feed_duals_to_model(model::ParametricOptimizer, param_dual_cum_sum::Vector{Float64})
    empty!(model.dual_value_of_parameters)
    model.dual_value_of_parameters = zeros(model.number_of_parameters_in_model)
    for (vi_val_minus_threshold, param_dual) in enumerate(param_dual_cum_sum)
        model.dual_value_of_parameters[vi_val_minus_threshold] = param_dual
    end
    return
end

struct ParameterDual <: MOI.AbstractVariableAttribute end

function MOI.get(model::ParametricOptimizer, ::ParameterDual, vi_val::Int64)
    if !is_additive(model, MOI.ConstraintIndex{MOI.SingleVariable, Parameter}(vi_val))
        error("Cannot calculate the dual of a multiplicative parameter")
    end
    return model.dual_value_of_parameters[vi_val - PARAMETER_INDEX_THRESHOLD]
end

function MOI.get(model::ParametricOptimizer, ::MOI.ConstraintDual, cp::MOI.ConstraintIndex{MOI.SingleVariable,POI.Parameter})
    if !is_additive(model, cp)
        error("Cannot calculate the dual of a multiplicative parameter")
    end
    return model.dual_value_of_parameters[cp.value - PARAMETER_INDEX_THRESHOLD]
end

function MOI.get(model::ParametricOptimizer, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex)
    return MOI.get(model.optimizer, attr, ci)
end

function is_additive(model::ParametricOptimizer, cp::MOI.ConstraintIndex)
    if cp.value in model.multiplicative_parameters
        return false
    end
    return true
end