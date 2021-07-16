function create_param_dual_cum_sum(model::ParametricOptimizer)
    param_dual_cum_sum = Dict{Int, Float64}()
    for vi in keys(model.parameters)
        param_dual_cum_sum[vi.value] = 0.0
    end
    return param_dual_cum_sum
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

function update_duals_with_affine_constraint_cache!(param_dual_cum_sum::Dict{Int, Float64}, model::POI.ParametricOptimizer)
    for S in SUPPORTED_SETS
        affine_constraint_cache_inner = model.affine_constraint_cache[MOI.ScalarAffineFunction{Float64}, S]
        if length(affine_constraint_cache_inner) > 0
            update_duals_with_affine_constraint_cache!(param_dual_cum_sum, model.optimizer, affine_constraint_cache_inner)
        end
    end
    return
end

function update_duals_with_affine_constraint_cache!(param_dual_cum_sum::Dict{Int, Float64}, optimizer::OT, affine_constraint_cache_inner::DD.WithType{F, S}) where {OT,F,S}
    for (ci, param_array) in affine_constraint_cache_inner
        calculate_parameters_in_ci!(param_dual_cum_sum, optimizer, param_array, ci)
    end
    return
end

function update_duals_with_quadratic_constraint_cache!(param_dual_cum_sum::Dict{Int, Float64}, model::POI.ParametricOptimizer)
    for S in SUPPORTED_SETS
        quadratic_constraint_cache_pc_inner = model.quadratic_constraint_cache_pc[MOI.ScalarQuadraticFunction{Float64}, S]
        if length(quadratic_constraint_cache_pc_inner) > 0
            update_duals_with_quadratic_constraint_cache!(param_dual_cum_sum, model, quadratic_constraint_cache_pc_inner)
        end
    end
    return
end

function update_duals_with_quadratic_constraint_cache!(param_dual_cum_sum::Dict{Int, Float64}, model::ParametricOptimizer, quadratic_constraint_cache_pc_inner::DD.WithType{F, S}) where {F,S}
    for (poi_ci, param_array) in quadratic_constraint_cache_pc_inner
        moi_ci = model.quadratic_added_cache[poi_ci]
        calculate_parameters_in_ci!(param_dual_cum_sum, model.optimizer, param_array, moi_ci)
    end
    return
end

function calculate_parameters_in_ci!(param_dual_cum_sum::Dict{Int, Float64}, optimizer::OT, param_array::Vector{MOI.ScalarAffineTerm{T}}, ci::CI) where {OT, CI, T}
    cons_dual = MOI.get(optimizer, MOI.ConstraintDual(), ci)

    for param in param_array
        param_dual_cum_sum[param.variable_index.value] += cons_dual*param.coefficient
    end
    return
end

function update_duals_in_affine_objective!(param_dual_cum_sum::Dict{Int, Float64}, affine_objective_cache::Vector{MOI.ScalarAffineTerm{T}}) where T
    for param in affine_objective_cache
        param_dual_cum_sum[param.variable_index.value] += param.coefficient
    end
    return
end

function update_duals_in_quadratic_objective!(param_dual_cum_sum::Dict{Int, Float64}, quadratic_objective_cache_pc::Vector{MOI.ScalarAffineTerm{T}}) where T
    for param in quadratic_objective_cache_pc
        param_dual_cum_sum[param.variable_index.value] += param.coefficient
    end
    return
end

function empty_and_feed_duals_to_model(model::ParametricOptimizer, param_dual_cum_sum::Dict{Int, Float64})
    empty!(model.dual_value_of_parameters)
    for (vi_val, param_dual) in param_dual_cum_sum
        model.dual_value_of_parameters[MOI.VariableIndex(vi_val)] = param_dual
    end
    return
end

function MOI.get(model::ParametricOptimizer, ::MOI.ConstraintDual, cp::MOI.ConstraintIndex{MOI.SingleVariable,POI.Parameter})
    if !is_additive(model, cp)
        error("Cannot calculate the dual of a multiplicative parameter")
    end
    return model.dual_value_of_parameters[MOI.VariableIndex(cp.value)]
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

function update_parameter_dual(optimizer::OP, affine_constraint_cache_inner::DD.WithType{F, S}, value::Int) where {OP, F, S}
    param_dual_affine_constraint = zero(Float64)
    for (ci::MOI.ConstraintIndex{F, S}, param_array) in affine_constraint_cache_inner
        param_dual_affine_constraint += calculate_parameter_in_ci(optimizer, param_array, ci, value)::Float64
    end
    return param_dual_affine_constraint
end

function calculate_parameter_in_ci(optimizer::OP, param_array::Vector{MOI.ScalarAffineTerm{T}}, ci::CI, value::Int) where {OP, CI, T}
    param_dual_affine_constraint = zero(T)
    for param in param_array
        if value == param.variable_index.value
            cons_dual = MOI.get(optimizer, MOI.ConstraintDual(), ci)
            param_dual_affine_constraint += cons_dual*param.coefficient
        end
    end
    return param_dual_affine_constraint
end