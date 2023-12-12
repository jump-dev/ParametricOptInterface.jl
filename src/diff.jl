using DiffOpt

# forward mode

function _affine_constraint_set_forward!(
    model::Optimizer{T},
    affine_constraint_cache_inner,
) where {T}
    for (inner_ci, pf) in affine_constraint_cache_inner
        cte = zero(T)
        terms = MOI.ScalarAffineTerm{T}[]
        sizehint!(terms, 0)
        for term in pf.p
            p = p_idx(term.variable)
            sensitivity = get(model.parameter_input_forward, p, 0.0)
            cte += sensitivity * term.coefficient
        end
        if !iszero(cte)
            MOI.set(
                model.optimizer,
                DiffOpt.ForwardConstraintFunction(),
                inner_ci,
                MOI.ScalarAffineFunction{T}(terms, cte),
            )
        end
    end
    return
end

function _vector_affine_constraint_set_forward!(
    model::Optimizer{T},
    vector_affine_constraint_cache_inner,
) where {T}
    for (inner_ci, pf) in vector_affine_constraint_cache_inner
        cte = zeros(T, length(pf.c))
        terms = MOI.VectorAffineTerm{T}[]
        sizehint!(terms, 0)
        for term in pf.p
            p = p_idx(term.scalar_term.variable)
            sensitivity = get(model.parameter_input_forward, p, 0.0)
            cte[term.output_index] += sensitivity * term.scalar_term.coefficient
        end
        if !iszero(cte)
            MOI.set(
                model.optimizer,
                DiffOpt.ForwardConstraintFunction(),
                inner_ci,
                MOI.VectorAffineFunction{T}(terms, cte),
            )
        end
    end
    return
end

function _quadratic_constraint_set_forward!(
    model::Optimizer{T},
    quadratic_constraint_cache_inner,
) where {T}
    for (inner_ci, pf) in quadratic_constraint_cache_inner
        cte = zero(T)
        terms = MOI.ScalarAffineTerm{T}[]
        # terms_dict = Dict{MOI.VariableIndex,T}() # canonicalize?
        sizehint!(terms, length(pf.pv))
        for term in pf.p
            p = p_idx(term.variable)
            sensitivity = get(model.parameter_input_forward, p, 0.0)
            cte += sensitivity * term.coefficient
        end
        for term in pf.pp
            p_1 = p_idx(term.variable_1)
            p_2 = p_idx(term.variable_2)
            sensitivity_1 = get(model.parameter_input_forward, p_1, 0.0)
            sensitivity_2 = get(model.parameter_input_forward, p_2, 0.0)
            cte +=
                sensitivity_1 * model.parameters[p_2] * term.coefficient /
                ifelse(term.variable_1 === term.variable_2, 2, 1)
            cte +=
                model.parameters[p_1] * sensitivity_2 * term.coefficient /
                ifelse(term.variable_1 === term.variable_2, 2, 1)
        end
        # canonicalize?
        for term in pf.pv
            p = p_idx(term.variable_1)
            sensitivity = get(model.parameter_input_forward, p, NaN)
            if !isnan(sensitivity)
                push!(
                    terms,
                    MOI.ScalarAffineTerm{T}(
                        sensitivity * term.coefficient,
                        term.variable_2,
                    ),
                )
            end
        end
        if !iszero(cte) || !isempty(terms)
            MOI.set(
                model.optimizer,
                DiffOpt.ForwardConstraintFunction(),
                inner_ci,
                MOI.ScalarAffineFunction{T}(terms, cte),
            )
        end
    end
    return
end

function _affine_objective_set_forward!(model::Optimizer{T}) where {T}
    cte = zero(T)
    terms = MOI.ScalarAffineTerm{T}[]
    pf = model.affine_objective_cache
    sizehint!(terms, 0)
    for term in pf.p
        p = p_idx(term.variable)
        sensitivity = get(model.parameter_input_forward, p, 0.0)
        cte += sensitivity * term.coefficient
    end
    if !iszero(cte)
        MOI.set(
            model.optimizer,
            DiffOpt.ForwardObjectiveFunction(),
            MOI.ScalarAffineFunction{T}(terms, cte),
        )
    end
    return
end

function _quadratic_objective_set_forward!(model::Optimizer{T}) where {T}
    cte = zero(T)
    terms = MOI.ScalarAffineTerm{T}[]
    pf = model.quadratic_objective_cache
    sizehint!(terms, length(pf.pv))
    for term in pf.p
        p = p_idx(term.variable)
        sensitivity = get(model.parameter_input_forward, p, 0.0)
        cte += sensitivity * term.coefficient
    end
    for term in pf.pp
        p_1 = p_idx(term.variable_1)
        p_2 = p_idx(term.variable_2)
        sensitivity_1 = get(model.parameter_input_forward, p_1, 0.0)
        sensitivity_2 = get(model.parameter_input_forward, p_2, 0.0)
        cte +=
            sensitivity_1 * model.parameters[p_2] * term.coefficient /
            ifelse(term.variable_1 === term.variable_2, 2, 1)
        cte +=
            model.parameters[p_1] * sensitivity_2 * term.coefficient /
            ifelse(term.variable_1 === term.variable_2, 2, 1)
    end
    # canonicalize?
    for term in pf.pv
        p = p_idx(term.variable_1)
        sensitivity = get(model.parameter_input_forward, p, NaN)
        if !isnan(sensitivity)
            push!(
                terms,
                MOI.ScalarAffineTerm{T}(
                    sensitivity * term.coefficient,
                    term.variable_2,
                ),
            )
        end
    end
    if !iszero(cte) || !isempty(terms)
        MOI.set(
            model.optimizer,
            DiffOpt.ForwardObjectiveFunction(),
            MOI.ScalarAffineFunction{T}(terms, cte),
        )
    end
    return
end

function _empty_input_cache!(model::Optimizer)
    _empty_input_cache!(model.optimizer)
    return
end
function _empty_input_cache!(model::MOI.Bridges.AbstractBridgeOptimizer)
    _empty_input_cache!(model.model)
    return
end
function _empty_input_cache!(model::MOI.Utilities.CachingOptimizer)
    _empty_input_cache!(model.optimizer)
    return
end
function _empty_input_cache!(model::DiffOpt.Optimizer)
    empty!(model.input_cache)
    return
end

function DiffOpt.forward_differentiate!(model::Optimizer{T}) where {T}
    # TODO: add a reset option
    _empty_input_cache!(model)
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(
        model.affine_constraint_cache,
    )
        _affine_constraint_set_forward!(
            model,
            model.affine_constraint_cache[F, S],
        )
    end
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(
        model.vector_affine_constraint_cache,
    )
        _vector_affine_constraint_set_forward!(
            model,
            model.vector_affine_constraint_cache[F, S],
        )
    end
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(
        model.quadratic_constraint_cache,
    )
        _quadratic_constraint_set_forward!(
            model,
            model.quadratic_constraint_cache[F, S],
        )
    end
    if model.affine_objective_cache !== nothing
        _affine_objective_set_forward!(model)
    elseif model.quadratic_objective_cache !== nothing
        _quadratic_objective_set_forward!(model)
    end
    DiffOpt.forward_differentiate!(model.optimizer)
    return
end

struct ForwardParameter <: MOI.AbstractVariableAttribute end

function MOI.set(
    model::Optimizer,
    ::ForwardParameter,
    variable::MOI.VariableIndex,
    value::Number,
)
    if _is_variable(variable)
        error("Trying to set a forward parameter sensitivity for a variable")
    end
    parameter = p_idx(variable)
    model.parameter_input_forward[parameter] = value
    return
end

function MOI.get(
    model::Optimizer,
    attr::DiffOpt.ForwardVariablePrimal,
    variable::MOI.VariableIndex,
)
    if _is_parameter(variable)
        error("Trying to get a forward variable sensitivity for a parameter")
    end
    return MOI.get(model.optimizer, attr, model.variables[variable])
end

# reverse mode

using JuMP

function _affine_constraint_get_reverse!(
    model::Optimizer{T},
    affine_constraint_cache_inner,
) where {T}
    for (inner_ci, pf) in affine_constraint_cache_inner
        if isempty(pf.p)
            continue
        end
        grad_pf_cte = MOI.constant(
            MOI.get(
                model.optimizer,
                DiffOpt.ReverseConstraintFunction(),
                inner_ci,
            ),
        )
        for term in pf.p
            p = p_idx(term.variable)
            value = get!(model.parameter_output_backward, p, 0.0)
            model.parameter_output_backward[p] =
                value + term.coefficient * grad_pf_cte
            # TODO: check sign
        end
    end
    return
end

function _vector_affine_constraint_get_reverse!(
    model::Optimizer{T},
    vector_affine_constraint_cache_inner,
) where {T}
    for (inner_ci, pf) in vector_affine_constraint_cache_inner
        if isempty(pf.p)
            continue
        end
        grad_pf_cte = MOI.constant(
            MOI.get(
                model.optimizer,
                DiffOpt.ReverseConstraintFunction(),
                inner_ci,
            ),
        )
        for term in pf.p
            p = p_idx(term.scalar_term.variable)
            value = get!(model.parameter_output_backward, p, 0.0)
            model.parameter_output_backward[p] =
                value +
                term.scalar_term.coefficient * grad_pf_cte[term.output_index]
        end
    end
    return
end

function _quadratic_constraint_get_reverse!(
    model::Optimizer{T},
    quadratic_constraint_cache_inner,
) where {T}
    for (inner_ci, pf) in quadratic_constraint_cache_inner
        if isempty(pf.p) && isempty(pf.pv) && isempty(pf.pp)
            continue
        end
        grad_pf = MOI.get(
            model.optimizer,
            DiffOpt.ReverseConstraintFunction(),
            inner_ci,
        )
        grad_pf_cte = MOI.constant(grad_pf)
        for term in pf.p
            p = p_idx(term.variable)
            value = get!(model.parameter_output_backward, p, 0.0)
            model.parameter_output_backward[p] =
                value + term.coefficient * grad_pf_cte
        end
        for term in pf.pp
            p_1 = p_idx(term.variable_1)
            p_2 = p_idx(term.variable_2)
            value_1 = get!(model.parameter_output_backward, p_1, 0.0)
            value_2 = get!(model.parameter_output_backward, p_2, 0.0)
            # TODO: why there is no factor of 2 here????
            # ANS: probably because it was SET
            model.parameter_output_backward[p_1] =
                value_1 +
                term.coefficient * grad_pf_cte * model.parameters[p_2] /
                ifelse(term.variable_1 === term.variable_2, 1, 1)
            model.parameter_output_backward[p_2] =
                value_2 +
                term.coefficient * grad_pf_cte * model.parameters[p_1] /
                ifelse(term.variable_1 === term.variable_2, 1, 1)
        end
        for term in pf.pv
            p = p_idx(term.variable_1)
            v = term.variable_2 # check if inner or outer (should be inner)
            value = get!(model.parameter_output_backward, p, 0.0)
            model.parameter_output_backward[p] =
                value + term.coefficient * JuMP.coefficient(grad_pf, v) # * fixed value of the parameter ?
        end
    end
    return
end

function _affine_objective_get_reverse!(model::Optimizer{T}) where {T}
    pf = model.affine_objective_cache
    if isempty(pf.p)
        return
    end
    grad_pf = MOI.get(model.optimizer, DiffOpt.ReverseObjectiveFunction())
    grad_pf_cte = MOI.constant(grad_pf)
    for term in pf.p
        p = p_idx(term.variable)
        value = get!(model.parameter_output_backward, p, 0.0)
        model.parameter_output_backward[p] =
            value + term.coefficient * grad_pf_cte
    end
    return
end
function _quadratic_objective_get_reverse!(model::Optimizer{T}) where {T}
    pf = model.quadratic_objective_cache
    if isempty(pf.p) && isempty(pf.pv) && isempty(pf.pp)
        return
    end
    grad_pf = MOI.get(model.optimizer, DiffOpt.ReverseObjectiveFunction())
    grad_pf_cte = MOI.constant(grad_pf)
    for term in pf.p
        p = p_idx(term.variable)
        value = get!(model.parameter_output_backward, p, 0.0)
        model.parameter_output_backward[p] =
            value + term.coefficient * grad_pf_cte
    end
    for term in pf.pp
        p_1 = p_idx(term.variable_1)
        p_2 = p_idx(term.variable_2)
        value_1 = get!(model.parameter_output_backward, p_1, 0.0)
        value_2 = get!(model.parameter_output_backward, p_2, 0.0)
        model.parameter_output_backward[p_1] =
            value_1 +
            term.coefficient * grad_pf_cte * model.parameters[p_2] /
            ifelse(term.variable_1 === term.variable_2, 2, 1)
        model.parameter_output_backward[p_2] =
            value_2 +
            term.coefficient * grad_pf_cte * model.parameters[p_1] /
            ifelse(term.variable_1 === term.variable_2, 2, 1)
    end
    for term in pf.pv
        p = p_idx(term.variable_1)
        v = term.variable_2 # check if inner or outer (should be inner)
        value = get!(model.parameter_output_backward, p, 0.0)
        model.parameter_output_backward[p] =
            value + term.coefficient * JuMP.coefficient(grad_pf, v) # * fixed value of the parameter ?
    end
    return
end

function DiffOpt.reverse_differentiate!(model::Optimizer)
    # TODO: add a reset option
    DiffOpt.reverse_differentiate!(model.optimizer)
    empty!(model.parameter_output_backward)
    sizehint!(model.parameter_output_backward, length(model.parameters))
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(
        model.affine_constraint_cache,
    )
        _affine_constraint_get_reverse!(
            model,
            model.affine_constraint_cache[F, S],
        )
    end
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(
        model.vector_affine_constraint_cache,
    )
        _vector_affine_constraint_get_reverse!(
            model,
            model.vector_affine_constraint_cache[F, S],
        )
    end
    for (F, S) in MOI.Utilities.DoubleDicts.nonempty_outer_keys(
        model.quadratic_constraint_cache,
    )
        _quadratic_constraint_get_reverse!(
            model,
            model.quadratic_constraint_cache[F, S],
        )
    end
    if model.affine_objective_cache !== nothing
        _affine_objective_get_reverse!(model)
    elseif model.quadratic_objective_cache !== nothing
        _quadratic_objective_get_reverse!(model)
    end
    return
end

function MOI.set(
    model::Optimizer,
    attr::DiffOpt.ReverseVariablePrimal,
    variable::MOI.VariableIndex,
    value::Number,
)
    if _is_parameter(variable)
        error("Trying to set a backward variable sensitivity for a parameter")
    end
    MOI.set(model.optimizer, attr, variable, value)
    return
end

struct ReverseParameter <: MOI.AbstractVariableAttribute end

MOI.is_set_by_optimize(::ReverseParameter) = true

function MOI.get(
    model::Optimizer,
    ::ReverseParameter,
    variable::MOI.VariableIndex,
)
    if _is_variable(variable)
        error("Trying to get a backward parameter sensitivity for a variable")
    end
    p = p_idx(variable)
    return get(model.parameter_output_backward, p, 0.0)
end

# extras to handle model_dirty

# FIXME Workaround for https://github.com/jump-dev/JuMP.jl/issues/2797
function _moi_get_result(model::MOI.ModelLike, args...)
    if MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        throw(OptimizeNotCalled())
    end
    return MOI.get(model, args...)
end

function _moi_get_result(model::MOI.Utilities.CachingOptimizer, args...)
    if MOI.Utilities.state(model) == MOI.Utilities.NO_OPTIMIZER
        throw(NoOptimizer())
    elseif MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        throw(OptimizeNotCalled())
    end
    return MOI.get(model, args...)
end

function MOI.get(
    model::JuMP.Model,
    attr::ReverseParameter,
    var_ref::JuMP.VariableRef,
)
    JuMP.check_belongs_to_model(var_ref, model)
    return _moi_get_result(JuMP.backend(model), attr, JuMP.index(var_ref))
end

# TODO: ignore ops that are 0
