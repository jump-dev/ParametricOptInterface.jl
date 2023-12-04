using DiffOpt

# forward mode

function DiffOpt.forward_differentiate!(model::Optimizer{T}) where {T}
    # TODO: add a reset option
    for (F, S) in keys(model.affine_constraint_cache.dict)
        affine_constraint_cache_inner = model.affine_constraint_cache[F, S]
        if !isempty(affine_constraint_cache_inner)
            # TODO add: barrier to avoid type instability of inner dicts
            for (inner_ci, pf) in affine_constraint_cache_inner
                cte = zero(T)
                terms = MOI.ScalarAffineTerm{T}[]
                sizehint!(terms, 0)
                if length(pf.p) != 0
                    for term in pf.p
                        p = p_idx(term.variable)
                        sensitivity = get(model.parameter_input_forward, p, 0.0)
                        # TODO: check sign
                        cte += sensitivity * term.coefficient
                    end
                    # TODO: if cte != 0
                    MOI.set(
                        model.optimizer,
                        DiffOpt.ForwardConstraintFunction(),
                        inner_ci,
                        MOI.ScalarAffineFunction{T}(terms, cte),
                    )
                end
            end
        end
    end
    for (F, S) in keys(model.vector_affine_constraint_cache.dict)
        vector_affine_constraint_cache_inner =
            model.vector_affine_constraint_cache[F, S]
        if !isempty(vector_affine_constraint_cache_inner)
            # barrier to avoid type instability of inner dicts
            for (inner_ci, pf) in vector_affine_constraint_cache_inner
                cte = zeros(T, length(pf.c))
                terms = MOI.VectorAffineTerm{T}[]
                sizehint!(terms, 0)
                if length(pf.p) != 0
                    for term in pf.p
                        p = p_idx(term.scalar_term.variable)
                        sensitivity = get(model.parameter_input_forward, p, 0.0)
                        # TODO: check sign
                        cte[term.output_index] += sensitivity * term.scalar_term.coefficient
                    end
                    # TODO: if cte != 0
                    MOI.set(
                        model.optimizer,
                        DiffOpt.ForwardConstraintFunction(),
                        inner_ci,
                        MOI.ScalarAffineFunction{T}(terms, cte),
                    )
                end
            end
        end
    end
    for (F, S) in keys(model.quadratic_constraint_cache.dict)
        quadratic_constraint_cache_inner =
            model.quadratic_constraint_cache[F, S]
        if !isempty(quadratic_constraint_cache_inner)
            # TODO add: barrier to avoid type instability of inner dicts
            for (inner_ci, pf) in quadratic_constraint_cache_inner
                cte = zero(T)
                terms = MOI.ScalarAffineTerm{T}[]
                # terms_dict = Dict{MOI.VariableIndex,T}() # canonicalize?
                sizehint!(terms, length(pf.pv))
                if length(pf.p) != 0 || length(pf.pv) != 0 || length(pf.pp) != 0
                    for term in pf.p
                        p = p_idx(term.variable)
                        sensitivity = get(model.parameter_input_forward, p, 0.0)
                        # TODO: check sign
                        cte += sensitivity * term.coefficient
                    end
                    for term in pf.pp
                        p_1 = p_idx(term.variable_1)
                        p_2 = p_idx(term.variable_2)
                        sensitivity_1 = get(model.parameter_input_forward, p_1, 0.0)
                        sensitivity_2 = get(model.parameter_input_forward, p_2, 0.0)
                        cte += sensitivity_1 * sensitivity_2 * term.coefficient
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
                    MOI.set(
                        model.optimizer,
                        DiffOpt.ForwardConstraintFunction(),
                        inner_ci,
                        MOI.ScalarAffineFunction{T}(terms, cte),
                    )
                end
            end
        end
    end
    if model.affine_objective_cache !== nothing
        cte = zero(T)
        terms = MOI.ScalarAffineTerm{T}[]
        pf = model.affine_objective_cache
        sizehint!(terms, 0)
        if length(pf.p) != 0
            for term in pf.p
                p = p_idx(term.variable)
                sensitivity = get(model.parameter_input_forward, p, 0.0)
                # TODO: check sign
                cte += sensitivity * term.coefficient
            end
            # TODO: if cte != 0
            MOI.set(
                model.optimizer,
                DiffOpt.ForwardObjectiveFunction(),
                inner_ci,
                MOI.ScalarAffineFunction{T}(terms, cte),
            )
        end
    elseif model.quadratic_objective_cache !== nothing
        cte = zero(T)
        terms = MOI.ScalarAffineTerm{T}[]
        pf = model.quadratic_objective_cache
        sizehint!(terms, length(pf.pv))
        if length(pf.p) != 0
            for term in pf.p
                p = p_idx(term.variable)
                sensitivity = get(model.parameter_input_forward, p, 0.0)
                # TODO: check sign
                cte += sensitivity * term.coefficient
            end
            for term in pf.pp
                p_1 = p_idx(term.variable_1)
                p_2 = p_idx(term.variable_2)
                sensitivity_1 = get(model.parameter_input_forward, p_1, 0.0)
                sensitivity_2 = get(model.parameter_input_forward, p_2, 0.0)
                cte += sensitivity_1 * sensitivity_2 * term.coefficient
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
            # TODO: if cte != 0
            MOI.set(
                model.optimizer,
                DiffOpt.ForwardObjectiveFunction(),
                inner_ci,
                MOI.ScalarAffineFunction{T}(terms, cte),
            )
        end
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
        error("Trying to set a parameter sensitivity for a variable")
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
        error("Trying to get a variable sensitivity for a parameter")
    end
    return MOI.get(model.optimizer, attr, model.variables[variable])
end

# reverse mode

function DiffOpt.reverse_differentiate!(model::Optimizer)
    error("Not implemented")
    DiffOpt.reverse_differentiate!(model.optimizer)
    return
end

function MOI.set(
    model::Optimizer,
    attr::DiffOpt.ReverseVariablePrimal,
    variable::MOI.VariableIndex,
    value::Number,
)
    MOI.set(model.optimizer, attr, variable, value)
    return
end

struct ReverseParameter <: MOI.AbstractVariableAttribute end

function MOI.get(
    model::Optimizer,
    attr::ReverseParameter,
    variable::MOI.VariableIndex,
)
    error("Not implemented")
    return
end
