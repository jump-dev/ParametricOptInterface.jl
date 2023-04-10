# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function next_variable_index!(model::Optimizer)
    return model.last_variable_index_added += 1
end

function next_parameter_index!(model::Optimizer)
    return model.last_parameter_index_added += 1
end

function update_number_of_parameters!(model::Optimizer)
    return model.number_of_parameters_in_model += 1
end

function is_parameter_in_model(model::Optimizer, v::MOI.VariableIndex)
    return PARAMETER_INDEX_THRESHOLD <
           v.value <=
           model.last_parameter_index_added
end

function is_variable_in_model(model::Optimizer, v::MOI.VariableIndex)
    return 0 < v.value <= model.last_variable_index_added
end

function has_quadratic_constraint_caches(model::Optimizer)
    return !isempty(model.moi_quadratic_to_poi_affine_map)
end

function function_has_parameters(
    model::Optimizer,
    f::MOI.ScalarAffineFunction{T},
) where {T}
    for term in f.terms
        if is_parameter_in_model(model, term.variable)
            return true
        end
    end
    return false
end

function function_has_parameters(model::Optimizer, f::MOI.VectorOfVariables)
    for variable in f.variables
        if is_parameter_in_model(model, variable)
            return true
        end
    end
    return false
end

function function_has_parameters(
    model::Optimizer,
    f::MOI.VectorAffineFunction{T},
) where {T}
    for term in f.terms
        if is_parameter_in_model(model, term.scalar_term.variable)
            return true
        end
    end
    return false
end

function function_has_parameters(
    model::Optimizer,
    f::MOI.ScalarQuadraticFunction{T},
) where {T}
    return function_affine_terms_has_parameters(model, f.affine_terms) ||
           function_quadratic_terms_has_parameters(model, f.quadratic_terms)
end

function function_affine_terms_has_parameters(
    model::Optimizer,
    affine_terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    for term in affine_terms
        if is_parameter_in_model(model, term.variable)
            return true
        end
    end
    return false
end

function function_quadratic_terms_has_parameters(
    model::Optimizer,
    quadratic_terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}
    for term in quadratic_terms
        if is_parameter_in_model(model, term.variable_1) ||
           is_parameter_in_model(model, term.variable_2)
            return true
        end
    end
    return false
end

function count_scalar_affine_terms_types(
    model::Optimizer,
    terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    num_vars = count(x -> is_variable_in_model(model, x.variable), terms)
    num_params = length(terms) - num_vars
    return num_vars, num_params
end

function count_scalar_affine_terms_types(
    model::Optimizer,
    terms::Vector{MOI.ScalarAffineTerm{T}},
    variables_associated_to_parameters::Vector{MOI.VariableIndex},
) where {T}
    num_vars = 0
    num_params = 0
    num_vars_associated_to_params = 0
    for term in terms
        if is_variable_in_model(model, term.variable)
            num_vars += 1
            if term.variable in variables_associated_to_parameters
                num_vars_associated_to_params += 1
            end
        else
            num_params += 1
        end
    end
    return num_vars, num_params, num_vars_associated_to_params
end

function is_variable(v::MOI.VariableIndex)
    return v.index < PARAMETER_INDEX_THRESHOLD
end

function count_scalar_affine_terms_types(
    terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    num_vars = 0
    num_params = 0
    for term in terms
        if is_variable(term.variable)
            num_vars += 1
        else
            num_params += 1
        end
    end
    return num_vars, num_params
end

function split_affine_terms(terms::Vector{MOI.ScalarAffineTerm{T}}) where {T}
    num_v, num_p = count_scalar_affine_terms_types(terms)
    v = Vector{MOI.ScalarAffineTerm{T}}(undef, num_v)
    p = Vector{MOI.ScalarAffineTerm{T}}(undef, num_p)
    i_v = 1
    i_p = 1
    for term in terms
        if is_variable(term.variable)
            v[i_v] = term
            i_v += 1
        else
            p[i_p] = term
            i_p += 1
        end
    end
    return v, p
end

function ParametricAffineFunction(
    f::MOI.ScalarAffineFunction{T}
) where {T}
    v, p = split_affine_terms(f.terms)
    return ParametricAffineFunction{T}(
        p,
        v,
        f.constant,
        zero(T),
    )
end

function parametric_constant(
    model::Optimizer,
    f::ParametricAffineFunction{T},
) where {T}
    param_constant = f.c
    for term in f.p
        param_constant +=
            term.coefficient * model.parameters[p_idx(term.variable)]
    end
    return param_constant
end

function delta_parametric_constant(
    model::Optimizer,
    f::ParametricAffineFunction{T},
) where {T}
    delta_constant = zero(T)
    for term in f.p
        p = p_idx(term.variable)
        if !isnan(model.updated_parameters[p])
            delta_constant += term.coefficient *
                (model.updated_parameters[p] - model.parameters[p])
        end
    end
    return delta_constant
end

function count_scalar_quadratic_terms_types(
    terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}
    num_vv = 0
    num_pp = 0
    num_pv = 0
    for term in terms
        if is_variable(term.variable_1)
            if is_variable(term.variable_2)
                num_vv += 1
            else
                num_pv += 1
            end
        else
            if is_variable(term.variable_2)
                num_pv += 1
            else
                num_pp += 1
            end
        end
    end
    return num_vv, num_pp, num_pv
end

function split_quadratic_terms(
    terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}
    num_vv, num_pp, num_pv = count_scalar_quadratic_terms_types(terms)
    pp = Vector{MOI.ScalarQuadraticTerm{T}}(undef, num_pp) # parameter x parameter
    pv = Vector{MOI.ScalarQuadraticTerm{T}}(undef, num_pv) # parameter (as a variable) x variable
    vv = Vector{MOI.ScalarQuadraticTerm{T}}(undef, num_vv) # variable x variable
    i_vv = 1
    i_pp = 1
    i_pv = 1
    for term in terms
        if is_variable(term.variable_1)
            if is_variable(term.variable_2)
                vv[i_vv] = term
                i_vv += 1
            else
                pv[i_pv] = MOI.ScalarQuadraticTerm(
                    term.coefficient,
                    term.variable_2,
                    term.variable_1,
                )
                i_pv += 1
            end
        else
            if is_variable(term.variable_2)
                pv[i_pv] = term
                i_pv += 1
            else
                pp[i_pp] = term
                i_pp += 1
            end
        end
    end
    return pv, pp, vv
end

function ParametricQuadraticFunction(
    f::MOI.ScalarQuadraticFunction{T}
) where {T}
    v, p = split_affine_terms(f.affine_terms)
    pv, pp, vv = split_affine_terms(f.quadratic_terms)

    # find variables related to parameters
    # so that we only cache the important part of the v (affine part)
    v_in_pv = Set{MOI.VariableIndex}()
    sizehint!(v_in_pv, length(pv))
    for term in pv
        push!(v_in_pv, term.variable_2)
    end
    affine_data = Dict{MOI.VariableIndex,T}()
    sizehint!(affine_data, length(v_in_pv))
    for term in v
        if term.variable in v_in_pv
            base = get(affine_data, term.variable, zero(T))
            affine_data[term.variable] = term.coefficient + base
        end
    end

    return ParametricQuadraticFunction{T}(
        affine_data,
        pv,
        pp,
        vv,
        p,
        v,
        f.constant,
        Dict{MOI.VariableIndex,T}(),
        zero(T),
    )
end

function parametric_constant(
    model::Optimizer,
    f::ParametricQuadraticFunction{T},
) where {T}
    param_constant = f.c
    for term in f.p
        param_constant +=
            term.coefficient * model.parameters[p_idx(term.variable)]
    end
    for term in f.pp
        param_constant +=
            term.coefficient * model.parameters[p_idx(term.variable_1)] *
            model.parameters[p_idx(term.variable_2)]
    end
    return param_constant
end

function delta_parametric_constant(
    model::Optimizer,
    f::ParametricQuadraticFunction{T},
) where {T}
    delta_constant = zero(T)
    for term in f.p
        p = p_idx(term.variable)
        if !isnan(model.updated_parameters[p])
            delta_constant +=
                term.coefficient * (
                    model.updated_parameters[p] - model.parameters[p]
                )
        end
    end
    for term in f.pp
        p1 = p_idx(term.variable_1)
        delta_1 = if !isnan(model.updated_parameters[p1])
            model.updated_parameters[p1] - model.parameters[p1]
        else
            0.0
        end
        p2 = p_idx(term.variable_2)
        delta_2 = if !isnan(model.updated_parameters[p2])
            model.updated_parameters[p2] - model.parameters[p2]
        else
            0.0
        end
        delta_constant += term.coefficient * delta_1 * delta_2
    end
    return delta_constant
end

function parametric_affine_terms(
    model::Optimizer,
    f::ParametricQuadraticFunction{T},
) where {T}
    param_terms_dict = Dict{MOI.VariableIndex,T}()
    sizehint!(param_terms_dict, length(f.pv))
    # remember a variable may appear more than once in pv
    for term in f.pv
        base = get(param_terms_dict, term.variable_2, zero(T))
        param_terms_dict[term.variable_2] = base + term.coefficient *
            model.parameters[p_idx(term.variable_1)]
    end
    # by definition affin data only contains variables that appear in pv
    for (var, coef) in f.affine_data
        param_terms_dict[var] += coef
    end
    return param_terms_dict
end

function delta_parametric_affine_terms(
    model::Optimizer,
    f::ParametricQuadraticFunction{T},
) where {T}
    delta_terms_dict = Dict{MOI.VariableIndex,T}()
    sizehint!(delta_terms_dict, length(f.pv))
    # remember a variable may appear more than once in pv
    for term in f.pv
        p = p_idx(term.variable_1)
        if !isnan(model.updated_parameters[p])
            base = get(delta_terms_dict, term.variable_2, zero(T))
            delta_terms_dict[term.variable_2] = base + term.coefficient *
                (model.updated_parameters[p2] - model.parameters[p])
        end
    end
    return delta_terms_dict
end

function move_set_constant_to_function(
    f::MOI.AbstractScalarFunction,
    s::Union{MOI.LessThan{T},MOI.GreaterThan{T},MOI.EqualTo{T}},
) where {T}
    c = constant(s)
    new_f = copy(f)
    new_f.constant -= c
    return new_f
end

function move_set_constant_to_function(
    f::MOI.AbstractScalarFunction,
    s::MOI.AbstractScalarSet,
)
    return f
end

function separate_possible_terms_and_calculate_parameter_constant(
    model::Optimizer,
    terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    num_vars, num_params = count_scalar_affine_terms_types(model, terms)
    vars = Vector{MOI.ScalarAffineTerm{T}}(undef, num_vars)
    params = Vector{MOI.ScalarAffineTerm{T}}(undef, num_params)
    param_constant = zero(T)
    i_vars = 1
    i_params = 1

    for term in terms
        # TODO: do we really need this checks?
        if is_variable_in_model(model, term.variable)
            vars[i_vars] = term
            i_vars += 1
        elseif is_parameter_in_model(model, term.variable)
            params[i_params] = term
            param_constant +=
                term.coefficient * model.parameters[p_idx(term.variable)]
            i_params += 1
        else
            error("Constraint uses a variable that is not in the model")
        end
    end
    return vars, params, param_constant
end

# This version is used on SQFs
function separate_possible_terms_and_calculate_parameter_constant(
    model::Optimizer,
    terms::Vector{MOI.ScalarAffineTerm{T}},
    variables_associated_to_parameters::Vector{MOI.VariableIndex},
) where {T}
    num_vars, num_params, num_vars_associated_to_params =
        count_scalar_affine_terms_types(
            model,
            terms,
            variables_associated_to_parameters,
        )
    vars = Vector{MOI.ScalarAffineTerm{T}}(undef, num_vars)
    params = Vector{MOI.ScalarAffineTerm{T}}(undef, num_params)
    terms_with_variables_associated_to_parameters =
        Vector{MOI.ScalarAffineTerm{T}}(undef, num_vars_associated_to_params)
    param_constant = zero(T)
    i_vars = 1
    i_params = 1
    i_vars_associated_to_params = 1

    for term in terms
        if is_variable_in_model(model, term.variable)
            vars[i_vars] = term
            if term.variable in variables_associated_to_parameters
                terms_with_variables_associated_to_parameters[i_vars_associated_to_params] =
                    term
                i_vars_associated_to_params += 1
            end
            i_vars += 1
        elseif is_parameter_in_model(model, term.variable)
            params[i_params] = term
            param_constant +=
                term.coefficient * model.parameters[p_idx(term.variable)]
            i_params += 1
        else
            error("Constraint uses a variable that is not in the model")
        end
    end
    return vars,
    params,
    terms_with_variables_associated_to_parameters,
    param_constant
end

function count_scalar_quadratic_terms_types(
    model::Optimizer,
    terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}
    num_vv = 0
    num_pp = 0
    num_pv = 0
    for term in terms
        if is_variable_in_model(model, term.variable_1) &&
           is_variable_in_model(model, term.variable_2)
            num_vv += 1
        elseif is_variable_in_model(model, term.variable_1) &&
               is_parameter_in_model(model, term.variable_2)
            num_pv += 1
        elseif is_parameter_in_model(model, term.variable_1) &&
               is_variable_in_model(model, term.variable_2)
            num_pv += 1
        else
            num_pp += 1
        end
    end
    return num_vv, num_pp, num_pv
end

function separate_possible_terms_and_calculate_parameter_constant(
    model::Optimizer,
    terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}
    num_vv, num_pp, num_pv =
        count_scalar_quadratic_terms_types(model, terms)

    pp = Vector{MOI.ScalarQuadraticTerm{T}}(undef, num_pp) # parameter x parameter
    pv = Vector{MOI.ScalarQuadraticTerm{T}}(undef, num_pv) # parameter (as a variable) x variable
    vv = Vector{MOI.ScalarQuadraticTerm{T}}(undef, num_vv) # variable x variable
    aff_terms = Vector{MOI.ScalarAffineTerm{T}}(undef, num_pv) # parameter (as a number) x variable
    variables_associated_to_parameters =
        Vector{MOI.VariableIndex}(undef, num_pv)
    quad_param_constant = zero(T)

    i_vv = 1
    i_pp = 1
    i_pv = 1

    # When we have a parameter x variable or a variable x parameter the convention is to rewrite
    # the SQT with parameter as variable_index_1 and variable as variable_index_2
    for term in terms
        if (
            is_variable_in_model(model, term.variable_1) &&
            is_variable_in_model(model, term.variable_2)
        )
            vv[i_vv] = term # if there are only variables, it remains a quadratic term
            i_vv += 1
        elseif (
            is_parameter_in_model(model, term.variable_1) &&
            is_variable_in_model(model, term.variable_2)
        )
            pv[i_pv] = term
            variables_associated_to_parameters[i_pv] =
                term.variable_2
            aff_terms[i_pv] = MOI.ScalarAffineTerm(
                term.coefficient * model.parameters[p_idx(term.variable_1)],
                term.variable_2,
            )
            if model.evaluate_duals
                push!(model.multiplicative_parameters, term.variable_1.value)
            end
            i_pv += 1
        elseif (
            is_variable_in_model(model, term.variable_1) &&
            is_parameter_in_model(model, term.variable_2)
        )
            # Check convention defined above. We use the convention to know decide who is a variable and who is
            # a parameter withou having to recheck which is which.
            pv[i_pv] = MOI.ScalarQuadraticTerm(
                term.coefficient,
                term.variable_2,
                term.variable_1,
            )
            variables_associated_to_parameters[i_pv] =
                term.variable_1
            aff_terms[i_pv] = MOI.ScalarAffineTerm(
                term.coefficient * model.parameters[p_idx(term.variable_2)],
                term.variable_1,
            )
            model.evaluate_duals &&
                push!(model.multiplicative_parameters, term.variable_2.value)
            i_pv += 1
        elseif (
            is_parameter_in_model(model, term.variable_1) &&
            is_parameter_in_model(model, term.variable_2)
        )
            pp[i_pp] = term
            model.evaluate_duals &&
                push!(model.multiplicative_parameters, term.variable_1.value)
            model.evaluate_duals &&
                push!(model.multiplicative_parameters, term.variable_2.value)
            quad_param_constant +=
                term.coefficient *
                model.parameters[p_idx(term.variable_1)] *
                model.parameters[p_idx(term.variable_2)]
            i_pp += 1
        else
            throw(
                ErrorException(
                    "Constraint uses a variable or parameter that is not in the model",
                ),
            )
        end
    end
    return vv,
    pv,
    pp,
    aff_terms,
    variables_associated_to_parameters,
    quad_param_constant
end

function fill_quadratic_constraint_caches!(
    model::Optimizer,
    new_ci::MOI.ConstraintIndex,
    pv::Vector{MOI.ScalarQuadraticTerm{T}},
    pp::Vector{MOI.ScalarQuadraticTerm{T}},
    aff_params::Vector{MOI.ScalarAffineTerm{T}},
    terms_with_variables_associated_to_parameters::Vector{
        MOI.ScalarAffineTerm{T},
    },
    ci::MOI.ConstraintIndex,
) where {T,S}
    if !isempty(pv)
        model.quadratic_constraint_cache_pv[new_ci] = pv
    end
    if !isempty(pp)
        model.quadratic_constraint_cache_pp[new_ci] = pp
        model.quadratic_constraint_cache_pp_set[new_ci] =
            MOI.get(model.optimizer, MOI.ConstraintSet(), ci)
    end
    if !isempty(aff_params)
        model.quadratic_constraint_cache_pc[new_ci] = aff_params
        model.quadratic_constraint_cache_pc_set[new_ci] =
            MOI.get(model.optimizer, MOI.ConstraintSet(), ci)
    end
    if !isempty(terms_with_variables_associated_to_parameters)
        model.variables_multiplied_by_parameters[new_ci] =
            terms_with_variables_associated_to_parameters
    end
    return nothing
end

function quadratic_constraint_cache_map_check(
    model::Optimizer,
    idx::MOI.ConstraintIndex{F,S},
) where {F,S}
    cached_constraints = values(model.moi_quadratic_to_poi_affine_map)
    # Using this becuase some custom brodcast method throws errors if
    # inner_idex .∈ cached_constraints is used
    return idx ∈ cached_constraints
end

# Vector Affine
function separate_possible_terms_and_calculate_parameter_constant(
    model::Optimizer,
    f::MOI.VectorAffineFunction{T},
    set::S,
) where {T,S<:MOI.AbstractVectorSet}
    vars = MOI.VectorAffineTerm{T}[]
    params = MOI.VectorAffineTerm{T}[]
    n_dims = length(f.constants)
    param_constants = zeros(T, n_dims)
    for term in f.terms
        oi = term.output_index

        if is_variable_in_model(model, term.scalar_term.variable)
            push!(vars, term)
        elseif is_parameter_in_model(model, term.scalar_term.variable)
            push!(params, term)
            param_constants[oi] +=
                term.scalar_term.coefficient *
                model.parameters[p_idx(term.scalar_term.variable)]
        else
            error("Constraint uses a variable that is not in the model")
        end
    end
    return vars, params, param_constants
end
