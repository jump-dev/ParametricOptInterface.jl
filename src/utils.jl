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
    return !isempty(model.quadratic_outer_to_inner)
end

function function_has_parameters(f::MOI.ScalarAffineFunction{T}) where {T}
    for term in f.terms
        if is_parameter(term.variable)
            return true
        end
    end
    return false
end

function function_has_parameters(f::MOI.VectorOfVariables)
    for variable in f.variables
        if is_parameter(variable)
            return true
        end
    end
    return false
end

function function_has_parameters(f::MOI.VectorAffineFunction{T}) where {T}
    for term in f.terms
        if is_parameter(term.scalar_term.variable)
            return true
        end
    end
    return false
end

function function_has_parameters(f::MOI.ScalarQuadraticFunction{T}) where {T}
    return function_affine_terms_has_parameters(f.affine_terms) ||
           function_quadratic_terms_has_parameters(f.quadratic_terms)
end

function function_affine_terms_has_parameters(
    affine_terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    for term in affine_terms
        if is_parameter(term.variable)
            return true
        end
    end
    return false
end

function function_quadratic_terms_has_parameters(
    quadratic_terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}
    for term in quadratic_terms
        if is_parameter(term.variable_1) || is_parameter(term.variable_2)
            return true
        end
    end
    return false
end

function is_variable(v::MOI.VariableIndex)
    return v.value < PARAMETER_INDEX_THRESHOLD
end

function is_parameter(v::MOI.VariableIndex)
    return v.value > PARAMETER_INDEX_THRESHOLD
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

function ParametricAffineFunction(f::MOI.ScalarAffineFunction{T}) where {T}
    v, p = split_affine_terms(f.terms)
    return ParametricAffineFunction{T}(p, v, f.constant, zero(T), zero(T))
end

function original_function(f::ParametricAffineFunction{T}) where {T}
    return MOI.ScalarAffineFunction{T}(vcat(f.p, f.v), f.c)
end

function current_function(f::ParametricAffineFunction{T}) where {T}
    return MOI.ScalarAffineFunction{T}(f.v, f.current_constant)
end

function update_cache!(
    f::ParametricAffineFunction{T},
    model::Optimizer,
) where {T}
    f.current_constant = parametric_constant(model, f)
    return nothing
end

function parametric_constant(
    model::Optimizer,
    f::ParametricAffineFunction{T},
) where {T}
    # do not add set_function here
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
            delta_constant +=
                term.coefficient *
                (model.updated_parameters[p] - model.parameters[p])
        end
    end
    return delta_constant
end

function count_vector_affine_terms_types(
    terms::Vector{MOI.VectorAffineTerm{T}},
) where {T}
    num_vars = 0
    num_params = 0
    for term in terms
        if is_variable(term.scalar_term.variable)
            num_vars += 1
        else
            num_params += 1
        end
    end
    return num_vars, num_params
end

function split_affine_terms(terms::Vector{MOI.VectorAffineTerm{T}}) where {T}
    num_v, num_p = count_vector_affine_terms_types(terms)
    v = Vector{MOI.VectorAffineTerm{T}}(undef, num_v)
    p = Vector{MOI.VectorAffineTerm{T}}(undef, num_p)
    i_v = 1
    i_p = 1
    for term in terms
        if is_variable(term.scalar_term.variable)
            v[i_v] = term
            i_v += 1
        else
            p[i_p] = term
            i_p += 1
        end
    end
    return v, p
end

function ParametricVectorAffineFunction(
    f::MOI.VectorAffineFunction{T},
) where {T}
    v, p = split_affine_terms(f.terms)
    return ParametricVectorAffineFunction{T}(
        p,
        v,
        copy(f.constants),
        zeros(T, length(f.constants)),
        zeros(T, length(f.constants)),
    )
end

function original_function(f::ParametricVectorAffineFunction{T}) where {T}
    return MOI.VectorAffineFunction{T}(vcat(f.p, f.v), f.c)
end

function current_function(f::ParametricVectorAffineFunction{T}) where {T}
    return MOI.VectorAffineFunction{T}(f.v, f.current_constant)
end

function update_cache!(
    f::ParametricVectorAffineFunction{T},
    model::Optimizer,
) where {T}
    f.current_constant = parametric_constant(model, f)
    return nothing
end

function parametric_constant(
    model::Optimizer,
    f::ParametricVectorAffineFunction{T},
) where {T}
    # do not add set_function here
    param_constant = copy(f.c)
    for term in f.p
        param_constant[term.output_index] +=
            term.scalar_term.coefficient *
            model.parameters[p_idx(term.scalar_term.variable)]
    end
    return param_constant
end

function delta_parametric_constant(
    model::Optimizer,
    f::ParametricVectorAffineFunction{T},
) where {T}
    delta_constant = zeros(T, length(f.c))
    for term in f.p
        p = p_idx(term.scalar_term.variable)
        if !isnan(model.updated_parameters[p])
            delta_constant[term.output_index] +=
                term.scalar_term.coefficient *
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
    f::MOI.ScalarQuadraticFunction{T},
) where {T}
    v, p = split_affine_terms(f.affine_terms)
    pv, pp, vv = split_quadratic_terms(f.quadratic_terms)

    # find variables related to parameters
    # so that we only cache the important part of the v (affine part)
    v_in_pv = Set{MOI.VariableIndex}()
    sizehint!(v_in_pv, length(pv))
    for term in pv
        push!(v_in_pv, term.variable_2)
    end
    affine_data = Dict{MOI.VariableIndex,T}()
    sizehint!(affine_data, length(v_in_pv))
    affine_data_np = Dict{MOI.VariableIndex,T}()
    sizehint!(affine_data, length(v))
    for term in v
        if term.variable in v_in_pv
            base = get(affine_data, term.variable, zero(T))
            affine_data[term.variable] = term.coefficient + base
        else
            base = get(affine_data_np, term.variable, zero(T))
            affine_data_np[term.variable] = term.coefficient + base
        end
    end

    return ParametricQuadraticFunction{T}(
        affine_data,
        affine_data_np,
        pv,
        pp,
        vv,
        p,
        v,
        f.constant,
        zero(T),
        Dict{MOI.VariableIndex,T}(),
        zero(T),
    )
end

function original_function(f::ParametricQuadraticFunction{T}) where {T}
    return MOI.ScalarQuadraticFunction{T}(
        vcat(f.pv, f.pp, f.vv),
        vcat(f.p, f.v),
        f.c,
    )
end

function current_function(f::ParametricQuadraticFunction{T}) where {T}
    affine = MOI.ScalarAffineTerm{T}[]
    sizehint!(affine, length(f.current_terms_with_p) + length(f.affine_data_np))
    for (v, c) in f.current_terms_with_p
        push!(affine, MOI.ScalarAffineTerm{T}(c, v))
    end
    for (v, c) in f.affine_data_np
        push!(affine, MOI.ScalarAffineTerm{T}(c, v))
    end
    return MOI.ScalarQuadraticFunction{T}(f.vv, affine, f.current_constant)
end

function update_cache!(
    f::ParametricQuadraticFunction{T},
    model::Optimizer,
) where {T}
    f.current_constant = parametric_constant(model, f)
    f.current_terms_with_p = parametric_affine_terms(model, f)
    return nothing
end

function parametric_constant(
    model::Optimizer,
    f::ParametricQuadraticFunction{T},
) where {T}
    # do not add set_function here
    param_constant = f.c
    for term in f.p
        param_constant +=
            term.coefficient * model.parameters[p_idx(term.variable)]
    end
    for term in f.pp
        param_constant +=
            term.coefficient *
            model.parameters[p_idx(term.variable_1)] *
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
                term.coefficient *
                (model.updated_parameters[p] - model.parameters[p])
        end
    end
    for term in f.pp
        p1 = p_idx(term.variable_1)
        p2 = p_idx(term.variable_2)
        isnan_1 = isnan(model.updated_parameters[p1])
        isnan_2 = isnan(model.updated_parameters[p2])
        if !isnan_1 || !isnan_2
            new_1 = ifelse(
                isnan_1,
                model.parameters[p1],
                model.updated_parameters[p1],
            )
            new_2 = ifelse(
                isnan_2,
                model.parameters[p2],
                model.updated_parameters[p2],
            )
            delta_constant +=
                term.coefficient *
                (new_1 * new_2 - model.parameters[p1] * model.parameters[p2])
        end
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
        param_terms_dict[term.variable_2] =
            base + term.coefficient * model.parameters[p_idx(term.variable_1)]
    end
    # by definition affine data only contains variables that appear in pv
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
            delta_terms_dict[term.variable_2] =
                base +
                term.coefficient *
                (model.updated_parameters[p] - model.parameters[p])
        end
    end
    return delta_terms_dict
end

function cache_set_constant!(
    f::ParametricAffineFunction{T},
    s::Union{MOI.LessThan{T},MOI.GreaterThan{T},MOI.EqualTo{T}},
) where {T}
    f.set_constant = MOI.constant(s)
    return
end

function cache_set_constant!(
    f::ParametricAffineFunction{T},
    s::MOI.AbstractScalarSet,
) where {T}
    return
end

function cache_set_constant!(
    f::ParametricQuadraticFunction{T},
    s::Union{MOI.LessThan{T},MOI.GreaterThan{T},MOI.EqualTo{T}},
) where {T}
    f.set_constant = MOI.constant(s)
    return
end

function cache_set_constant!(
    f::ParametricQuadraticFunction{T},
    s::MOI.AbstractScalarSet,
) where {T}
    return
end

function is_affine(f::MOI.ScalarQuadraticFunction)
    if isempty(f.quadratic_terms)
        return true
    end
    return false
end

function cache_multiplicative_params!(
    model::Optimizer{T},
    f::ParametricQuadraticFunction{T},
) where {T}
    for term in f.pv
        push!(model.multiplicative_parameters, term.variable_1.value)
    end
    # TODO compute these duals might be feasible
    for term in f.pp
        push!(model.multiplicative_parameters, term.variable_1.value)
        push!(model.multiplicative_parameters, term.variable_2.value)
    end
    return
end

# TODO: review comment
function quadratic_constraint_cache_map_check(
    model::Optimizer,
    idx::MOI.ConstraintIndex{F,S},
) where {F,S}
    cached_constraints = values(model.quadratic_outer_to_inner)
    # Using this because some custom brodcast method throws errors if
    # inner_idex .∈ cached_constraints is used
    return idx ∈ cached_constraints
end
