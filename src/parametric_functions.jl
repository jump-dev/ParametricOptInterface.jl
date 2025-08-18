abstract type ParametricFunction{T} end

function _cache_set_constant!(
    f::ParametricFunction{T},
    s::Union{MOI.LessThan{T},MOI.GreaterThan{T},MOI.EqualTo{T}},
) where {T}
    f.set_constant = MOI.constant(s)
    return
end

function _cache_set_constant!(
    ::ParametricFunction{T},
    ::MOI.AbstractScalarSet,
) where {T}
    return
end

mutable struct ParametricQuadraticFunction{T} <: ParametricFunction{T}
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

function ParametricQuadraticFunction(
    f::MOI.ScalarQuadraticFunction{T},
) where {T}
    v, p = _split_affine_terms(f.affine_terms)
    pv, pp, vv = _split_quadratic_terms(f.quadratic_terms)

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

function affine_parameter_terms(f::ParametricQuadraticFunction)
    return f.p
end

function affine_variable_terms(f::ParametricQuadraticFunction)
    return f.v
end

function quadratic_parameter_variable_terms(f::ParametricQuadraticFunction)
    return f.pv
end

function quadratic_parameter_parameter_terms(f::ParametricQuadraticFunction)
    return f.pp
end

function quadratic_variable_variable_terms(f::ParametricQuadraticFunction)
    return f.vv
end

function _split_quadratic_terms(
    terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}
    num_vv, num_pp, num_pv = _count_scalar_quadratic_terms_types(terms)
    pp = Vector{MOI.ScalarQuadraticTerm{T}}(undef, num_pp) # parameter x parameter
    pv = Vector{MOI.ScalarQuadraticTerm{T}}(undef, num_pv) # parameter (as a variable) x variable
    vv = Vector{MOI.ScalarQuadraticTerm{T}}(undef, num_vv) # variable x variable
    i_vv = 1
    i_pp = 1
    i_pv = 1
    for term in terms
        if _is_variable(term.variable_1)
            if _is_variable(term.variable_2)
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
            if _is_variable(term.variable_2)
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

function _count_scalar_quadratic_terms_types(
    terms::Vector{MOI.ScalarQuadraticTerm{T}},
) where {T}
    num_vv = 0
    num_pp = 0
    num_pv = 0
    for term in terms
        if _is_variable(term.variable_1)
            if _is_variable(term.variable_2)
                num_vv += 1
            else
                num_pv += 1
            end
        else
            if _is_variable(term.variable_2)
                num_pv += 1
            else
                num_pp += 1
            end
        end
    end
    return num_vv, num_pp, num_pv
end

function _original_function(f::ParametricQuadraticFunction{T}) where {T}
    return MOI.ScalarQuadraticFunction{T}(
        vcat(
            quadratic_parameter_variable_terms(f),
            quadratic_parameter_parameter_terms(f),
            quadratic_variable_variable_terms(f),
        ),
        vcat(affine_parameter_terms(f), affine_variable_terms(f)),
        f.c,
    )
end

function _current_function(f::ParametricQuadraticFunction{T}) where {T}
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

function _parametric_constant(
    model,
    f::ParametricQuadraticFunction{T},
) where {T}
    # do not add set_function here
    param_constant = f.c
    for term in affine_parameter_terms(f)
        param_constant +=
            term.coefficient * model.parameters[p_idx(term.variable)]
    end
    for term in quadratic_parameter_parameter_terms(f)
        param_constant +=
            (
                term.coefficient /
                ifelse(term.variable_1 == term.variable_2, 2, 1)
            ) *
            model.parameters[p_idx(term.variable_1)] *
            model.parameters[p_idx(term.variable_2)]
    end
    return param_constant
end

function _delta_parametric_constant(
    model,
    f::ParametricQuadraticFunction{T},
) where {T}
    delta_constant = zero(T)
    for term in affine_parameter_terms(f)
        p = p_idx(term.variable)
        if !isnan(model.updated_parameters[p])
            delta_constant +=
                term.coefficient *
                (model.updated_parameters[p] - model.parameters[p])
        end
    end
    for term in quadratic_parameter_parameter_terms(f)
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
                (
                    term.coefficient /
                    ifelse(term.variable_1 == term.variable_2, 2, 1)
                ) *
                (new_1 * new_2 - model.parameters[p1] * model.parameters[p2])
        end
    end
    return delta_constant
end

function _parametric_affine_terms(
    model,
    f::ParametricQuadraticFunction{T},
) where {T}
    param_terms_dict = Dict{MOI.VariableIndex,T}()
    sizehint!(param_terms_dict, length(quadratic_parameter_variable_terms(f)))
    # remember a variable may appear more than once in pv
    for term in quadratic_parameter_variable_terms(f)
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

function _delta_parametric_affine_terms(
    model,
    f::ParametricQuadraticFunction{T},
) where {T}
    delta_terms_dict = Dict{MOI.VariableIndex,T}()
    sizehint!(delta_terms_dict, length(quadratic_parameter_variable_terms(f)))
    # remember a variable may appear more than once in pv
    for term in quadratic_parameter_variable_terms(f)
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

function _update_cache!(f::ParametricQuadraticFunction{T}, model) where {T}
    f.current_constant = _parametric_constant(model, f)
    f.current_terms_with_p = _parametric_affine_terms(model, f)
    return nothing
end

mutable struct ParametricAffineFunction{T} <: ParametricFunction{T}
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

function ParametricAffineFunction(f::MOI.ScalarAffineFunction{T}) where {T}
    v, p = _split_affine_terms(f.terms)
    return ParametricAffineFunction(p, v, f.constant)
end

function ParametricAffineFunction(
    terms_p::Vector{MOI.ScalarAffineTerm{T}},
    terms_v::Vector{MOI.ScalarAffineTerm{T}},
    constant::T,
) where {T}
    return ParametricAffineFunction{T}(
        terms_p,
        terms_v,
        constant,
        zero(T),
        zero(T),
    )
end

function affine_parameter_terms(f::ParametricAffineFunction)
    return f.p
end

function affine_variable_terms(f::ParametricAffineFunction)
    return f.v
end

function _split_affine_terms(terms::Vector{MOI.ScalarAffineTerm{T}}) where {T}
    num_v, num_p = _count_scalar_affine_terms_types(terms)
    v = Vector{MOI.ScalarAffineTerm{T}}(undef, num_v)
    p = Vector{MOI.ScalarAffineTerm{T}}(undef, num_p)
    i_v = 1
    i_p = 1
    for term in terms
        if _is_variable(term.variable)
            v[i_v] = term
            i_v += 1
        else
            p[i_p] = term
            i_p += 1
        end
    end
    return v, p
end

function _count_scalar_affine_terms_types(
    terms::Vector{MOI.ScalarAffineTerm{T}},
) where {T}
    num_vars = 0
    num_params = 0
    for term in terms
        if _is_variable(term.variable)
            num_vars += 1
        else
            num_params += 1
        end
    end
    return num_vars, num_params
end

function _original_function(f::ParametricAffineFunction{T}) where {T}
    return MOI.ScalarAffineFunction{T}(
        vcat(affine_parameter_terms(f), affine_variable_terms(f)),
        f.c,
    )
end

function _current_function(f::ParametricAffineFunction{T}) where {T}
    return MOI.ScalarAffineFunction{T}(
        affine_variable_terms(f),
        f.current_constant,
    )
end

function _parametric_constant(model, f::ParametricAffineFunction{T}) where {T}
    # do not add set_function here
    param_constant = f.c
    for term in affine_parameter_terms(f)
        param_constant +=
            term.coefficient * model.parameters[p_idx(term.variable)]
    end
    return param_constant
end

function _delta_parametric_constant(
    model,
    f::ParametricAffineFunction{T},
) where {T}
    delta_constant = zero(T)
    for term in affine_parameter_terms(f)
        p = p_idx(term.variable)
        if !isnan(model.updated_parameters[p])
            delta_constant +=
                term.coefficient *
                (model.updated_parameters[p] - model.parameters[p])
        end
    end
    return delta_constant
end

function _update_cache!(f::ParametricAffineFunction{T}, model) where {T}
    f.current_constant = _parametric_constant(model, f)
    return nothing
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

function ParametricVectorAffineFunction(
    f::MOI.VectorAffineFunction{T},
) where {T}
    v, p = _split_vector_affine_terms(f.terms)
    return ParametricVectorAffineFunction{T}(
        p,
        v,
        copy(f.constants),
        zeros(T, length(f.constants)),
        zeros(T, length(f.constants)),
    )
end

function vector_affine_parameter_terms(f::ParametricVectorAffineFunction)
    return f.p
end

function vector_affine_variable_terms(f::ParametricVectorAffineFunction)
    return f.v
end

function _split_vector_affine_terms(
    terms::Vector{MOI.VectorAffineTerm{T}},
) where {T}
    num_v, num_p = _count_vector_affine_terms_types(terms)
    v = Vector{MOI.VectorAffineTerm{T}}(undef, num_v)
    p = Vector{MOI.VectorAffineTerm{T}}(undef, num_p)
    i_v = 1
    i_p = 1
    for term in terms
        if _is_variable(term.scalar_term.variable)
            v[i_v] = term
            i_v += 1
        else
            p[i_p] = term
            i_p += 1
        end
    end
    return v, p
end

function _count_vector_affine_terms_types(
    terms::Vector{MOI.VectorAffineTerm{T}},
) where {T}
    num_vars = 0
    num_params = 0
    for term in terms
        if _is_variable(term.scalar_term.variable)
            num_vars += 1
        else
            num_params += 1
        end
    end
    return num_vars, num_params
end

function _original_function(f::ParametricVectorAffineFunction{T}) where {T}
    return MOI.VectorAffineFunction{T}(
        vcat(vector_affine_parameter_terms(f), vector_affine_variable_terms(f)),
        f.c,
    )
end

function _current_function(f::ParametricVectorAffineFunction{T}) where {T}
    return MOI.VectorAffineFunction{T}(
        vector_affine_variable_terms(f),
        f.current_constant,
    )
end

function _parametric_constant(
    model,
    f::ParametricVectorAffineFunction{T},
) where {T}
    # do not add set_function here
    param_constant = copy(f.c)
    for term in vector_affine_parameter_terms(f)
        param_constant[term.output_index] +=
            term.scalar_term.coefficient *
            model.parameters[p_idx(term.scalar_term.variable)]
    end
    return param_constant
end

function _delta_parametric_constant(
    model,
    f::ParametricVectorAffineFunction{T},
) where {T}
    delta_constant = zeros(T, length(f.c))
    for term in vector_affine_parameter_terms(f)
        p = p_idx(term.scalar_term.variable)
        if !isnan(model.updated_parameters[p])
            delta_constant[term.output_index] +=
                term.scalar_term.coefficient *
                (model.updated_parameters[p] - model.parameters[p])
        end
    end
    return delta_constant
end

function _update_cache!(f::ParametricVectorAffineFunction{T}, model) where {T}
    f.current_constant = _parametric_constant(model, f)
    return nothing
end

mutable struct ParametricVectorQuadraticFunction{T}
    # helper to efficiently update affine terms 
    affine_data::Dict{Tuple{MOI.VariableIndex,Int},T}
    affine_data_np::Dict{Tuple{MOI.VariableIndex,Int},T}
    # constant * parameter * variable (in this order)
    pv::Vector{MOI.VectorQuadraticTerm{T}}
    # constant * parameter * parameter
    pp::Vector{MOI.VectorQuadraticTerm{T}}
    # constant * variable * variable
    vv::Vector{MOI.VectorQuadraticTerm{T}}
    # constant * parameter
    p::Vector{MOI.VectorAffineTerm{T}}
    # constant * variable
    v::Vector{MOI.VectorAffineTerm{T}}
    # constant
    c::Vector{T}
    # to avoid unnecessary lookups in updates
    set_constant::Vector{T}
    # cache data that is inside the solver to avoid slow getters
    current_terms_with_p::Dict{Tuple{MOI.VariableIndex,Int},T}
    current_constant::Vector{T}
end

function ParametricVectorQuadraticFunction(
    f::MOI.VectorQuadraticFunction{T},
) where {T}
    v, p = _split_vector_affine_terms(f.affine_terms)
    pv, pp, vv = _split_vector_quadratic_terms(f.quadratic_terms)

    # Find variables related to parameters in parameter-variable quadratic terms
    v_in_pv = Set{MOI.VariableIndex}()
    sizehint!(v_in_pv, length(pv))
    for term in pv
        push!(v_in_pv, term.scalar_term.variable_2)
    end
    affine_data = Dict{Tuple{MOI.VariableIndex,Int},T}()
    sizehint!(affine_data, length(v_in_pv))
    affine_data_np = Dict{Tuple{MOI.VariableIndex,Int},T}()
    sizehint!(affine_data_np, length(v))
    for term in v
        if term.scalar_term.variable in v_in_pv
            base = get(
                affine_data,
                (term.scalar_term.variable, term.output_index),
                zero(T),
            )
            affine_data[(term.scalar_term.variable, term.output_index)] =
                term.scalar_term.coefficient + base
        else
            base = get(
                affine_data_np,
                (term.scalar_term.variable, term.output_index),
                zero(T),
            )
            affine_data_np[(term.scalar_term.variable, term.output_index)] =
                term.scalar_term.coefficient + base
        end
    end

    return ParametricVectorQuadraticFunction{T}(
        affine_data,
        affine_data_np,
        pv,
        pp,
        vv,
        p,
        v,
        copy(f.constants),
        zeros(T, length(f.constants)),
        Dict{Tuple{MOI.VariableIndex,Int},T}(),
        zeros(T, length(f.constants)),
    )
end

function vector_quadratic_parameter_variable_terms(
    f::ParametricVectorQuadraticFunction,
)
    return f.pv
end

function vector_quadratic_parameter_parameter_terms(
    f::ParametricVectorQuadraticFunction,
)
    return f.pp
end

function vector_quadratic_variable_variable_terms(
    f::ParametricVectorQuadraticFunction,
)
    return f.vv
end

function vector_affine_parameter_terms(f::ParametricVectorQuadraticFunction)
    return f.p
end

function vector_affine_variable_terms(f::ParametricVectorQuadraticFunction)
    return f.v
end

function _split_vector_quadratic_terms(
    terms::Vector{MOI.VectorQuadraticTerm{T}},
) where {T}
    num_vv = 0
    num_pp = 0
    num_pv = 0
    for term in terms
        if _is_variable(term.scalar_term.variable_1)
            if _is_variable(term.scalar_term.variable_2)
                num_vv += 1
            else
                num_pv += 1
            end
        else
            if _is_variable(term.scalar_term.variable_2)
                num_pv += 1
            else
                num_pp += 1
            end
        end
    end
    vv = Vector{MOI.VectorQuadraticTerm{T}}(undef, num_vv)
    pp = Vector{MOI.VectorQuadraticTerm{T}}(undef, num_pp)
    pv = Vector{MOI.VectorQuadraticTerm{T}}(undef, num_pv)
    i_vv = 1
    i_pp = 1
    i_pv = 1
    for term in terms
        if _is_variable(term.scalar_term.variable_1)
            if _is_variable(term.scalar_term.variable_2)
                vv[i_vv] = term
                i_vv += 1
            else
                pv[i_pv] = MOI.VectorQuadraticTerm(
                    term.output_index,
                    MOI.ScalarQuadraticTerm(
                        term.scalar_term.coefficient,
                        term.scalar_term.variable_2,
                        term.scalar_term.variable_1,
                    ),
                )
                i_pv += 1
            end
        else
            if _is_variable(term.scalar_term.variable_2)
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

function _parametric_affine_terms(
    model,
    f::ParametricVectorQuadraticFunction{T},
) where {T}
    param_terms_dict = Dict{Tuple{MOI.VariableIndex,Int},T}()
    sizehint!(
        param_terms_dict,
        length(vector_quadratic_parameter_variable_terms(f)),
    )

    for term in vector_quadratic_parameter_variable_terms(f)
        p_idx_val = p_idx(term.scalar_term.variable_1)
        var = term.scalar_term.variable_2
        output_idx = term.output_index
        base = get(param_terms_dict, (var, output_idx), zero(T))
        param_terms_dict[(var, output_idx)] =
            base + term.scalar_term.coefficient * model.parameters[p_idx_val]
    end

    for (term, coef) in f.affine_data
        output_idx = term.output_index
        var = term.scalar_term.variable
        param_terms_dict[(var, output_idx)] = coef
    end

    return param_terms_dict
end

function _delta_parametric_affine_terms(
    model,
    f::ParametricVectorQuadraticFunction{T},
) where {T}
    delta_terms = Dict{Tuple{Int,MOI.VariableIndex},T}()

    # Handle parameter-variable quadratic terms (px) that become affine (x) when p is updated
    for term in f.pv
        p_idx_val = p_idx(term.scalar_term.variable_1)
        var = term.scalar_term.variable_2
        output_idx = term.output_index

        if haskey(model.updated_parameters, p_idx_val) &&
           !isnan(model.updated_parameters[p_idx_val])
            old_param_val = model.parameters[p_idx_val]
            new_param_val = model.updated_parameters[p_idx_val]
            delta_coef =
                term.scalar_term.coefficient * (new_param_val - old_param_val)

            key = (output_idx, var)
            current_delta = get(delta_terms, key, zero(T))
            delta_terms[key] = current_delta + delta_coef
        end
    end

    # Handle parameter-only affine terms
    for term in f.p
        p_idx_val = p_idx(term.scalar_term.variable)
        output_idx = term.output_index

        if haskey(model.updated_parameters, p_idx_val) &&
           !isnan(model.updated_parameters[p_idx_val])
            old_param_val = model.parameters[p_idx_val]
            new_param_val = model.updated_parameters[p_idx_val]

            # This becomes a constant change, not an affine term change
            # We'll handle this in the constant update function
        end
    end

    return delta_terms
end

function _update_cache!(
    f::ParametricVectorQuadraticFunction{T},
    model,
) where {T}
    f.current_constant = _parametric_constant(model, f)
    f.current_terms_with_p = _parametric_affine_terms(model, f)
    return nothing
end

function _original_function(f::ParametricVectorQuadraticFunction{T}) where {T}
    return MOI.VectorQuadraticFunction{T}(
        vcat(
            vector_quadratic_parameter_variable_terms(f),
            vector_quadratic_parameter_parameter_terms(f),
            vector_quadratic_variable_variable_terms(f),
        ),
        vcat(vector_affine_parameter_terms(f), vector_affine_variable_terms(f)),
        f.c,
    )
end

function _parametric_constant(
    model,
    f::ParametricVectorQuadraticFunction{T},
) where {T}
    param_constant = copy(f.c)

    # Add contributions from parameter terms in affine part
    for term in vector_affine_parameter_terms(f)
        param_constant[term.output_index] +=
            term.scalar_term.coefficient *
            model.parameters[p_idx(term.scalar_term.variable)]
    end

    # Add contributions from parameter-parameter quadratic terms
    for term in vector_quadratic_parameter_parameter_terms(f)
        idx = term.output_index
        coef =
            term.scalar_term.coefficient /
            (term.scalar_term.variable_1 == term.scalar_term.variable_2 ? 2 : 1)
        param_constant[idx] +=
            coef *
            model.parameters[p_idx(term.scalar_term.variable_1)] *
            model.parameters[p_idx(term.scalar_term.variable_2)]
    end

    return param_constant
end

function _current_function(f::ParametricVectorQuadraticFunction{T}) where {T}
    affine_terms = MOI.VectorAffineTerm{T}[]
    sizehint!(affine_terms, length(f.current_constant) + length(f.v))
    for ((var, idx), coef) in f.current_terms_with_p
        push!(
            affine_terms,
            MOI.VectorAffineTerm{T}(idx, MOI.ScalarAffineTerm{T}(coef, var)),
        )
    end
    for ((var, idx), coef) in f.affine_data_np
        push!(
            affine_terms,
            MOI.VectorAffineTerm{T}(idx, MOI.ScalarAffineTerm{T}(coef, var)),
        )
    end
    return MOI.VectorQuadraticFunction{T}(
        f.vv,
        affine_terms,
        f.current_constant,
    )
end
