# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    ParametricCubicFunction{T} <: ParametricFunction{T}

Represents a cubic function where parameters multiply up to quadratic variable terms.

Supports the general form:
    constant + Σ(affine) + Σ(quadratic) + Σ(cubic)

After parameter substitution, cubic terms become:
- PVV (p*x*y) → quadratic term (c*p_val*x*y)
- PPV (p*q*x) → affine term (c*p_val*q_val*x)
- PPP (p*q*r) → constant (c*p_val*q_val*r_val)
"""
mutable struct ParametricCubicFunction{T} <: ParametricFunction{T}
    # === Cubic terms (degree 3) - split by type like quadratic terms ===
    pvv::Vector{_ScalarCubicTerm{T}}  # p*x*y → becomes quadratic
    ppv::Vector{_ScalarCubicTerm{T}}  # p*q*x → becomes affine
    ppp::Vector{_ScalarCubicTerm{T}}  # p*q*r → becomes constant

    # === Quadratic terms (degree 2) - same pattern as ParametricQuadraticFunction ===
    pv::Vector{MOI.ScalarQuadraticTerm{T}}   # p*x → becomes affine
    pp::Vector{MOI.ScalarQuadraticTerm{T}}   # p*q → becomes constant
    vv::Vector{MOI.ScalarQuadraticTerm{T}}   # x*y → stays quadratic

    # === Affine terms (degree 1) ===
    p::Vector{MOI.ScalarAffineTerm{T}}       # p → becomes constant
    v::Vector{MOI.ScalarAffineTerm{T}}       # x → stays affine

    # === Constant (degree 0) ===
    c::T

    # === Caches for efficient updates ===
    # Quadratic coefficients (from vv + pvv terms) - tracks current values in solver
    quadratic_data::Dict{Tuple{MOI.VariableIndex,MOI.VariableIndex},T}
    # Affine coefficients (from v + pv + ppv terms)
    affine_data::Dict{MOI.VariableIndex,T}
    # Affine coefficients not dependent on parameters
    affine_data_np::Dict{MOI.VariableIndex,T}
    # Current constant after parameter substitution
    current_constant::T
    # Set constant (for constraint handling, not used for objectives)
    set_constant::T
end

"""
    ParametricCubicFunction(parsed::_ParsedCubicExpression{T}) where {T}

Construct a ParametricCubicFunction from a _ParsedCubicExpression.
"""
function ParametricCubicFunction(parsed::_ParsedCubicExpression{T}) where {T}

    # Find variables related to parameters (from pv and ppv terms)
    v_in_param_terms = Set{MOI.VariableIndex}()
    for term in parsed.pv
        push!(v_in_param_terms, term.variable_2)
    end
    for term in parsed.ppv
        var = term.index_3
        push!(v_in_param_terms, var)
    end

    # Split affine data
    affine_data = Dict{MOI.VariableIndex,T}()
    affine_data_np = Dict{MOI.VariableIndex,T}()
    for term in parsed.v
        if term.variable in v_in_param_terms
            affine_data[term.variable] =
                get(affine_data, term.variable, zero(T)) + term.coefficient
        else
            affine_data_np[term.variable] =
                get(affine_data_np, term.variable, zero(T)) + term.coefficient
        end
    end

    # Find variable pairs related to parameters (from pvv terms)
    var_pairs_in_param_terms = Set{Tuple{MOI.VariableIndex,MOI.VariableIndex}}()
    for term in parsed.pvv
        first_is_greater = term.index_2.value > term.index_3.value
        v1 = ifelse(first_is_greater, term.index_3, term.index_2)
        v2 = ifelse(first_is_greater, term.index_2, term.index_3)
        push!(var_pairs_in_param_terms, (v1, v2))
    end

    # Initialize quadratic data
    # Note: vv terms come from the parsed quadratic_func, which already has
    # the MOI coefficient convention applied. We need to convert to internal form.
    # MOI convention:
    #   - Off-diagonal (v1 != v2): coefficient C means C*v1*v2 (use as-is)
    #   - Diagonal (v1 == v2): coefficient C means (C/2)*v1^2 (divide by 2)
    quadratic_data = Dict{Tuple{MOI.VariableIndex,MOI.VariableIndex},T}()
    for term in parsed.vv
        first_is_greater = term.variable_1.value > term.variable_2.value
        v1 = ifelse(first_is_greater, term.variable_2, term.variable_1)
        v2 = ifelse(first_is_greater, term.variable_1, term.variable_2)
        coef = term.coefficient
        if term.variable_1 == term.variable_2
            coef = coef / 2  # Diagonal: undo MOI's factor
        end
        quadratic_data[(v1, v2)] = get(quadratic_data, (v1, v2), zero(T)) + coef
    end
    # Add entries for pvv terms (will be updated with parameter values later)
    for pair in var_pairs_in_param_terms
        if !haskey(quadratic_data, pair)
            quadratic_data[pair] = zero(T)
        end
    end

    return ParametricCubicFunction{T}(
        parsed.pvv,
        parsed.ppv,
        parsed.ppp,
        parsed.pv,
        parsed.pp,
        parsed.vv,
        parsed.p,
        parsed.v,
        parsed.constant,
        quadratic_data,
        affine_data,
        affine_data_np,
        zero(T),  # current_constant (computed later)
        zero(T),  # set_constant
    )
end

# Accessors for cubic terms by type (direct field access)
_cubic_pvv_terms(f::ParametricCubicFunction) = f.pvv
_cubic_ppv_terms(f::ParametricCubicFunction) = f.ppv
_cubic_ppp_terms(f::ParametricCubicFunction) = f.ppp

"""
    _effective_param_value(model, pi::ParameterIndex)

Get the effective parameter value: updated value if available, otherwise current value.
"""
function _effective_param_value(model, pi::ParameterIndex)
    if haskey(model.updated_parameters, pi) &&
       !isnan(model.updated_parameters[pi])
        return model.updated_parameters[pi]
    end
    return model.parameters[pi]
end

"""
    _parametric_constant(model, f::ParametricCubicFunction{T}) where {T}

Compute the constant term after parameter substitution.
Includes contributions from: c + p terms + pp terms + ppp cubic terms
"""
function _parametric_constant(model, f::ParametricCubicFunction{T}) where {T}
    constant = f.c

    # From affine parameter terms (p)
    for term in f.p
        constant +=
            term.coefficient *
            _effective_param_value(model, p_idx(term.variable))
    end

    # From quadratic parameter-parameter terms (pp)
    # MOI convention: diagonal C means C/2*p^2, off-diagonal C means C*p1*p2
    for term in f.pp
        divisor = term.variable_1 == term.variable_2 ? 2 : 1
        constant +=
            (term.coefficient / divisor) *
            _effective_param_value(model, p_idx(term.variable_1)) *
            _effective_param_value(model, p_idx(term.variable_2))
    end

    # From cubic ppp terms (all 3 indices are parameters)
    for term in _cubic_ppp_terms(f)
        p1 = term.index_1
        p2 = term.index_2
        p3 = term.index_3
        constant +=
            term.coefficient *
            _effective_param_value(model, p_idx(p1)) *
            _effective_param_value(model, p_idx(p2)) *
            _effective_param_value(model, p_idx(p3))
    end

    return constant
end

"""
    _parametric_affine_terms(model, f::ParametricCubicFunction{T}) where {T}

Compute affine coefficients after parameter substitution.
Includes contributions from: v terms + pv terms + ppv cubic terms
"""
function _parametric_affine_terms(
    model,
    f::ParametricCubicFunction{T},
) where {T}
    # Start with non-parametric terms
    terms_dict = copy(f.affine_data)

    # Add contributions from pv terms (parameter * variable)
    # These are always off-diagonal (p != v), so coefficient is used as-is
    for term in f.pv
        var = term.variable_2
        coef = term.coefficient
        p_val = _effective_param_value(model, p_idx(term.variable_1))
        terms_dict[var] = get(terms_dict, var, zero(T)) + coef * p_val
    end

    # Add contributions from ppv cubic terms
    for term in _cubic_ppv_terms(f)
        var = term.index_3
        p1_val = _effective_param_value(model, p_idx(term.index_1))
        p2_val = _effective_param_value(model, p_idx(term.index_2))
        terms_dict[var] =
            get(terms_dict, var, zero(T)) + term.coefficient * p1_val * p2_val
    end

    return terms_dict
end

"""
    _parametric_quadratic_terms(model, f::ParametricCubicFunction{T}) where {T}

Compute quadratic coefficients after parameter substitution.
Includes contributions from: vv terms + pvv terms
"""
function _parametric_quadratic_terms(
    model,
    f::ParametricCubicFunction{T},
) where {T}
    # Start with vv terms
    terms_dict = copy(f.quadratic_data)

    # Add contributions from pvv cubic terms
    for term in _cubic_pvv_terms(f)
        p = term.index_1
        first_is_greater = term.index_2.value > term.index_3.value
        v1 = ifelse(first_is_greater, term.index_3, term.index_2)
        v2 = ifelse(first_is_greater, term.index_2, term.index_3)
        var_pair = (v1, v2)
        p_val = _effective_param_value(model, p_idx(p))
        terms_dict[var_pair] =
            get(terms_dict, var_pair, zero(T)) + term.coefficient * p_val
    end

    return terms_dict
end

"""
    _current_function(f::ParametricCubicFunction{T}, model) where {T}

Evaluate the cubic function with current parameter values and return
the appropriate MOI function type.
"""
function _current_function(f::ParametricCubicFunction{T}, model) where {T}
    # Get current values
    quad_data = _parametric_quadratic_terms(model, f)
    affine_data = _parametric_affine_terms(model, f)
    constant = _parametric_constant(model, f)

    # Build quadratic terms
    # MOI convention:
    #   - Off-diagonal (v1 != v2): coefficient C means C*v1*v2 (use as-is)
    #   - Diagonal (v1 == v2): coefficient C means (C/2)*v1^2 (multiply by 2)
    quadratic_terms = MOI.ScalarQuadraticTerm{T}[]
    for ((v1, v2), coef) in quad_data
        if !iszero(coef)
            moi_coef = v1 == v2 ? coef * 2 : coef
            push!(quadratic_terms, MOI.ScalarQuadraticTerm{T}(moi_coef, v1, v2))
        end
    end

    # Build affine terms
    affine_terms = MOI.ScalarAffineTerm{T}[]
    for (v, coef) in affine_data
        if !iszero(coef)
            push!(affine_terms, MOI.ScalarAffineTerm{T}(coef, v))
        end
    end
    # Add non-parametric affine terms
    for (v, coef) in f.affine_data_np
        if !iszero(coef)
            push!(affine_terms, MOI.ScalarAffineTerm{T}(coef, v))
        end
    end

    # Note: We don't update f.affine_data or f.quadratic_data here.
    # These store the BASE coefficients (from v and vv terms) and must remain unchanged.
    # current_constant is the only cache we update for reference.
    f.current_constant = constant

    # Always return a ScalarQuadraticFunction, even if it has no quadratic terms.
    return MOI.ScalarQuadraticFunction{T}(
        quadratic_terms,
        affine_terms,
        constant,
    )
end

# === Delta functions for efficient updates ===

"""
    _delta_parametric_constant(model, f::ParametricCubicFunction{T}) where {T}

Compute the CHANGE in constant when parameters are updated.
"""
function _delta_parametric_constant(
    model,
    f::ParametricCubicFunction{T},
) where {T}
    delta = zero(T)

    # From p terms
    for term in f.p
        p_i = p_idx(term.variable)
        if haskey(model.updated_parameters, p_i) &&
           !isnan(model.updated_parameters[p_i])
            old_val = model.parameters[p_i]
            new_val = model.updated_parameters[p_i]
            delta += term.coefficient * (new_val - old_val)
        end
    end

    # From pp terms
    for term in f.pp
        pi1 = p_idx(term.variable_1)
        pi2 = p_idx(term.variable_2)
        updated1 =
            haskey(model.updated_parameters, pi1) &&
            !isnan(model.updated_parameters[pi1])
        updated2 =
            haskey(model.updated_parameters, pi2) &&
            !isnan(model.updated_parameters[pi2])

        if updated1 || updated2
            divisor = term.variable_1 == term.variable_2 ? 2 : 1
            old_val =
                (term.coefficient / divisor) *
                model.parameters[pi1] *
                model.parameters[pi2]
            new_p1 =
                updated1 ? model.updated_parameters[pi1] : model.parameters[pi1]
            new_p2 =
                updated2 ? model.updated_parameters[pi2] : model.parameters[pi2]
            new_val = (term.coefficient / divisor) * new_p1 * new_p2
            delta += new_val - old_val
        end
    end

    # From ppp cubic terms
    for term in _cubic_ppp_terms(f)
        pi1 = p_idx(term.index_1)
        pi2 = p_idx(term.index_2)
        pi3 = p_idx(term.index_3)
        updated1 =
            haskey(model.updated_parameters, pi1) &&
            !isnan(model.updated_parameters[pi1])
        updated2 =
            haskey(model.updated_parameters, pi2) &&
            !isnan(model.updated_parameters[pi2])
        updated3 =
            haskey(model.updated_parameters, pi3) &&
            !isnan(model.updated_parameters[pi3])

        if updated1 || updated2 || updated3
            old_val =
                term.coefficient *
                model.parameters[pi1] *
                model.parameters[pi2] *
                model.parameters[pi3]
            new_p1 =
                updated1 ? model.updated_parameters[pi1] : model.parameters[pi1]
            new_p2 =
                updated2 ? model.updated_parameters[pi2] : model.parameters[pi2]
            new_p3 =
                updated3 ? model.updated_parameters[pi3] : model.parameters[pi3]
            new_val = term.coefficient * new_p1 * new_p2 * new_p3
            delta += new_val - old_val
        end
    end

    return delta
end

"""
    _delta_parametric_affine_terms(model, f::ParametricCubicFunction{T}) where {T}

Compute the CHANGE in affine coefficients when parameters are updated.
"""
function _delta_parametric_affine_terms(
    model,
    f::ParametricCubicFunction{T},
) where {T}
    delta_dict = Dict{MOI.VariableIndex,T}()

    # From pv terms (parameter * variable, always off-diagonal)
    for term in f.pv
        p_i = p_idx(term.variable_1)
        if haskey(model.updated_parameters, p_i) &&
           !isnan(model.updated_parameters[p_i])
            var = term.variable_2
            coef = term.coefficient  # Off-diagonal: use as-is
            old_val = model.parameters[p_i]
            new_val = model.updated_parameters[p_i]
            delta_dict[var] =
                get(delta_dict, var, zero(T)) + coef * (new_val - old_val)
        end
    end

    # From ppv cubic terms
    for term in _cubic_ppv_terms(f)
        var = term.index_3
        pi1 = p_idx(term.index_1)
        pi2 = p_idx(term.index_2)
        updated1 =
            haskey(model.updated_parameters, pi1) &&
            !isnan(model.updated_parameters[pi1])
        updated2 =
            haskey(model.updated_parameters, pi2) &&
            !isnan(model.updated_parameters[pi2])

        if updated1 || updated2
            old_val =
                term.coefficient * model.parameters[pi1] * model.parameters[pi2]
            new_p1 =
                updated1 ? model.updated_parameters[pi1] : model.parameters[pi1]
            new_p2 =
                updated2 ? model.updated_parameters[pi2] : model.parameters[pi2]
            new_val = term.coefficient * new_p1 * new_p2
            delta_dict[var] =
                get(delta_dict, var, zero(T)) + (new_val - old_val)
        end
    end

    return delta_dict
end

"""
    _delta_parametric_quadratic_terms(model, f::ParametricCubicFunction{T}) where {T}

Compute the CHANGE in quadratic coefficients when parameters are updated.
"""
function _delta_parametric_quadratic_terms(
    model,
    f::ParametricCubicFunction{T},
) where {T}
    delta_dict = Dict{Tuple{MOI.VariableIndex,MOI.VariableIndex},T}()

    for term in _cubic_pvv_terms(f)
        p_i = p_idx(term.index_1)
        first_is_greater = term.index_2.value > term.index_3.value
        v1 = ifelse(first_is_greater, term.index_3, term.index_2)
        v2 = ifelse(first_is_greater, term.index_2, term.index_3)
        var_pair = (v1, v2)

        if haskey(model.updated_parameters, p_i) &&
           !isnan(model.updated_parameters[p_i])
            old_val = model.parameters[p_i]
            new_val = model.updated_parameters[p_i]
            delta = term.coefficient * (new_val - old_val)
            delta_dict[var_pair] = get(delta_dict, var_pair, zero(T)) + delta
        end
    end

    return delta_dict
end
