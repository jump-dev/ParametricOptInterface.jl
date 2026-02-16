# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    _Monomial{T}

Intermediate representation of a monomial during parsing.
"""
struct _Monomial{T}
    coefficient::T
    variables::Vector{MOI.VariableIndex}   # includes both vars and params
end

function _Monomial{T}(coefficient::T) where {T}
    return _Monomial{T}(coefficient, MOI.VariableIndex[])
end

function _Monomial{T}(coefficient::T, var::MOI.VariableIndex) where {T}
    return _Monomial{T}(coefficient, [var])
end

"""
    _monomial_degree(m::_Monomial) -> Int

Total degree of a monomial (number of variable/parameter factors).
"""
function _monomial_degree(m::_Monomial)
    return length(m.variables)
end

"""
    _multiply_monomials(m1::_Monomial{T}, m2::_Monomial{T}) where {T}

Multiply two monomials together.
"""
function _multiply_monomials(m1::_Monomial{T}, m2::_Monomial{T}) where {T}
    return _Monomial{T}(
        m1.coefficient * m2.coefficient,
        vcat(m1.variables, m2.variables),
    )
end

"""
    _scale_monomial(m::_Monomial{T}, scalar::T) where {T}

Scale a monomial by a scalar.
"""
function _scale_monomial(m::_Monomial{T}, scalar::T) where {T}
    return _Monomial{T}(m.coefficient * scalar, copy(m.variables))
end

"""
    _is_polynomial_operator(head::Symbol) -> Bool

Check if the operator is valid for polynomial expressions.
"""
function _is_polynomial_operator(head::Symbol)
    return head in (:+, :-, :*, :^, :/)
end

"""
    _ParsedCubicExpression{T}

Result of parsing a ScalarNonlinearFunction into cubic polynomial form.
"""
struct _ParsedCubicExpression{T}
    pvv::Vector{_ScalarCubicTerm{T}}  # Cubic terms with 1 parameter and 2 variables
    ppv::Vector{_ScalarCubicTerm{T}}  # Cubic terms with 2 parameters and 1 variable
    ppp::Vector{_ScalarCubicTerm{T}}  # Cubic terms with 3 parameters

    vv::Vector{MOI.ScalarQuadraticTerm{T}}  # Quadratic terms with 2 variables
    pv::Vector{MOI.ScalarQuadraticTerm{T}}  # Quadratic terms with 1 parameter and 1 variable
    pp::Vector{MOI.ScalarQuadraticTerm{T}}  # Quadratic terms with 2 parameters

    v::Vector{MOI.ScalarAffineTerm{T}}  # Affine terms with 1 variable
    p::Vector{MOI.ScalarAffineTerm{T}}  # Affine terms with 1 parameter

    constant::T  # Constant term
end

"""
    _expand_to_monomials(arg, ::Type{T}) where {T} -> Union{Vector{_Monomial{T}}, Nothing}

Expand an expression argument to a list of monomials.
Returns `nothing` if the expression is not a valid polynomial.
"""
function _expand_to_monomials(arg::Real, ::Type{T}) where {T}
    return [_Monomial{T}(T(arg))]
end

function _expand_to_monomials(arg::MOI.VariableIndex, ::Type{T}) where {T}
    return [_Monomial{T}(one(T), arg)]
end

function _expand_to_monomials(
    arg::MOI.ScalarAffineFunction{T},
    ::Type{T},
) where {T}
    monomials = _Monomial{T}[]
    for term in arg.terms
        push!(monomials, _Monomial{T}(term.coefficient, term.variable))
    end
    if !iszero(arg.constant)
        push!(monomials, _Monomial{T}(arg.constant))
    end
    return monomials
end

function _expand_to_monomials(
    arg::MOI.ScalarQuadraticFunction{T},
    ::Type{T},
) where {T}
    monomials = _Monomial{T}[]
    # Quadratic terms
    # MOI convention:
    #   - Off-diagonal (v1 != v2): coefficient C represents C*v1*v2
    #   - Diagonal (v1 == v2): coefficient C represents (C/2)*v1^2
    for term in arg.quadratic_terms
        coef = term.coefficient
        if term.variable_1 == term.variable_2
            coef = coef / 2  # Diagonal: undo MOI's factor of 2
        end
        # Off-diagonal: use coefficient as-is
        push!(monomials, _Monomial{T}(coef, [term.variable_1, term.variable_2]))
    end
    # Affine terms
    for term in arg.affine_terms
        push!(monomials, _Monomial{T}(term.coefficient, term.variable))
    end
    # Constant
    if !iszero(arg.constant)
        push!(monomials, _Monomial{T}(arg.constant))
    end
    return monomials
end

function _expand_to_monomials(
    f::MOI.ScalarNonlinearFunction,
    ::Type{T},
) where {T}
    head = f.head
    args = f.args

    if !_is_polynomial_operator(head)
        return nothing  # Non-polynomial operator
    end

    if head == :+
        return _expand_addition(args, T)
    elseif head == :-
        return _expand_subtraction(args, T)
    elseif head == :*
        return _expand_multiplication(args, T)
    elseif head == :/
        return _expand_division(args, T)
    elseif head == :^
        return _expand_power(args, T)
    end

    return nothing
end

"""
    _expand_addition(args, ::Type{T}) where {T}

Expand addition: collect monomials from all arguments.
"""
function _expand_addition(args, ::Type{T}) where {T}
    result = _Monomial{T}[]
    for arg in args
        monomials = _expand_to_monomials(arg, T)
        if monomials === nothing
            return nothing
        end
        append!(result, monomials)
    end
    return result
end

"""
    _expand_subtraction(args, ::Type{T}) where {T}

Expand subtraction: first arg positive, rest negative.
"""
function _expand_subtraction(args, ::Type{T}) where {T}
    result = _Monomial{T}[]

    if length(args) == 1
        # Unary minus
        monomials = _expand_to_monomials(args[1], T)
        if monomials === nothing
            return nothing
        end
        for m in monomials
            push!(result, _scale_monomial(m, -one(T)))
        end
    else
        # Binary subtraction
        for (i, arg) in enumerate(args)
            monomials = _expand_to_monomials(arg, T)
            if monomials === nothing
                return nothing
            end
            if i == 1
                append!(result, monomials)
            else
                for m in monomials
                    push!(result, _scale_monomial(m, -one(T)))
                end
            end
        end
    end
    return result
end

"""
    _expand_multiplication(args, ::Type{T}) where {T}

Expand multiplication: multiply all arguments together.
"""
function _expand_multiplication(args, ::Type{T}) where {T}
    # Start with identity monomial
    result = [_Monomial{T}(one(T))]

    for arg in args
        monomials = _expand_to_monomials(arg, T)
        if monomials === nothing
            return nothing
        end

        # Multiply each result monomial with each new monomial
        new_result = _Monomial{T}[]
        for m1 in result
            for m2 in monomials
                push!(new_result, _multiply_monomials(m1, m2))
            end
        end
        result = new_result
    end

    return result
end

"""
    _expand_division(args, ::Type{T}) where {T}

Expand division: multiply numerator by the inverse of denominator
"""
function _expand_division(args, ::Type{T}) where {T}
    if length(args) != 2
        return nothing
    end

    numerator = args[1]
    denominator = args[2]

    # denominator must be a nonzero constant (no variables or parameters)
    if !(denominator isa Real) || iszero(denominator)
        return nothing
    end

    return _expand_multiplication([one(T) / denominator, numerator], T)
end

"""
    _expand_power(args, ::Type{T}) where {T}

Expand power: x^n becomes x*x*...*x (n times).
"""
function _expand_power(args, ::Type{T}) where {T}
    if length(args) != 2
        return nothing
    end

    base = args[1]
    exponent = args[2]

    # Exponent must be a non-negative integer
    if !(exponent isa Integer) || exponent < 0
        return nothing
    end

    n = Int(exponent)

    if n == 0
        return [_Monomial{T}(one(T))]
    end

    base_monomials = _expand_to_monomials(base, T)
    if base_monomials === nothing
        return nothing
    end

    # x^n = x * x * ... * x (n times)
    result = base_monomials
    for _ in 2:n
        new_result = _Monomial{T}[]
        for m1 in result
            for m2 in base_monomials
                push!(new_result, _multiply_monomials(m1, m2))
            end
        end
        result = new_result
    end

    return result
end

"""
    _combine_like_monomials(monomials::Vector{_Monomial{T}}) where {T}

Combine like monomials (same variables, regardless of order).
"""
function _combine_like_monomials(monomials::Vector{_Monomial{T}}) where {T}
    # Use a dict keyed by sorted variable tuple
    combined = Dict{Vector{MOI.VariableIndex},T}()

    for m in monomials
        # Sort variables for canonical key
        key = sort(m.variables, by = v -> v.value)
        combined[key] = get(combined, key, zero(T)) + m.coefficient
    end

    result = _Monomial{T}[]
    for (vars, coef) in combined
        if !iszero(coef)
            push!(result, _Monomial{T}(coef, vars))
        end
    end

    return result
end

"""
    _classify_monomial(m::_Monomial) -> Symbol

Classify a monomial by its structure.
"""
function _classify_monomial(m::_Monomial)
    degree = _monomial_degree(m)
    num_params = count(_is_parameter, m.variables)

    if degree == 0
        return :constant
    elseif degree == 1
        return num_params == 1 ? :p : :v
    elseif degree == 2
        if num_params == 0
            return :vv
        elseif num_params == 1
            return :pv
        else
            return :pp
        end
    elseif degree == 3
        if num_params == 0
            return :vvv  # Invalid - no parameter
        elseif num_params == 1
            return :pvv
        elseif num_params == 2
            return :ppv
        else
            return :ppp
        end
    else
        return :invalid  # Degree > 3
    end
end

"""
    _parse_cubic_expression(f::MOI.ScalarNonlinearFunction, ::Type{T}) where {T} -> Union{_ParsedCubicExpression{T}, Nothing}

Parse a ScalarNonlinearFunction and return a _ParsedCubicExpression if it represents
a valid cubic polynomial (with parameters multiplying at most quadratic variable terms).

Returns `nothing` if the expression:
- Contains non-polynomial operations (sin, exp, etc.)
- Has degree > 3 in any monomial
- Has a cubic term with no parameters (x*y*z)
"""
function _parse_cubic_expression(
    f::MOI.ScalarNonlinearFunction,
    ::Type{T},
) where {T}
    # Expand to monomials
    monomials = _expand_to_monomials(f, T)
    if monomials === nothing
        return nothing
    end

    # Combine like terms
    monomials = _combine_like_monomials(monomials)

    # Classify and collect terms
    cubic_terms = _ScalarCubicTerm{T}[]

    cubic_ppp = _ScalarCubicTerm{T}[]
    cubic_ppv = _ScalarCubicTerm{T}[]
    cubic_pvv = _ScalarCubicTerm{T}[]

    quadratic_pp = MOI.ScalarQuadraticTerm{T}[]
    quadratic_pv = MOI.ScalarQuadraticTerm{T}[]
    quadratic_vv = MOI.ScalarQuadraticTerm{T}[]

    affine_p = MOI.ScalarAffineTerm{T}[]
    affine_v = MOI.ScalarAffineTerm{T}[]

    constant = zero(T)

    for m in monomials
        classification = _classify_monomial(m)

        if classification == :invalid || classification == :vvv
            return nothing  # Invalid degree or no parameter in cubic
        elseif classification == :constant
            constant += m.coefficient
        elseif classification == :v
            push!(
                affine_v,
                MOI.ScalarAffineTerm{T}(m.coefficient, m.variables[1]),
            )
        elseif classification == :p
            push!(
                affine_p,
                MOI.ScalarAffineTerm{T}(m.coefficient, m.variables[1]),
            )
        elseif classification == :pp
            p1 = m.variables[1]
            p2 = m.variables[2]
            # Sort for canonical order
            if p1.value > p2.value
                p1, p2 = p2, p1
            end
            divisor = p1 == p2 ? T(2) : T(1)  # Diagonal vs off-diagonal
            push!(
                quadratic_pp,
                MOI.ScalarQuadraticTerm{T}(m.coefficient * divisor, p1, p2),
            )
        elseif classification == :pv
            # Convention: variable_1 = parameter, variable_2 = variable
            # This matches the expectation in _parametric_affine_terms and
            # _delta_parametric_affine_terms
            if _is_parameter(m.variables[1])
                p_idx_v, v_idx_v = m.variables[1], m.variables[2]
            else
                p_idx_v, v_idx_v = m.variables[2], m.variables[1]
            end
            push!(
                quadratic_pv,
                MOI.ScalarQuadraticTerm{T}(
                    m.coefficient,
                    p_idx_v,
                    v_idx_v,
                ),
            )
        elseif classification == :vv
            v1 = m.variables[1]
            v2 = m.variables[2]
            # Sort for canonical order
            if v1.value > v2.value
                v1, v2 = v2, v1
            end
            divisor = v1 == v2 ? T(2) : T(1)  # Diagonal vs off-diagonal
            push!(
                quadratic_vv,
                MOI.ScalarQuadraticTerm{T}(m.coefficient * divisor, v1, v2),
            )
        elseif classification == :ppp
            push!(
                cubic_ppp,
                _make_cubic_term(
                    m.coefficient,
                    m.variables[1],
                    m.variables[2],
                    m.variables[3],
                ),
            )
        elseif classification == :ppv
            push!(
                cubic_ppv,
                _make_cubic_term(
                    m.coefficient,
                    m.variables[1],
                    m.variables[2],
                    m.variables[3],
                ),
            )
        else  # classification == :pvv
            push!(
                cubic_pvv,
                _make_cubic_term(
                    m.coefficient,
                    m.variables[1],
                    m.variables[2],
                    m.variables[3],
                ),
            )
        end
    end

    return _ParsedCubicExpression{T}(
        cubic_pvv,
        cubic_ppv,
        cubic_ppp,
        quadratic_vv,
        quadratic_pv,
        quadratic_pp,
        affine_v,
        affine_p,
        constant,
    )
end

# Convenience method with type inference
function _parse_cubic_expression(f::MOI.ScalarNonlinearFunction)
    return _parse_cubic_expression(f, Float64)
end
