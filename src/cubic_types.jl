# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    _ScalarCubicTerm{T}

Represents a cubic term of the form `coefficient * index_1 * index_2 * index_3`.

Each index can be either a variable (MOI.VariableIndex) or a parameter (encoded as
VariableIndex with value > PARAMETER_INDEX_THRESHOLD).

# Fields
- `coefficient::T`: The numeric coefficient
- `index_1::MOI.VariableIndex`: First factor (parameter always)
- `index_2::MOI.VariableIndex`: Second factor (variable or parameter)
- `index_3::MOI.VariableIndex`: Third factor (variable or parameter)

# Convention
Indices are stored in canonical order:
- Parameters come before variables
This ensures `2*p*x*y` and `2*x*p*y` produce the same term.
"""
struct _ScalarCubicTerm{T}
    coefficient::T
    index_1::MOI.VariableIndex
    index_2::MOI.VariableIndex
    index_3::MOI.VariableIndex
end

"""
    _normalize_cubic_indices(idx1, idx2, idx3) -> (idx1, idx2, idx3)

Normalize cubic term indices to canonical order:
- Parameters come before variables
"""
function _normalize_cubic_indices(
    idx1::MOI.VariableIndex,
    idx2::MOI.VariableIndex,
    idx3::MOI.VariableIndex,
)
    params = MOI.VariableIndex[]
    vars = MOI.VariableIndex[]
    for idx in (idx1, idx2, idx3)
        if _is_parameter(idx)
            push!(params, idx)
        else
            push!(vars, idx)
        end
    end
    all_indices = vcat(params, vars)
    return all_indices[1], all_indices[2], all_indices[3]
end

"""
    _make_cubic_term(coefficient::T, idx1, idx2, idx3) where {T}

Create a cubic term with normalized index order.
"""
function _make_cubic_term(
    coefficient::T,
    idx1::MOI.VariableIndex,
    idx2::MOI.VariableIndex,
    idx3::MOI.VariableIndex,
) where {T}
    n1, n2, n3 = _normalize_cubic_indices(idx1, idx2, idx3)
    return _ScalarCubicTerm{T}(coefficient, n1, n2, n3)
end
