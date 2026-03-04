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
    p1 = _is_parameter(idx1)
    p2 = _is_parameter(idx2)
    p3 = _is_parameter(idx3)
    # Place parameters before variables, preserving relative order within each group.
    if p1 && p2 && p3
        return idx1, idx2, idx3  # ppp — already ordered
    elseif p1 && p2             # p p v
        return idx1, idx2, idx3
    elseif p1 && p3             # p v p → p p v
        return idx1, idx3, idx2
    elseif p2 && p3             # v p p → p p v
        return idx2, idx3, idx1
    elseif p1                   # p v v — already ordered
        return idx1, idx2, idx3
    elseif p2                   # v p v → p v v
        return idx2, idx1, idx3
    elseif p3                   # v v p → p v v
        return idx3, idx1, idx2
    else                        # v v v — no parameter (caller validates)
        return idx1, idx2, idx3
    end
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
