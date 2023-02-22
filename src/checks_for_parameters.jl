function isparameter(vi::MOI.VariableIndex)
    return vi.value > VARIABLE_INDEX_THRESHOLD
end

function isvariable(vi::MOI.VariableIndex)
    return vi.value <= VARIABLE_INDEX_THRESHOLD
end

function function_has_parameters(f::MOI.ScalarAffineFunction{T}) where {T}
    for term in f.terms
        if isparameter(term.variable)
            return true
        end
    end
    return false
end

function has_at_lest_two_variables(f::MOI.ScalarAffineFunction{T}) where {T}
    count = 0
    for term in f.terms
        if isvariable(term.variable)
            count += 1
        end
        if count >= 2
            return true
        end
    end
    return false
end