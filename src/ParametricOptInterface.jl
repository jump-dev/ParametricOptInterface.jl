module ParametricOptInterface

using MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const POI = ParametricOptInterface
const PARAMETER_INDEX_THRESHOLD = 1_000_000_000_000_000_000

"""
    Parameter(Float64)

The `Parameter` structure stores the numerical value associated to a given parameter.
# Example:
```julia-repl
julia> ParametricOptInterface.Parameter(5)
ParametricOptInterface.Parameter(5)
```
"""
struct Parameter <: MOI.AbstractScalarSet
    val::Float64
end

"""
    ParametricOptimizer{T, OT <: MOI.ModelLike} <: MOI.AbstractOptimizer

Declares a `ParametricOptimizer`, which allows the handling of parameters in a optimization model.
# Example:
```julia-repl
julia> ParametricOptInterface.ParametricOptimizer(GLPK.Optimizer())
ParametricOptInterface.ParametricOptimizer{Float64,GLPK.Optimizer}
```
"""
mutable struct ParametricOptimizer{T, OT <: MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT 
    parameters::Dict{MOI.VariableIndex, T}
    parameters_name::Dict{MOI.VariableIndex, String}
    updated_parameters::Dict{MOI.VariableIndex, T}
    variables::Dict{MOI.VariableIndex, MOI.VariableIndex}
    last_variable_index_added::Int
    last_parameter_index_added::Int
    affine_constraint_cache::Dict{MOI.ConstraintIndex, Array{MOI.ScalarAffineTerm{Float64},1}}
    quadratic_constraint_cache_pv::Dict{MOI.ConstraintIndex, Array{MOI.ScalarQuadraticTerm{Float64},1}}
    quadratic_constraint_cache_pp::Dict{MOI.ConstraintIndex, Array{MOI.ScalarQuadraticTerm{Float64},1}}
    quadratic_constraint_cache_pc::Dict{MOI.ConstraintIndex, Array{MOI.ScalarAffineTerm{Float64},1}}
    quadratic_constraint_variables_associated_to_parameters_cache::Dict{MOI.ConstraintIndex, Array{MOI.ScalarAffineTerm{Float64},1}} 
    quadratic_added_cache::Dict{MOI.ConstraintIndex, MOI.ConstraintIndex} 
    last_quad_add_added::Int
    affine_objective_cache::Array{MOI.ScalarAffineTerm{T},1}
    quadratic_objective_cache_pv::Array{MOI.ScalarQuadraticTerm{T},1}
    quadratic_objective_cache_pp::Array{MOI.ScalarQuadraticTerm{T},1}
    quadratic_objective_cache_pc::Array{MOI.ScalarAffineTerm{T},1}
    quadratic_objective_variables_associated_to_parameters_cache::Array{MOI.ScalarAffineTerm{T},1}
    function ParametricOptimizer(optimizer::OT) where OT
        new{Float64, OT}(
            optimizer,
            Dict{MOI.VariableIndex, Float64}(),
            Dict{MOI.VariableIndex, String}(),
            Dict{MOI.VariableIndex, Float64}(),
            Dict{MOI.VariableIndex, MOI.VariableIndex}(),
            0,
            PARAMETER_INDEX_THRESHOLD,
            Dict{MOI.ConstraintIndex, Array{MOI.ScalarAffineTerm{Float64},1}}(),
            Dict{MOI.ConstraintIndex, Array{MOI.ScalarQuadraticTerm{Float64},1}}(),
            Dict{MOI.ConstraintIndex, Array{MOI.ScalarQuadraticTerm{Float64},1}}(),
            Dict{MOI.ConstraintIndex, Array{MOI.ScalarAffineTerm{Float64},1}}(),
            Dict{MOI.ConstraintIndex, Array{MOI.ScalarAffineTerm{Float64},1}}(),
            Dict{MOI.ConstraintIndex, MOI.ConstraintIndex}(),
            0,
            Array{MOI.ScalarAffineTerm{Float64},1}(),
            Array{MOI.ScalarQuadraticTerm{Float64},1}(),
            Array{MOI.ScalarQuadraticTerm{Float64},1}(),
            Array{MOI.ScalarAffineTerm{Float64},1}(),
            Array{MOI.ScalarAffineTerm{Float64},1}()
        )
    end
end

include("utils.jl")

function MOI.is_empty(model::ParametricOptimizer)
    return MOI.is_empty(model.optimizer) &&
    isempty(model.parameters) &&
    isempty(model.parameters_name) &&
    isempty(model.variables) &&
    isempty(model.updated_parameters) &&
    isempty(model.variables) &&
    model.last_variable_index_added == 0 &&
    model.last_parameter_index_added == PARAMETER_INDEX_THRESHOLD &&
    isempty(model.affine_constraint_cache) &&
    isempty(model.quadratic_constraint_cache_pv) &&
    isempty(model.quadratic_constraint_cache_pp) &&
    isempty(model.quadratic_constraint_cache_pc) &&
    isempty(model.quadratic_constraint_variables_associated_to_parameters_cache) &&
    isempty(model.quadratic_added_cache) &&
    model.last_quad_add_added == 0 &&
    isempty(model.affine_objective_cache) &&
    isempty(model.quadratic_objective_cache_pv) &&
    isempty(model.quadratic_objective_cache_pp) &&
    isempty(model.quadratic_objective_cache_pc) &&
    isempty(model.quadratic_objective_variables_associated_to_parameters_cache)
end

function MOI.supports_constraint(
    model::ParametricOptimizer,
    F::Union{Type{MOI.SingleVariable}, 
             Type{MOI.ScalarAffineFunction{T}}, 
             Type{MOI.VectorOfVariables}, 
             Type{MOI.VectorAffineFunction{T}}},
    S::Type{<:MOI.AbstractSet}) where T

    return MOI.supports_constraint(model.optimizer, F, S)
end

function MOI.supports_constraint(
    model::ParametricOptimizer,
    ::Type{MOI.ScalarQuadraticFunction{T}},
    S::Type{<:MOI.AbstractSet}) where T
    return MOI.supports_constraint(model.optimizer, MOI.ScalarAffineFunction{T}, S)
end

function MOI.supports(
    model::ParametricOptimizer,
    attr::Union{MOI.ObjectiveSense,
            MOI.ObjectiveFunction{MOI.SingleVariable},
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}},
            MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}
            }) where T
    return MOI.supports(model.optimizer, attr)
end

function MOI.empty!(model::ParametricOptimizer{T}) where T
    MOI.empty!(model.optimizer)
    empty!(model.parameters)
    empty!(model.parameters_name)
    empty!(model.updated_parameters)
    empty!(model.variables)
    model.last_variable_index_added = 0
    model.last_parameter_index_added = PARAMETER_INDEX_THRESHOLD
    empty!(model.affine_constraint_cache)
    empty!(model.quadratic_constraint_cache_pv)
    empty!(model.quadratic_constraint_cache_pp)
    empty!(model.quadratic_constraint_cache_pc)
    empty!(model.quadratic_constraint_variables_associated_to_parameters_cache)
    empty!(model.quadratic_added_cache)
    model.last_quad_add_added = 0
    empty!(model.affine_objective_cache)
    empty!(model.quadratic_objective_cache_pv)
    empty!(model.quadratic_objective_cache_pp)
    empty!(model.quadratic_objective_cache_pc)
    empty!(model.quadratic_objective_variables_associated_to_parameters_cache)
    return
end

function MOI.set(model::ParametricOptimizer, attr::MOI.VariableName, v::MOI.VariableIndex, name::String)
    if haskey(model.parameters, v)
        model.parameters_name[v] = name
    else
        return MOI.set(model.optimizer, attr, v, name)
    end 
end

function MOI.get(model::ParametricOptimizer, attr::MOI.VariableName, v::MOI.VariableIndex)
    if haskey(model.parameters, v)
        return model.parameters_name[v]
    else
        return MOI.get(model.optimizer, attr, v)
    end
end

function MOI.supports(model::ParametricOptimizer, attr::MOI.VariableName, tp::Type{MOI.VariableIndex})
    return MOI.supports(model.optimizer, attr, tp)
end

function MOI.set(model::ParametricOptimizer, attr::MOI.ConstraintName, c::MOI.ConstraintIndex, name::String)
    return MOI.set(model.optimizer, attr, c, name)
end

function MOI.get(model::ParametricOptimizer, attr::MOI.ConstraintName, c::MOI.ConstraintIndex)
    return MOI.get(model.optimizer, attr, c)
end

function MOI.supports(model::ParametricOptimizer, attr::MOI.ConstraintName, tp::Type{<:MOI.ConstraintIndex})
    return MOI.supports(model.optimizer, attr, tp)
end

struct ParameterValue <: MOI.AbstractVariableAttribute end

function MOI.set(model::ParametricOptimizer, ::ParameterValue, vi::MOI.VariableIndex, val)
    cv = MOI.ConstraintIndex{MOI.SingleVariable, POI.Parameter}(vi.value)
    return MOI.set(model, MOI.ConstraintSet(), cv, POI.Parameter(val))
end 

# TODO
# This is not correct, you need to put the parameters back into the function
# function MOI.get(model::ParametricOptimizer, attr::MOI.ConstraintFunction, ci::MOI.ConstraintIndex{F, S}) where {F, S}
#     MOI.get(model.optimizer, attr, ci)
# end

function MOI.get(model::ParametricOptimizer, attr::MOI.ConstraintSet, ci::MOI.ConstraintIndex{F, S}) where {F, S}  
    MOI.throw_if_not_valid(model, ci)
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.get(model::ParametricOptimizer, attr::MOI.ObjectiveSense)
    return MOI.get(model.optimizer, attr)
end

# TODO
# This is not correct, you might have transformed a quadratic function into an affine function,
# you need to give the type that was given by the user, not the type of the inner model.
# function MOI.get(model::ParametricOptimizer, attr::MOI.ObjectiveFunctionType)
#     MOI.get(model.optimizer, attr)
# end

# TODO
# Same as ConstraintFunction getter. And you also need to convert to F
# function MOI.get(
#     model::ParametricOptimizer,
#     attr::MOI.ObjectiveFunction{F}) where F <: Union{MOI.SingleVariable,MOI.ScalarAffineFunction{T}} where T

#     MOI.get(model.optimizer, attr)
# end

# TODO
# You might have transformed quadratic functions into affine functions so this is incorrect
# function MOI.get(model::ParametricOptimizer, ::MOI.ListOfConstraints)
#     constraints = Set{Tuple{DataType, DataType}}()
#     inner_ctrs = MOI.get(model.optimizer, MOI.ListOfConstraints())
#     for (F, S) in inner_ctrs
#         push!(constraints, (F,S))
#     end

#     collect(constraints)
# end

function MOI.get(
    model::ParametricOptimizer,
    attr::MOI.ListOfConstraintIndices{F, S}
) where {S, F<:Union{
    MOI.VectorOfVariables,
    MOI.SingleVariable,
}}
    return MOI.get(model.optimizer, attr)
end

# TODO
# You might have transformed quadratic functions into affine functions so this is incorrect
# function MOI.get(
#     model::ParametricOptimizer,
#     ::MOI.ListOfConstraintIndices{F, S}
# ) where {S<:MOI.AbstractSet, F<:Union{
#     MOI.ScalarAffineFunction{T},
#     MOI.VectorAffineFunction{T},
# }} where T
#     MOI.get(model.optimizer, MOI.ListOfConstraintIndices{F, S}())
# end

function MOI.supports_add_constrained_variable(
    ::ParametricOptimizer, ::Type{Parameter})
    return true
end

function MOI.supports_add_constrained_variables(
    model::ParametricOptimizer, ::Type{MOI.Reals})
    return MOI.supports_add_constrained_variables(model.optimizer, MOI.Reals)
end

function MOI.add_variable(model::ParametricOptimizer)
    next_variable_index!(model)
    v_p = MOI.VariableIndex(model.last_variable_index_added)
    v = MOI.add_variable(model.optimizer)
    model.variables[v_p] = v
    return v_p
end

"""
    MOI.add_constrained_variable(model::ParametricOptimizer, set::Parameter)

Adds a parameter, that is, a variable parameterized to the value provided in `set`, to the `model` specified.
The `model` must be a `ParametricOptInterface.ParametricOptimizer` model.
Returns the MOI.VariableIndex of the parameterized variable and the MOI.ConstraintIndex associated.

#Example:
```julia-repl
julia> w, cw = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
(MathOptInterface.VariableIndex(1), MathOptInterface.ConstraintIndex{MathOptInterface.SingleVariable,ParametricOptInterface.Parameter}(1))
```
"""
function MOI.add_constrained_variable(model::ParametricOptimizer, set::Parameter)
    next_parameter_index!(model)
    p = MOI.VariableIndex(model.last_parameter_index_added)
    model.parameters[p] = set.val
    cp = MOI.ConstraintIndex{MOI.SingleVariable, Parameter}(model.last_parameter_index_added)
    return p, cp
end

function MOI.add_constraint(model::ParametricOptimizer, f::MOI.SingleVariable, set::MOI.AbstractScalarSet)
    if is_parameter_in_model(model, f.variable)
        error("Cannot constrain a parameter")
    elseif is_variable_in_model(model, f.variable)
        return MOI.add_constraint(model.optimizer, f, set)
    else
        error("Variable not in the model")
    end
end

function add_constraint_with_parameters_on_function(model::ParametricOptimizer, f::MOI.ScalarAffineFunction{T}, set::MOI.AbstractScalarSet) where T  
    vars, params = separate_variables_from_parameters(model, f.terms)
    param_constant = calculate_param_constant(model, params)
    ci = MOIU.normalize_and_add_constraint(model.optimizer, MOI.ScalarAffineFunction(vars, f.constant + param_constant), set)     
    model.affine_constraint_cache[ci] = params
    return ci
end

function MOI.add_constraint(model::ParametricOptimizer, f::MOI.ScalarAffineFunction{T}, set::MOI.AbstractScalarSet) where T   
    if !function_has_parameters(model, f)
        return MOI.add_constraint(model.optimizer, f, set)
    else
        return add_constraint_with_parameters_on_function(model, f, set)
    end
end

function update_constant!(s::MOI.LessThan{T}, val) where T
    return MOI.LessThan{T}(s.upper - val)
end

function update_constant!(s::MOI.GreaterThan{T}, val) where T
    return MOI.GreaterThan{T}(s.lower - val)
end

function update_constant!(s::MOI.EqualTo{T}, val) where T
    return MOI.EqualTo{T}(s.value - val)
end

function MOI.get(model::ParametricOptimizer, attr::MOI.VariablePrimal, v::MOI.VariableIndex)
    if is_parameter_in_model(model, v)
         return model.parameters[v]
    elseif is_variable_in_model(model, v)
        return MOI.get(model.optimizer, attr, model.variables[v])
    else
        error("Variable not in the model")
    end
end

"""
    MOI.set(model::ParametricOptimizer, ::MOI.ConstraintSet, cp::MOI.ConstraintIndex{MOI.SingleVariable, Parameter}, set::Parameter)

Sets the parameter to a given value, using its `MOI.ConstraintIndex` as reference.

#Example:
```julia-repl
julia> MOI.set(optimizer, MOI.ConstraintSet(), cw, POI.Parameter(2.0))
2.0
```
"""
function MOI.set(model::ParametricOptimizer, ::MOI.ConstraintSet, cp::MOI.ConstraintIndex{MOI.SingleVariable, Parameter}, set::Parameter)
    p = MOI.VariableIndex(cp.value)
    if haskey(model.parameters, p)
        return model.updated_parameters[p] = set.val
    else
        error("Parameter not in the model")
    end
end

function MOI.set(model::ParametricOptimizer, attr::MOI.ObjectiveFunction, f::MOI.ScalarAffineFunction{T}) where T

    # clear previously defined objetive function cache
    model.affine_objective_cache = Vector{MOI.ScalarAffineTerm{T}}()
    model.quadratic_objective_cache_pv = Vector{MOI.ScalarQuadraticTerm{T}}()
    model.quadratic_objective_cache_pp = Vector{MOI.ScalarQuadraticTerm{T}}()
    model.quadratic_objective_cache_pc = Vector{MOI.ScalarQuadraticTerm{T}}()


    if !function_has_parameters(model, f)
        MOI.set(model.optimizer, attr, f) 
        return
    else
        vars, params = separate_variables_from_parameters(model, f.terms)
        param_constant = calculate_param_constant(model, params)
        MOI.set(model.optimizer, attr, MOI.ScalarAffineFunction(vars, f.constant + param_constant))
        model.affine_objective_cache = params
        return
    end
end

function MOI.set(model::ParametricOptimizer, attr::MOI.ObjectiveFunction, v::MOI.SingleVariable)
    if haskey(model.parameters, v)
        error("Cannot use a parameter as objective function alone")
    elseif haskey(model.variables, v)
        return MOI.set(model.optimizer, attr, MOI.SingleVariable(model.variables[v.variable])) 
    else
        error("Variable not in the model")
    end
end

function MOI.set(model::ParametricOptimizer, attr::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    return MOI.set(model.optimizer, attr, sense)
end

function MOI.get(model::ParametricOptimizer, attr::T) where {
                                                                T <: Union{
                                                                    MOI.TerminationStatus,
                                                                    MOI.ObjectiveValue,
                                                                    MOI.PrimalStatus
                                                                    }
                                                            }
    return MOI.get(model.optimizer, attr)
end

function MOI.set(model::ParametricOptimizer, ::MOI.Silent, bool::Bool)
    MOI.set(model.optimizer, MOI.Silent(), bool)
end

function MOI.get(optimizer::ParametricOptimizer, ::MOI.SolverName)
    return "ParametricOptimizer with " *
           MOI.get(optimizer.optimizer, MOI.SolverName()) *
           " attached"
end

function MOI.add_constraint(model::ParametricOptimizer, f::MOI.VectorOfVariables, set::MOI.AbstractVectorSet) where T   
    if function_has_parameters(model, f)
        error("VectorOfVariables does not allow parameters")
    else
        return MOI.add_constraint(model.optimizer, f, set)
    end
end

function MOI.add_constraint(model::ParametricOptimizer, f::MOI.VectorAffineFunction{T}, set::MOI.AbstractVectorSet) where T   
    if function_has_parameters(model, f)
        error("VectorAffineFunction does not allow parameters")
    else
        return MOI.add_constraint(model.optimizer, f, set)
    end
end

function add_constraint_with_parameters_on_function(model::ParametricOptimizer, f::MOI.ScalarQuadraticFunction{T}, set::MOI.AbstractScalarSet) where T

    quad_params = MOI.ScalarQuadraticTerm{T}[] #outside declaration so it has default value
    quad_aff_vars = MOI.ScalarQuadraticTerm{T}[] #outside declaration so it has default value; parameter as variable_index_1
    aux_variables_associated_to_parameters =  MOI.VariableIndex[] #outside declaration so it has default value

    if function_quadratic_terms_has_parameters(model, f.quadratic_terms)
        
        quad_terms = MOI.ScalarQuadraticTerm{T}[]
        
        for term in f.quadratic_terms
            if is_variable_in_model(model, term.variable_index_1) && is_variable_in_model(model, term.variable_index_2)
                push!(quad_terms, term) # if there are only variables, it remains a quadratic term

            elseif is_parameter_in_model(model, term.variable_index_1) && is_variable_in_model(model, term.variable_index_2)
                # This is the case when term.variable_index_1 is a parameter and term.variable_index_2 is a variable.
                # Thus, it creates an affine term. Convention: param as 1, var as 2
                aux = MOI.ScalarQuadraticTerm(term.coefficient, term.variable_index_1, term.variable_index_2)
                push!(quad_aff_vars, aux)
                push!(aux_variables_associated_to_parameters, term.variable_index_2)  

            elseif is_variable_in_model(model, term.variable_index_1) && is_parameter_in_model(model, term.variable_index_2)
                # Check convention defined above
                aux = MOI.ScalarQuadraticTerm(term.coefficient, term.variable_index_2, term.variable_index_1)
                push!(quad_aff_vars, aux)
                push!(aux_variables_associated_to_parameters, term.variable_index_1)          

            elseif is_parameter_in_model(model, term.variable_index_1) && is_parameter_in_model(model, term.variable_index_2)
                # This is the case where both variable_index_1,2 are actually parameters
                aux = MOI.ScalarQuadraticTerm(term.coefficient, term.variable_index_1, term.variable_index_2)
                push!(quad_params, aux)

            else
                error("Constraint uses a variable that is not in the model")
            end
        end

    else
        quad_terms = f.quadratic_terms
    end


    aff_params = MOI.ScalarAffineTerm{T}[] #outside declaration so it has default value
    variables_associated_to_parameters =  MOI.ScalarAffineTerm{T}[] #outside declaration so it has default value

    if function_affine_terms_has_parameters(model, f.affine_terms)

        aff_vars = MOI.ScalarAffineTerm{T}[]

        for term in f.affine_terms
            if is_variable_in_model(model, term.variable_index)
                push!(aff_vars, term)
                if term.variable_index in aux_variables_associated_to_parameters
                    push!(variables_associated_to_parameters, term)
                end
            elseif is_parameter_in_model(model, term.variable_index)
                push!(aff_params, term)
            else
                error("Constraint uses a variable that is not in the model")
            end
        end

    else
        aff_vars = f.affine_terms
    end
    
    aff_terms = MOI.ScalarAffineTerm{T}[]

    for term in quad_aff_vars
        push!(
            aff_terms, 
            MOI.ScalarAffineTerm(term.coefficient * model.parameters[term.variable_index_1], term.variable_index_2)
        )
    end

    aff_terms = vcat(aff_terms, aff_vars)

    const_term = f.constant

    for j in aff_params
        const_term += j.coefficient * model.parameters[j.variable_index]
    end

    for j in quad_params
        const_term += j.coefficient * model.parameters[j.variable_index_1] * model.parameters[j.variable_index_2]
    end

    f_quad = if !isempty(quad_terms)
        MOI.ScalarQuadraticFunction(
                    aff_terms,
                    quad_terms,
                    const_term 
                )
    else
        MOI.ScalarAffineFunction(
                    aff_terms,
                    const_term
                )
    end
    
    
    model.last_quad_add_added += 1
    ci = MOIU.normalize_and_add_constraint(model.optimizer, f_quad, set)
    new_ci = MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{Float64}, typeof(set)}(model.last_quad_add_added)
    model.quadratic_added_cache[new_ci] = ci

    if !isempty(quad_aff_vars)
        model.quadratic_constraint_cache_pv[new_ci] = quad_aff_vars
    end

    if !isempty(quad_params)
        model.quadratic_constraint_cache_pp[new_ci] = quad_params
    end

    if !isempty(aff_params)
        model.quadratic_constraint_cache_pc[new_ci] = aff_params
    end

    if !isempty(variables_associated_to_parameters)
        model.quadratic_constraint_variables_associated_to_parameters_cache[new_ci] = variables_associated_to_parameters
    end

    return ci
end

function MOI.add_constraint(model::ParametricOptimizer, f::MOI.ScalarQuadraticFunction{T}, set::MOI.AbstractScalarSet) where T
    if !function_has_parameters(model, f)
        return MOI.add_constraint(model.optimizer, f, set) 
    else
        return add_constraint_with_parameters_on_function(model, f, set)
    end
end

function MOI.set(model::ParametricOptimizer, attr::MOI.ObjectiveFunction, f::MOI.ScalarQuadraticFunction{T}) where T


    # clear previously defined objetive function cache
    model.affine_objective_cache = Array{MOI.ScalarAffineTerm{T},1}()
    model.quadratic_objective_cache_pv = Array{MOI.ScalarQuadraticTerm{T},1}()
    model.quadratic_objective_cache_pp = Array{MOI.ScalarQuadraticTerm{T},1}()
    model.quadratic_objective_cache_pc = Array{MOI.ScalarQuadraticTerm{T},1}()


    if !function_has_parameters(model, f)
        MOI.set(model.optimizer, attr, f) 
        return
    else
                
        quad_params = MOI.ScalarQuadraticTerm{T}[] #outside declaration so it has default value
        quad_aff_vars = MOI.ScalarQuadraticTerm{T}[] #outside declaration so it has default value; parameter as variable_index_1
        aux_variables_associated_to_parameters =  MOI.VariableIndex[] #outside declaration so it has default value

        if function_quadratic_terms_has_parameters(model, f.quadratic_terms)
            
            quad_terms = MOI.ScalarQuadraticTerm{T}[]  

            for term in f.quadratic_terms
                if haskey(model.variables, term.variable_index_1) && haskey(model.variables, term.variable_index_2)
                    push!(quad_terms, term) # if there are only variables, it remains a quadratic term

                elseif haskey(model.parameters, term.variable_index_1) && haskey(model.variables, term.variable_index_2)
                    # This is the case when term.variable_index_1 is a parameter and term.variable_index_2 is a variable.
                    # Thus, it creates an affine term. Convention: param as 1, var as 2
                    push!(quad_aff_vars, MOI.ScalarQuadraticTerm(term.coefficient, term.variable_index_1, term.variable_index_2))
                    push!(aux_variables_associated_to_parameters, term.variable_index_2)

                elseif haskey(model.variables, term.variable_index_1) && haskey(model.parameters, term.variable_index_2)
                    # Check convention defined above
                    push!(quad_aff_vars, MOI.ScalarQuadraticTerm(term.coefficient, term.variable_index_2, term.variable_index_1))
                    push!(aux_variables_associated_to_parameters, term.variable_index_1)

                elseif haskey(model.parameters, term.variable_index_1) && haskey(model.parameters, term.variable_index_2)
                    # This is the case where both variable_index_1,2 are actually parameters
                    push!(quad_params, MOI.ScalarQuadraticTerm(term.coefficient, term.variable_index_1, term.variable_index_2))

                else
                    error("Constraint uses a variable that is not in the model")
                end
            end

        else
            quad_terms = f.quadratic_terms
        end


        aff_params = MOI.ScalarAffineTerm{T}[] #outside declaration so it has default value
        variables_associated_to_parameters =  MOI.ScalarAffineTerm{T}[] #outside declaration so it has default value

        if function_affine_terms_has_parameters(model, f.affine_terms)

            aff_vars = MOI.ScalarAffineTerm{T}[]

            for term in f.affine_terms
                if haskey(model.variables, term.variable_index)
                    push!(aff_vars, term)
                    if term.variable_index in aux_variables_associated_to_parameters
                        push!(variables_associated_to_parameters, term)
                    end
                elseif haskey(model.parameters, term.variable_index)
                    push!(aff_params, term)
                else
                    error("Constraint uses a variable that is not in the model")
                end
            end

        else
            aff_vars = f.affine_terms
        end

        aff_terms = MOI.ScalarAffineTerm{T}[]

        for variable in quad_aff_vars
            push!(aff_terms, MOI.ScalarAffineTerm(variable.coefficient * model.parameters[variable.variable_index_1], variable.variable_index_2))
        end

        aff_terms = vcat(aff_terms, aff_vars)

        const_term = f.constant

        for param in aff_params
            const_term += param.coefficient * model.parameters[param.variable_index]
        end

        for param in quad_params
            const_term += param.coefficient * model.parameters[param.variable_index_1] * model.parameters[param.variable_index_2]
        end

        if !isempty(quad_terms)
            f_quad = MOI.ScalarQuadraticFunction(
                        aff_terms,
                        quad_terms,
                        const_term 
                    )

            MOI.set(model.optimizer, attr, f_quad)
        else
           f_quad = MOI.ScalarAffineFunction(
                        aff_terms,
                        const_term
                    )

            MOI.set(model.optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), f_quad)
        end

        model.quadratic_objective_cache_pv = quad_aff_vars
        model.quadratic_objective_cache_pp = quad_params
        model.quadratic_objective_cache_pc = aff_params
        model.quadratic_objective_variables_associated_to_parameters_cache = variables_associated_to_parameters

        return
    end
end

function MOI.optimize!(model::ParametricOptimizer)
    if !isempty(model.updated_parameters)

        for (ci, fparam) in model.affine_constraint_cache
            param_constant = 0
            for j in fparam
                if haskey(model.updated_parameters, j.variable_index)
                    param_old = model.parameters[j.variable_index]
                    param_new = model.updated_parameters[j.variable_index]
                    aux = param_new - param_old
                    param_constant += j.coefficient * aux
                end
            end
            if param_constant != 0
                set = MOI.get(model.optimizer, MOI.ConstraintSet(), ci)
                set = POI.update_constant!(set, param_constant)
                MOI.set(model.optimizer, MOI.ConstraintSet(), ci, set)
            end
        end

        if !isempty(model.affine_objective_cache)
            objective_constant = 0
            for j in model.affine_objective_cache
                if haskey(model.updated_parameters, j.variable_index)
                    param_old = model.parameters[j.variable_index]
                    param_new = model.updated_parameters[j.variable_index]
                    aux = param_new - param_old
                    objective_constant += j.coefficient * aux
                end
            end
            if objective_constant != 0
                F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
                f = MOI.get(model.optimizer, MOI.ObjectiveFunction{F}())
                fvar = MOI.ScalarAffineFunction(f.terms, f.constant + objective_constant)
                MOI.set(model.optimizer,MOI.ObjectiveFunction{F}(), fvar)       
            end
        end

        for (ci, fparam) in model.quadratic_constraint_cache_pc
            param_constant = 0
            for j in fparam
                if haskey(model.updated_parameters, j.variable_index)
                    param_old = model.parameters[j.variable_index]
                    param_new = model.updated_parameters[j.variable_index]
                    aux = param_new - param_old
                    param_constant += j.coefficient * aux
                end
            end
            if param_constant != 0
                set = MOI.get(model.optimizer, MOI.ConstraintSet(), model.quadratic_added_cache[ci])
                set = POI.update_constant!(set, param_constant)
                MOI.set(model.optimizer, MOI.ConstraintSet(), model.quadratic_added_cache[ci], set)
            end
        end

        if !isempty(model.quadratic_objective_cache_pc)
            objective_constant = 0
            for j in model.quadratic_objective_cache_pc
                if haskey(model.updated_parameters, j.variable_index)
                    param_old = model.parameters[j.variable_index]
                    param_new = model.updated_parameters[j.variable_index]
                    aux = param_new - param_old
                    objective_constant += j.coefficient * aux
                end
            end
            if objective_constant != 0
                F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
                f = MOI.get(model.optimizer, MOI.ObjectiveFunction{F}())
                fvar = if F == MathOptInterface.ScalarAffineFunction{Float64}
                    MOI.ScalarAffineFunction(f.terms, f.constant + objective_constant)
                else
                    MOI.ScalarQuadraticFunction(f.affine_terms, f.quadratic_terms, f.constant + objective_constant)                
                end
                MOI.set(model.optimizer,MOI.ObjectiveFunction{F}(), fvar)
            end
        end

        for (ci, fparam) in model.quadratic_constraint_cache_pp 
            param_constant = 0
            for j in fparam
                if haskey(model.updated_parameters, j.variable_index_1) && haskey(model.updated_parameters, j.variable_index_2)
                    param_new_1 = model.updated_parameters[j.variable_index_1]
                    param_new_2 = model.updated_parameters[j.variable_index_2]
                    param_old_1 = model.parameters[j.variable_index_1]
                    param_old_2 = model.parameters[j.variable_index_2]
                    param_constant += j.coefficient * ((param_new_1 * param_new_2)-(param_old_1 * param_old_2))
                elseif haskey(model.updated_parameters, j.variable_index_1)
                    param_new_1 = model.updated_parameters[j.variable_index_1]
                    param_old_1 = model.parameters[j.variable_index_1]
                    param_old_2 = model.parameters[j.variable_index_2]
                    param_constant += j.coefficient * param_old_2 * (param_new_1 - param_old_1)
                elseif haskey(model.updated_parameters, j.variable_index_2) 
                    param_new_2 = model.updated_parameters[j.variable_index_2]
                    param_old_1 = model.parameters[j.variable_index_1]
                    param_old_2 = model.parameters[j.variable_index_2]
                    param_constant += j.coefficient * param_old_1 * (param_new_2 - param_old_2)
                end
            end
            if param_constant != 0
                set = MOI.get(model.optimizer, MOI.ConstraintSet(), model.quadratic_added_cache[ci])
                set = POI.update_constant!(set, param_constant)
                MOI.set(model.optimizer, MOI.ConstraintSet(), model.quadratic_added_cache[ci], set)
            end
        end

        if !isempty(model.quadratic_objective_cache_pp)
            objective_constant = 0
            for j in model.quadratic_objective_cache_pp
                if haskey(model.updated_parameters, j.variable_index_1) && haskey(model.updated_parameters, j.variable_index_2)
                    param_new_1 = model.updated_parameters[j.variable_index_1]
                    param_new_2 = model.updated_parameters[j.variable_index_2]
                    param_old_1 = model.parameters[j.variable_index_1]
                    param_old_2 = model.parameters[j.variable_index_2]
                    objective_constant += j.coefficient * ((param_new_1 * param_new_2)-(param_old_1 * param_old_2))
                elseif haskey(model.updated_parameters, j.variable_index_1)
                    param_new_1 = model.updated_parameters[j.variable_index_1]
                    param_old_1 = model.parameters[j.variable_index_1]
                    param_old_2 = model.parameters[j.variable_index_2]
                    objective_constant += j.coefficient * param_old_2 * (param_new_1 - param_old_1)
                elseif haskey(model.updated_parameters, j.variable_index_2) 
                    param_new_2 = model.updated_parameters[j.variable_index_2]
                    param_old_1 = model.parameters[j.variable_index_1]
                    param_old_2 = model.parameters[j.variable_index_2]
                    objective_constant += j.coefficient * param_old_1 * (param_new_2 - param_old_2)
                end
            end
            if objective_constant != 0
                F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
                f = MOI.get(model.optimizer, MOI.ObjectiveFunction{F}())
                fvar = if F == MathOptInterface.ScalarAffineFunction{Float64}
                    MOI.ScalarAffineFunction(f.terms, f.constant + objective_constant)
                else
                    MOI.ScalarQuadraticFunction(f.affine_terms, f.quadratic_terms, f.constant + objective_constant)
                end
                MOI.set(model.optimizer,MOI.ObjectiveFunction{F}(), fvar)
            end
        end

        constraint_aux_dict = Dict{Any,Any}()

        for (ci, fparam) in model.quadratic_constraint_cache_pv
            for j in fparam
                if haskey(model.updated_parameters, j.variable_index_1)
                    coef = j.coefficient
                    param_new = model.updated_parameters[j.variable_index_1]
                    if haskey(constraint_aux_dict, (ci, j.variable_index_2))
                        constraint_aux_dict[(ci, j.variable_index_2)] += param_new*coef
                    else
                        constraint_aux_dict[(ci, j.variable_index_2)] = param_new*coef
                    end
                end
            end
        end

        for (ci, fparam) in model.quadratic_constraint_variables_associated_to_parameters_cache
            for j in fparam
                coef = j.coefficient
                if haskey(constraint_aux_dict, (ci, j.variable_index))#
                    constraint_aux_dict[(ci, j.variable_index)] += coef
                else
                    constraint_aux_dict[(ci, j.variable_index)] = coef
                end
            end
        end

        for ((ci, vi), value) in constraint_aux_dict
            old_ci = model.quadratic_added_cache[ci]
            MOI.modify(model.optimizer, old_ci, MOI.ScalarCoefficientChange(vi, value))
        end

        objective_aux_dict = Dict{Any,Any}()

        if !isempty(model.quadratic_objective_cache_pv)
            for j in model.quadratic_objective_cache_pv
                if haskey(model.updated_parameters, j.variable_index_1)
                    coef = j.coefficient
                    param_new = model.updated_parameters[j.variable_index_1]
                    if haskey(objective_aux_dict, (j.variable_index_2))
                        objective_aux_dict[(j.variable_index_2)] += param_new*coef
                    else
                        objective_aux_dict[(j.variable_index_2)] = param_new*coef
                    end
                end
            end
        end

        for j in model.quadratic_objective_variables_associated_to_parameters_cache
            coef = j.coefficient
            if haskey(objective_aux_dict, j.variable_index)
                objective_aux_dict[j.variable_index] += coef
            else
                objective_aux_dict[j.variable_index] = coef
            end
        end

        F_pv = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())

        for (key, value) in objective_aux_dict
            MOI.modify(model.optimizer, MOI.ObjectiveFunction{F_pv}(), MOI.ScalarCoefficientChange(key, value))
        end

        for (i, val) in model.updated_parameters
            model.parameters[i] = val
        end
        empty!(model.updated_parameters)
    end
    MOI.optimize!(model.optimizer)
end

end # module

