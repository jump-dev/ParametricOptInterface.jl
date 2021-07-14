module ParametricOptInterface

using MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const DD = MOIU.DoubleDicts
const POI = ParametricOptInterface
const PARAMETER_INDEX_THRESHOLD = 1_000_000_000_000_000_000
const ScalarLinearSetList = [MOI.EqualTo, MOI.GreaterThan, MOI.LessThan]

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
    affine_constraint_cache::DD.DoubleDict{Vector{MOI.ScalarAffineTerm{Float64}}}
    quadratic_constraint_cache_pv::Dict{MOI.ConstraintIndex, Vector{MOI.ScalarQuadraticTerm{Float64}}}
    quadratic_constraint_cache_pp::Dict{MOI.ConstraintIndex, Vector{MOI.ScalarQuadraticTerm{Float64}}}
    quadratic_constraint_cache_pc::Dict{MOI.ConstraintIndex, Vector{MOI.ScalarAffineTerm{Float64}}}
    quadratic_constraint_variables_associated_to_parameters_cache::Dict{MOI.ConstraintIndex, Vector{MOI.ScalarAffineTerm{T}}} 
    quadratic_added_cache::Dict{MOI.ConstraintIndex, MOI.ConstraintIndex} 
    last_quad_add_added::Int
    affine_objective_cache::Vector{MOI.ScalarAffineTerm{T}}
    quadratic_objective_cache_pv::Vector{MOI.ScalarQuadraticTerm{T}}
    quadratic_objective_cache_pp::Vector{MOI.ScalarQuadraticTerm{T}}
    quadratic_objective_cache_pc::Vector{MOI.ScalarAffineTerm{T}}
    quadratic_objective_variables_associated_to_parameters_cache::Vector{MOI.ScalarAffineTerm{T}}
    multiplicative_parameters::BitSet
    dual_value_of_parameters::Dict{MOI.VariableIndex, Float64}
    evaluate_duals::Bool
    function ParametricOptimizer(optimizer::OT; evaluate_duals::Bool=true) where OT
        new{Float64, OT}(
            optimizer,
            Dict{MOI.VariableIndex, Float64}(),
            Dict{MOI.VariableIndex, String}(),
            Dict{MOI.VariableIndex, Float64}(),
            Dict{MOI.VariableIndex, MOI.VariableIndex}(),
            0,
            PARAMETER_INDEX_THRESHOLD,
            DD.DoubleDict{Vector{MOI.ScalarAffineTerm{Float64}}}(),
            Dict{MOI.ConstraintIndex, Vector{MOI.ScalarQuadraticTerm{Float64}}}(),
            Dict{MOI.ConstraintIndex, Vector{MOI.ScalarQuadraticTerm{Float64}}}(),
            Dict{MOI.ConstraintIndex, Vector{MOI.ScalarAffineTerm{Float64}}}(),
            Dict{MOI.ConstraintIndex, Vector{MOI.ScalarAffineTerm{Float64}}}(),
            Dict{MOI.ConstraintIndex, MOI.ConstraintIndex}(),
            0,
            Vector{MOI.ScalarAffineTerm{Float64}}(),
            Vector{MOI.ScalarQuadraticTerm{Float64}}(),
            Vector{MOI.ScalarQuadraticTerm{Float64}}(),
            Vector{MOI.ScalarAffineTerm{Float64}}(),
            Vector{MOI.ScalarAffineTerm{Float64}}(),
            BitSet(),
            Dict{MOI.VariableIndex, Float64}(),
            evaluate_duals
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
    isempty(model.quadratic_objective_variables_associated_to_parameters_cache) &&
    isempty(model.dual_value_of_parameters)
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
    empty!(model.dual_value_of_parameters)
    return
end

function MOI.set(model::ParametricOptimizer, attr::MOI.VariableName, v::MOI.VariableIndex, name::String)
    if is_parameter_in_model(model, v)
        model.parameters_name[v] = name
    else
        return MOI.set(model.optimizer, attr, v, name)
    end 
end

function MOI.get(model::ParametricOptimizer, attr::MOI.VariableName, v::MOI.VariableIndex)
    if is_parameter_in_model(model, v)
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
    vars, params, param_constant = separate_possible_terms_and_calculate_parameter_constant(model, f.terms)
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
    if is_parameter_in_model(model,p)
        return model.updated_parameters[p] = set.val
    else
        error("Parameter not in the model")
    end
end

struct ParameterValue <: MOI.AbstractVariableAttribute end

"""
    MOI.set(model::ParametricOptimizer, ::MOI.ConstraintSet, cp::MOI.ConstraintIndex{MOI.SingleVariable, Parameter}, set::Parameter)

Sets the parameter to a given value, using its `MOI.ConstraintIndex` as reference.

#Example:
```julia-repl
julia> MOI.set(model, ParameterValue(), w, 2.0)
2.0
```
"""
function MOI.set(model::ParametricOptimizer, ::ParameterValue, vi::MOI.VariableIndex, val::Float64)
    if is_parameter_in_model(model, vi)
        return model.updated_parameters[vi] = val
    else
        error("Parameter not in the model")
    end
end 
function MOI.set(model::ParametricOptimizer, ::ParameterValue, vi::MOI.VariableIndex, val::Real)
    return MOI.set(model, ParameterValue(), vi, convert(Float64, val))
end

function empty_objective_function_caches!(model:: ParametricOptimizer)
    empty!(model.affine_objective_cache)
    empty!(model.quadratic_objective_cache_pv)
    empty!(model.quadratic_objective_cache_pp)
    empty!(model.quadratic_objective_cache_pc)
    return nothing
end

function MOI.set(model::ParametricOptimizer, attr::MOI.ObjectiveFunction, f::MOI.ScalarAffineFunction{T}) where T

    # clear previously defined objetive function cache
    empty_objective_function_caches!(model)
    
    if !function_has_parameters(model, f)
        MOI.set(model.optimizer, attr, f) 
        return
    else
        vars, params, param_constant = separate_possible_terms_and_calculate_parameter_constant(model, f.terms)
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

function add_constraint_with_parameters_on_function(
                            model::ParametricOptimizer, 
                            f::MOI.ScalarQuadraticFunction{T}, 
                            set::S) where {T, S <: MOI.AbstractScalarSet}

    if function_quadratic_terms_has_parameters(model, f.quadratic_terms)
        (quad_vars, 
         quad_aff_vars, 
         quad_params,
         aff_terms,
         variables_associated_to_parameters,
         quad_param_constant) = separate_possible_terms_and_calculate_parameter_constant(model, f.quadratic_terms)
    else
        quad_vars = f.quadratic_terms
        quad_aff_vars = MOI.ScalarQuadraticTerm{T}[]
        quad_params = MOI.ScalarQuadraticTerm{T}[]
        aff_terms = MOI.ScalarAffineTerm{T}[]
        quad_param_constant = zero(T)
        variables_associated_to_parameters = MOI.VariableIndex[]
    end

    if function_affine_terms_has_parameters(model, f.affine_terms)
        (aff_vars, 
         aff_params, 
         terms_with_variables_associated_to_parameters,
         aff_param_constant) = separate_possible_terms_and_calculate_parameter_constant(
                                                                                model, 
                                                                                f.affine_terms, 
                                                                                variables_associated_to_parameters
                                                                            )
    else
        aff_vars = f.affine_terms
        aff_params = MOI.ScalarAffineTerm{T}[]
        terms_with_variables_associated_to_parameters = MOI.ScalarAffineTerm{T}[]
        aff_param_constant = zero(T)
    end

    aff_terms = vcat(aff_terms, aff_vars)
    const_term = f.constant + aff_param_constant + quad_param_constant
    quad_terms = quad_vars

    f_quad = if !isempty(quad_vars)
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
    # This part is used to remember that ci came from a quadratic function
    # It is particularly useful because sometimes
    new_ci = MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T}, S}(model.last_quad_add_added)
    model.quadratic_added_cache[new_ci] = ci

    fill_quadratic_constraint_caches!(
        model, 
        new_ci,
        quad_aff_vars,
        quad_params,
        aff_params,
        terms_with_variables_associated_to_parameters
    )

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
    empty_objective_function_caches!(model)

    if !function_has_parameters(model, f)
        MOI.set(model.optimizer, attr, f) 
        return
    else
        if function_quadratic_terms_has_parameters(model, f.quadratic_terms)
            (quad_vars, 
             quad_aff_vars, 
             quad_params,
             aff_terms,
             variables_associated_to_parameters,
             quad_param_constant) = separate_possible_terms_and_calculate_parameter_constant(model, f.quadratic_terms)
        else
            quad_vars = f.quadratic_terms
            quad_aff_vars = MOI.ScalarQuadraticTerm{T}[]
            quad_params = MOI.ScalarQuadraticTerm{T}[]
            aff_terms = MOI.ScalarAffineTerm{T}[]
            quad_param_constant = zero(T)
            variables_associated_to_parameters = MOI.VariableIndex[]
        end
    
        if function_affine_terms_has_parameters(model, f.affine_terms)
            (aff_vars, 
             aff_params, 
             terms_with_variables_associated_to_parameters,
             aff_param_constant) = separate_possible_terms_and_calculate_parameter_constant(
                                                                                    model, 
                                                                                    f.affine_terms, 
                                                                                    variables_associated_to_parameters
                                                                                )
        else
            aff_vars = f.affine_terms
            aff_params = MOI.ScalarAffineTerm{T}[]
            terms_with_variables_associated_to_parameters = MOI.ScalarAffineTerm{T}[]
            aff_param_constant = zero(T)
        end
    
        aff_terms = vcat(aff_terms, aff_vars)
        const_term = f.constant + aff_param_constant + quad_param_constant
        quad_terms = quad_vars
    
        if !isempty(quad_vars)
            MOI.set(
                model.optimizer, 
                attr, 
                MOI.ScalarQuadraticFunction(aff_terms, quad_terms, const_term)
            )
        else
            MOI.set(
                model.optimizer, 
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(), 
                MOI.ScalarAffineFunction(aff_terms, const_term)
            )
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

            MOI.set(model.optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(), f_quad)
        end

        model.quadratic_objective_cache_pv = quad_aff_vars
        model.quadratic_objective_cache_pp = quad_params
        model.quadratic_objective_cache_pc = aff_params
        model.quadratic_objective_variables_associated_to_parameters_cache = terms_with_variables_associated_to_parameters

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

                # TODO
                # Is there another way to verify the Type of F without expliciting {Float64}?
                # Something like isa(F, MOI.ScalarAffineFunction)
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

                # TODO
                # Is there another way to verify the Type of F without expliciting {Float64}?
                # Something like isa(F, MOI.ScalarAffineFunction)
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

    if model.evaluate_duals
        calculate_dual_of_parameters(model)
    end
end

### Duals

function calculate_dual_of_parameters(model::ParametricOptimizer) 
    param_dual_cum_sum = Dict{Int, Float64}()

    for vi in keys(model.parameters)
        param_dual_cum_sum[vi.value] = 0.0
    end
    
    # TODO
    # Quadratic constraints
    for S in ScalarLinearSetList
        affine_constraint_cache_inner = model.affine_constraint_cache[MOI.ScalarAffineFunction{Float64}, S{Float64}]

        if length(affine_constraint_cache_inner) > 0
            param_dual_cum_sum = iterate_over_constraint_cache_affine(model, affine_constraint_cache_inner ,param_dual_cum_sum)
        end

        # for F in [MOI.ScalarQuadraticFunction]
        #     affine_constraint_cache_inner = model.affine_constraint_cache[F{Float64}, S{Float64}]

        #     if length(affine_constraint_cache_inner) > 0
        #         param_dual_cum_sum = iterate_over_constraint_cache_quadratic(model, affine_constraint_cache_inner ,param_dual_cum_sum)
        #     end
        # end
    end

    for param in model.affine_objective_cache
        vi = param.variable_index.value
        param_dual_cum_sum[vi] += param.coefficient
    end

    empty!(model.dual_value_of_parameters)
    for (vi_val, param_dual) in param_dual_cum_sum
        model.dual_value_of_parameters[MOI.VariableIndex(vi_val)] = param_dual
    end
end

function iterate_over_constraint_cache_affine(model::POI.ParametricOptimizer, constraint_cache_inner::DD.WithType{F, S}, param_dual_cum_sum::Dict{Int, Float64}) where {F,S}
    for (ci, param_array) in constraint_cache_inner
        param_dual_cum_sum = calculate_parameters_in_ci(model.optimizer, param_array, ci, param_dual_cum_sum)
    end

    return param_dual_cum_sum
end

# TODO
# Quadratic constraints
function iterate_over_constraint_cache_quadratic(model::POI.ParametricOptimizer, constraint_cache_inner::DD.WithType{F, S}, param_dual_cum_sum::Dict{Int, Float64}) where {F,S}
    for (poi_ci, param_array) in model.quadratic_constraint_cache_pc
        moi_ci = model.quadratic_added_cache[poi_ci]
        for param in param_array
            if cp.value == param.variable_index.value
                cons_dual = MOI.get(model.optimizer, MOI.ConstraintDual(), moi_ci)
                param_dual_quadratic_constraint_affine_part += cons_dual*param.coefficient
            end
        end
    end
    
    for (ci, param_array) in constraint_cache_inner
        param_dual_cum_sum = calculate_parameters_in_ci(model.optimizer, param_array, ci, param_dual_cum_sum)
    end

    return param_dual_cum_sum
end

function calculate_parameters_in_ci(optimizer::OP, param_array::Vector{MOI.ScalarAffineTerm{T}}, ci::CI, param_dual_cum_sum::Dict{Int, Float64}) where {OP, CI, T}
    cons_dual = MOI.get(optimizer, MOI.ConstraintDual(), ci)

    for param in param_array
        vi = param.variable_index.value
        param_dual_cum_sum[vi] += cons_dual*param.coefficient
    end

    return param_dual_cum_sum
end

function MOI.get(model::ParametricOptimizer, ::MOI.ConstraintDual, cp::MOI.ConstraintIndex{MOI.SingleVariable,POI.Parameter})
    if !is_additive(model, cp)
        error("Cannot calculate the dual of a multiplicative parameter")
    end
    return model.dual_value_of_parameters[MOI.VariableIndex(cp.value)]
end

function MOI.get(model::ParametricOptimizer, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex)
    return MOI.get(model.optimizer, attr, ci)
end

function is_additive(model::ParametricOptimizer, cp::MOI.ConstraintIndex)
    if cp.value in model.multiplicative_parameters
        return false
    end
    return true
end

function parameter_dual_in_affine_constraint(model::POI.ParametricOptimizer, cp::MOI.ConstraintIndex)::Float64
    param_dual_affine_constraint = update_parameter_dual(
                                    model.optimizer, 
                                    model.affine_constraint_cache[MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}], 
                                    cp.value
                                )
    return param_dual_affine_constraint
end

function update_parameter_dual(optimizer::OP, affine_constraint_cache_inner::DD.WithType{F, S}, value::Int) where {OP, F, S}
    param_dual_affine_constraint = zero(Float64)
    for (ci::MOI.ConstraintIndex{F, S}, param_array) in affine_constraint_cache_inner
        param_dual_affine_constraint += calculate_parameter_in_ci(optimizer, param_array, ci, value)::Float64
    end
    return param_dual_affine_constraint
end

function calculate_parameter_in_ci(optimizer::OP, param_array::Vector{MOI.ScalarAffineTerm{T}}, ci::CI, value::Int) where {OP, CI, T}
    param_dual_affine_constraint = zero(T)
    for param in param_array
        if value == param.variable_index.value
            cons_dual = MOI.get(optimizer, MOI.ConstraintDual(), ci)
            param_dual_affine_constraint += cons_dual*param.coefficient
        end
    end
    return param_dual_affine_constraint
end

function parameter_dual_in_affine_objective(model::POI.ParametricOptimizer, cp::MOI.ConstraintIndex)
    param_dual_affine_objective = 0
        for param in model.affine_objective_cache
            if cp.value == param.variable_index.value
                param_dual_affine_objective += param.coefficient
            end
        end
    return param_dual_affine_objective
end

function parameter_dual_in_quadratic_constraint_affine_part(model::POI.ParametricOptimizer, cp::MOI.ConstraintIndex)
    param_dual_quadratic_constraint_affine_part = 0
    for (poi_ci, param_array) in model.quadratic_constraint_cache_pc
        moi_ci = model.quadratic_added_cache[poi_ci]
        for param in param_array
            if cp.value == param.variable_index.value
                cons_dual = MOI.get(model.optimizer, MOI.ConstraintDual(), moi_ci)
                param_dual_quadratic_constraint_affine_part += cons_dual*param.coefficient
            end
        end
    end
    return param_dual_quadratic_constraint_affine_part
end

function parameter_dual_in_quadratic_objective_affine_part(model::POI.ParametricOptimizer, cp::MOI.ConstraintIndex)
    param_dual_quadratic_objective_affine_part = 0
        for param in model.quadratic_objective_cache_pc
            if cp.value == param.variable_index.value
                param_dual_quadratic_objective_affine_part += param.coefficient
            end
        end
    return param_dual_quadratic_objective_affine_part
end
    
end # module


