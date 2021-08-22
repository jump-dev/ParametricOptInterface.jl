module ParametricOptInterface

using MathOptInterface

const MOI = MathOptInterface

const PARAMETER_INDEX_THRESHOLD = 1_000_000_000_000_000_000
const SUPPORTED_SETS =
    (MOI.LessThan{Float64}, MOI.EqualTo{Float64}, MOI.GreaterThan{Float64})
const SUPPORTED_VECTOR_SETS = (MOI.Nonnegatives, MOI.SecondOrderCone)

"""
    Parameter(::Float64)

The `Parameter` structure stores the numerical value associated to a given
parameter.

## Example

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

Declares a `ParametricOptimizer`, which allows the handling of parameters in a
optimization model.

## Example

```julia-repl
julia> ParametricOptInterface.ParametricOptimizer(GLPK.Optimizer())
ParametricOptInterface.ParametricOptimizer{Float64,GLPK.Optimizer}
```
"""
mutable struct ParametricOptimizer{T,OT<:MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT
    parameters::Dict{MOI.VariableIndex,T}
    parameters_name::Dict{MOI.VariableIndex,String}
    updated_parameters::Dict{MOI.VariableIndex,T}
    variables::Dict{MOI.VariableIndex,MOI.VariableIndex}
    last_variable_index_added::Int64
    last_parameter_index_added::Int64
    affine_constraint_cache::MOI.Utilities.DoubleDicts.DoubleDict{
        Vector{MOI.ScalarAffineTerm{Float64}},
    }
    quadratic_constraint_cache_pv::Dict{
        MOI.ConstraintIndex,
        Vector{MOI.ScalarQuadraticTerm{Float64}},
    }
    quadratic_constraint_cache_pp::Dict{
        MOI.ConstraintIndex,
        Vector{MOI.ScalarQuadraticTerm{Float64}},
    }
    quadratic_constraint_cache_pc::MOI.Utilities.DoubleDicts.DoubleDict{
        Vector{MOI.ScalarAffineTerm{Float64}},
    }
    quadratic_constraint_variables_associated_to_parameters_cache::Dict{
        MOI.ConstraintIndex,
        Vector{MOI.ScalarAffineTerm{T}},
    }
    quadratic_added_cache::Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}
    last_quad_add_added::Int64
    vector_constraint_cache::MOI.Utilities.DoubleDicts.DoubleDict{
        Vector{MOI.VectorAffineTerm{Float64}},
    }
    affine_objective_cache::Vector{MOI.ScalarAffineTerm{T}}
    quadratic_objective_cache_pv::Vector{MOI.ScalarQuadraticTerm{T}}
    quadratic_objective_cache_pp::Vector{MOI.ScalarQuadraticTerm{T}}
    quadratic_objective_cache_pc::Vector{MOI.ScalarAffineTerm{T}}
    quadratic_objective_variables_associated_to_parameters_cache::Vector{
        MOI.ScalarAffineTerm{T},
    }
    multiplicative_parameters::Set{Int64}
    dual_value_of_parameters::Vector{Float64}
    evaluate_duals::Bool
    number_of_parameters_in_model::Int64
    function ParametricOptimizer(
        optimizer::OT;
        evaluate_duals::Bool = true,
    ) where {OT}
        return new{Float64,OT}(
            optimizer,
            Dict{MOI.VariableIndex,Float64}(),
            Dict{MOI.VariableIndex,String}(),
            Dict{MOI.VariableIndex,Float64}(),
            Dict{MOI.VariableIndex,MOI.VariableIndex}(),
            0,
            PARAMETER_INDEX_THRESHOLD,
            MOI.Utilities.DoubleDicts.DoubleDict{
                Vector{MOI.ScalarAffineTerm{Float64}},
            }(),
            Dict{MOI.ConstraintIndex,Vector{MOI.ScalarQuadraticTerm{Float64}}}(),
            Dict{MOI.ConstraintIndex,Vector{MOI.ScalarQuadraticTerm{Float64}}}(),
            MOI.Utilities.DoubleDicts.DoubleDict{
                Vector{MOI.ScalarAffineTerm{Float64}},
            }(),
            Dict{MOI.ConstraintIndex,Vector{MOI.ScalarAffineTerm{Float64}}}(),
            Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}(),
            0,
            MOI.Utilities.DoubleDicts.DoubleDict{
                Vector{MOI.VectorAffineTerm{Float64}},
            }(),
            Vector{MOI.ScalarAffineTerm{Float64}}(),
            Vector{MOI.ScalarQuadraticTerm{Float64}}(),
            Vector{MOI.ScalarQuadraticTerm{Float64}}(),
            Vector{MOI.ScalarAffineTerm{Float64}}(),
            Vector{MOI.ScalarAffineTerm{Float64}}(),
            Set{Int64}(),
            Vector{Float64}(),
            evaluate_duals,
            0,
        )
    end
end

include("utils.jl")
include("duals.jl")
include("update_parameters.jl")

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
           isempty(
               model.quadratic_constraint_variables_associated_to_parameters_cache,
           ) &&
           isempty(model.quadratic_added_cache) &&
           model.last_quad_add_added == 0 &&
           isempty(model.affine_objective_cache) &&
           isempty(model.quadratic_objective_cache_pv) &&
           isempty(model.quadratic_objective_cache_pp) &&
           isempty(model.quadratic_objective_cache_pc) &&
           isempty(
               model.quadratic_objective_variables_associated_to_parameters_cache,
           ) &&
           isempty(model.dual_value_of_parameters) &&
           model.number_of_parameters_in_model == 0
end

function MOI.supports_constraint(
    model::ParametricOptimizer,
    F::Union{
        Type{MOI.SingleVariable},
        Type{MOI.ScalarAffineFunction{T}},
        Type{MOI.VectorOfVariables},
        Type{MOI.VectorAffineFunction{T}},
    },
    S::Type{<:MOI.AbstractSet},
) where {T}
    return MOI.supports_constraint(model.optimizer, F, S)
end

function MOI.supports_constraint(
    model::ParametricOptimizer,
    ::Type{MOI.ScalarQuadraticFunction{T}},
    S::Type{<:MOI.AbstractSet},
) where {T}
    return MOI.supports_constraint(
        model.optimizer,
        MOI.ScalarAffineFunction{T},
        S,
    )
end

function MOI.supports(
    model::ParametricOptimizer,
    attr::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{MOI.SingleVariable},
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}},
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}},
    },
) where {T}
    return MOI.supports(model.optimizer, attr)
end

function MOI.empty!(model::ParametricOptimizer{T}) where {T}
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
    empty!(model.vector_constraint_cache)
    empty!(model.affine_objective_cache)
    empty!(model.quadratic_objective_cache_pv)
    empty!(model.quadratic_objective_cache_pp)
    empty!(model.quadratic_objective_cache_pc)
    empty!(model.quadratic_objective_variables_associated_to_parameters_cache)
    empty!(model.dual_value_of_parameters)
    model.number_of_parameters_in_model = 0
    return
end

function MOI.set(
    model::ParametricOptimizer,
    attr::MOI.VariableName,
    v::MOI.VariableIndex,
    name::String,
)
    if is_parameter_in_model(model, v)
        model.parameters_name[v] = name
    else
        MOI.set(model.optimizer, attr, v, name)
    end
    return
end

function MOI.get(
    model::ParametricOptimizer,
    attr::MOI.VariableName,
    v::MOI.VariableIndex,
)
    if is_parameter_in_model(model, v)
        return get(model.parameters_name, v, "")
    else
        return MOI.get(model.optimizer, attr, v)
    end
end

function MOI.supports(
    model::ParametricOptimizer,
    attr::MOI.VariableName,
    tp::Type{MOI.VariableIndex},
)
    return MOI.supports(model.optimizer, attr, tp)
end

function MOI.set(
    model::ParametricOptimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex,
    name::String,
)
    MOI.set(model.optimizer, attr, c, name)
    return
end

function MOI.get(
    model::ParametricOptimizer,
    attr::MOI.ConstraintName,
    c::MOI.ConstraintIndex,
)
    return MOI.get(model.optimizer, attr, c)
end

function MOI.get(model::ParametricOptimizer, ::MOI.NumberOfVariables)
    return length(model.parameters) + length(model.variables)
end

function MOI.supports(
    model::ParametricOptimizer,
    attr::MOI.ConstraintName,
    tp::Type{<:MOI.ConstraintIndex},
)
    return MOI.supports(model.optimizer, attr, tp)
end

# TODO
# This is not correct, you need to put the parameters back into the function
# function MOI.get(model::ParametricOptimizer, attr::MOI.ConstraintFunction, ci::MOI.ConstraintIndex{F, S}) where {F, S}
#     MOI.get(model.optimizer, attr, ci)
# end

function MOI.get(
    model::ParametricOptimizer,
    attr::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    MOI.throw_if_not_valid(model, ci)
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.get(model::ParametricOptimizer, attr::MOI.ObjectiveSense)
    return MOI.get(model.optimizer, attr)
end

function MOI.get(model::ParametricOptimizer, attr::MOI.ObjectiveFunctionType)
    if !isempty(model.quadratic_constraint_cache_pv) ||
       !isempty(model.quadratic_constraint_cache_pc)
        return MOI.ScalarQuadraticFunction{Float64}
    end
    return MOI.get(model.optimizer, attr)
end

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
    attr::MOI.ListOfConstraintIndices{F,S},
) where {S,F<:Union{MOI.VectorOfVariables,MOI.SingleVariable}}
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
    ::ParametricOptimizer,
    ::Type{Parameter},
)
    return true
end

function MOI.supports_add_constrained_variables(
    model::ParametricOptimizer,
    ::Type{MOI.Reals},
)
    return MOI.supports_add_constrained_variables(model.optimizer, MOI.Reals)
end

function MOI.add_variable(model::ParametricOptimizer)
    next_variable_index!(model)
    v_p = MOI.VariableIndex(model.last_variable_index_added)
    v = MOI.add_variable(model.optimizer)
    model.variables[v_p] = v
    return v_p
end

function MOI.add_constrained_variable(
    model::ParametricOptimizer,
    set::Parameter,
)
    next_parameter_index!(model)
    p = MOI.VariableIndex(model.last_parameter_index_added)
    model.parameters[p] = set.val
    cp = MOI.ConstraintIndex{MOI.SingleVariable,Parameter}(
        model.last_parameter_index_added,
    )
    update_number_of_parameters!(model)
    return p, cp
end

function MOI.add_constraint(
    model::ParametricOptimizer,
    f::MOI.SingleVariable,
    set::MOI.AbstractScalarSet,
)
    if is_parameter_in_model(model, f.variable)
        error("Cannot constrain a parameter")
    elseif !is_variable_in_model(model, f.variable)
        error("Variable not in the model")
    end
    return MOI.add_constraint(model.optimizer, f, set)
end

function add_constraint_with_parameters_on_function(
    model::ParametricOptimizer,
    f::MOI.ScalarAffineFunction{T},
    set::MOI.AbstractScalarSet,
) where {T}
    vars, params, param_constant =
        separate_possible_terms_and_calculate_parameter_constant(model, f.terms)
    ci = MOI.Utilities.normalize_and_add_constraint(
        model.optimizer,
        MOI.ScalarAffineFunction(vars, f.constant + param_constant),
        set,
    )
    model.affine_constraint_cache[ci] = params
    return ci
end

function MOI.add_constraint(
    model::ParametricOptimizer,
    f::MOI.ScalarAffineFunction{T},
    set::MOI.AbstractScalarSet,
) where {T}
    if !function_has_parameters(model, f)
        return MOI.add_constraint(model.optimizer, f, set)
    else
        return add_constraint_with_parameters_on_function(model, f, set)
    end
end

function MOI.get(
    model::ParametricOptimizer,
    attr::MOI.VariablePrimal,
    v::MOI.VariableIndex,
)
    if is_parameter_in_model(model, v)
        return model.parameters[v]
    elseif is_variable_in_model(model, v)
        return MOI.get(model.optimizer, attr, model.variables[v])
    else
        error("Variable not in the model")
    end
end

function MOI.set(
    model::ParametricOptimizer,
    ::MOI.ConstraintSet,
    cp::MOI.ConstraintIndex{MOI.SingleVariable,Parameter},
    set::Parameter,
)
    p = MOI.VariableIndex(cp.value)
    if !is_parameter_in_model(model, p)
        error("Parameter not in the model")
    end
    return model.updated_parameters[p] = set.val
end

struct ParameterValue <: MOI.AbstractVariableAttribute end

"""
    MOI.set(
        model::ParametricOptimizer,
        ::ParameterValue,
        vi::MOI.VariableIndex,
        val::Float64,
    )

Sets the parameter to a given value, using its `MOI.VariableIndex` as reference.

# Example

```julia-repl
julia> MOI.set(model, ParameterValue(), w, 2.0)
2.0
```
"""
function MOI.set(
    model::ParametricOptimizer,
    ::ParameterValue,
    vi::MOI.VariableIndex,
    val::Float64,
)
    if !is_parameter_in_model(model, vi)
        error("Parameter not in the model")
    end
    return model.updated_parameters[vi] = val
end

function MOI.set(
    model::ParametricOptimizer,
    ::ParameterValue,
    vi::MOI.VariableIndex,
    val::Real,
)
    return MOI.set(model, ParameterValue(), vi, convert(Float64, val))
end

function empty_objective_function_caches!(model::ParametricOptimizer)
    empty!(model.affine_objective_cache)
    empty!(model.quadratic_objective_cache_pv)
    empty!(model.quadratic_objective_cache_pp)
    empty!(model.quadratic_objective_cache_pc)
    return
end

function MOI.set(
    model::ParametricOptimizer,
    attr::MOI.ObjectiveFunction,
    f::MOI.ScalarAffineFunction{T},
) where {T}
    # clear previously defined objetive function cache
    empty_objective_function_caches!(model)
    if !function_has_parameters(model, f)
        MOI.set(model.optimizer, attr, f)
    else
        vars, params, param_constant =
            separate_possible_terms_and_calculate_parameter_constant(
                model,
                f.terms,
            )
        MOI.set(
            model.optimizer,
            attr,
            MOI.ScalarAffineFunction(vars, f.constant + param_constant),
        )
        model.affine_objective_cache = params
    end
    return
end

function MOI.set(
    model::ParametricOptimizer,
    attr::MOI.ObjectiveFunction,
    v::MOI.SingleVariable,
)
    if haskey(model.parameters, v)
        error("Cannot use a parameter as objective function alone")
    elseif !haskey(model.variables, v)
        error("Variable not in the model")
    end
    return MOI.set(
        model.optimizer,
        attr,
        MOI.SingleVariable(model.variables[v.variable]),
    )
end

function MOI.set(
    model::ParametricOptimizer,
    attr::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    MOI.set(model.optimizer, attr, sense)
    return
end

function MOI.get(
    model::ParametricOptimizer,
    attr::T,
) where {
    T<:Union{
        MOI.TerminationStatus,
        MOI.ObjectiveValue,
        MOI.DualObjectiveValue,
        MOI.PrimalStatus,
        MOI.DualStatus,
    },
}
    return MOI.get(model.optimizer, attr)
end

function MOI.get(
    model::ParametricOptimizer,
    attr::T,
    c::MOI.ConstraintIndex,
) where {T<:Union{MOI.ConstraintPrimal,MOI.ConstraintDual}}
    return MOI.get(model.optimizer, attr, c)
end

function MOI.set(model::ParametricOptimizer, ::MOI.Silent, bool::Bool)
    MOI.set(model.optimizer, MOI.Silent(), bool)
    return
end

function MOI.set(model::ParametricOptimizer, attr::MOI.RawParameter, val::Any)
    MOI.set(model.optimizer, attr, val)
    return
end

# TODO(odow): remove this.
function MOI.set(model::ParametricOptimizer, attr::String, val::Any)
    return MOI.set(model.optimizer, MOI.RawParameter(attr), val)
end

function MOI.get(model::ParametricOptimizer, ::MOI.SolverName)
    name = MOI.get(model.optimizer, MOI.SolverName())
    return "ParametricOptimizer with $(name) attached"
end

function MOI.add_constraint(
    model::ParametricOptimizer,
    f::MOI.VectorOfVariables,
    set::MOI.AbstractVectorSet,
) where {T}
    if function_has_parameters(model, f)
        error("VectorOfVariables does not allow parameters")
    end
    return MOI.add_constraint(model.optimizer, f, set)
end

function MOI.add_constraint(
    model::ParametricOptimizer,
    f::MOI.VectorAffineFunction{T},
    set::MOI.AbstractVectorSet,
) where {T}
    if !function_has_parameters(model, f)
        return MOI.add_constraint(model.optimizer, f, set)
    else
        return add_constraint_with_parameters_on_function(model, f, set)
    end
end

function add_constraint_with_parameters_on_function(
    model::ParametricOptimizer,
    f::MOI.VectorAffineFunction{T},
    set::MOI.AbstractVectorSet,
) where {T}
    vars, params, param_constants =
        separate_possible_terms_and_calculate_parameter_constant(model, f, set)
    ci = MOI.add_constraint(
        model.optimizer,
        MOI.VectorAffineFunction(vars, f.constants + param_constants),
        set,
    )
    model.vector_constraint_cache[ci] = params
    return ci
end

function add_constraint_with_parameters_on_function(
    model::ParametricOptimizer,
    f::MOI.ScalarQuadraticFunction{T},
    set::S,
) where {T,S<:MOI.AbstractScalarSet}
    if function_quadratic_terms_has_parameters(model, f.quadratic_terms)
        (
            quad_vars,
            quad_aff_vars,
            quad_params,
            aff_terms,
            variables_associated_to_parameters,
            quad_param_constant,
        ) = separate_possible_terms_and_calculate_parameter_constant(
            model,
            f.quadratic_terms,
        )
    else
        quad_vars = f.quadratic_terms
        quad_aff_vars = MOI.ScalarQuadraticTerm{T}[]
        quad_params = MOI.ScalarQuadraticTerm{T}[]
        aff_terms = MOI.ScalarAffineTerm{T}[]
        quad_param_constant = zero(T)
        variables_associated_to_parameters = MOI.VariableIndex[]
    end
    if function_affine_terms_has_parameters(model, f.affine_terms)
        (
            aff_vars,
            aff_params,
            terms_with_variables_associated_to_parameters,
            aff_param_constant,
        ) = separate_possible_terms_and_calculate_parameter_constant(
            model,
            f.affine_terms,
            variables_associated_to_parameters,
        )
    else
        aff_vars = f.affine_terms
        aff_params = MOI.ScalarAffineTerm{T}[]
        terms_with_variables_associated_to_parameters =
            MOI.ScalarAffineTerm{T}[]
        aff_param_constant = zero(T)
    end
    aff_terms = vcat(aff_terms, aff_vars)
    const_term = f.constant + aff_param_constant + quad_param_constant
    quad_terms = quad_vars
    f_quad = if !isempty(quad_vars)
        MOI.ScalarQuadraticFunction(aff_terms, quad_terms, const_term)
    else
        MOI.ScalarAffineFunction(aff_terms, const_term)
    end
    model.last_quad_add_added += 1
    ci =
        MOI.Utilities.normalize_and_add_constraint(model.optimizer, f_quad, set)
    # This part is used to remember that ci came from a quadratic function
    # It is particularly useful because sometimes the constraint mutates
    new_ci = MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S}(
        model.last_quad_add_added,
    )
    model.quadratic_added_cache[new_ci] = ci
    fill_quadratic_constraint_caches!(
        model,
        new_ci,
        quad_aff_vars,
        quad_params,
        aff_params,
        terms_with_variables_associated_to_parameters,
    )
    return new_ci
end

function MOI.add_constraint(
    model::ParametricOptimizer,
    f::MOI.ScalarQuadraticFunction{T},
    set::MOI.AbstractScalarSet,
) where {T}
    if !function_has_parameters(model, f)
        return MOI.add_constraint(model.optimizer, f, set)
    else
        return add_constraint_with_parameters_on_function(model, f, set)
    end
end

function MOI.set(
    model::ParametricOptimizer,
    attr::MOI.ObjectiveFunction,
    f::MOI.ScalarQuadraticFunction{T},
) where {T}
    # clear previously defined objetive function cache
    empty_objective_function_caches!(model)
    if !function_has_parameters(model, f)
        MOI.set(model.optimizer, attr, f)
        return
    end
    if function_quadratic_terms_has_parameters(model, f.quadratic_terms)
        (
            quad_vars,
            quad_aff_vars,
            quad_params,
            aff_terms,
            variables_associated_to_parameters,
            quad_param_constant,
        ) = separate_possible_terms_and_calculate_parameter_constant(
            model,
            f.quadratic_terms,
        )
    else
        quad_vars = f.quadratic_terms
        quad_aff_vars = MOI.ScalarQuadraticTerm{T}[]
        quad_params = MOI.ScalarQuadraticTerm{T}[]
        aff_terms = MOI.ScalarAffineTerm{T}[]
        quad_param_constant = zero(T)
        variables_associated_to_parameters = MOI.VariableIndex[]
    end

    if function_affine_terms_has_parameters(model, f.affine_terms)
        (
            aff_vars,
            aff_params,
            terms_with_variables_associated_to_parameters,
            aff_param_constant,
        ) = separate_possible_terms_and_calculate_parameter_constant(
            model,
            f.affine_terms,
            variables_associated_to_parameters,
        )
    else
        aff_vars = f.affine_terms
        aff_params = MOI.ScalarAffineTerm{T}[]
        terms_with_variables_associated_to_parameters =
            MOI.ScalarAffineTerm{T}[]
        aff_param_constant = zero(T)
    end

    aff_terms = vcat(aff_terms, aff_vars)
    const_term = f.constant + aff_param_constant + quad_param_constant
    quad_terms = quad_vars

    if !isempty(quad_vars)
        MOI.set(
            model.optimizer,
            attr,
            MOI.ScalarQuadraticFunction(aff_terms, quad_terms, const_term),
        )
    else
        MOI.set(
            model.optimizer,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
            MOI.ScalarAffineFunction(aff_terms, const_term),
        )
    end

    if !isempty(quad_terms)
        f_quad = MOI.ScalarQuadraticFunction(aff_terms, quad_terms, const_term)

        MOI.set(model.optimizer, attr, f_quad)
    else
        f_quad = MOI.ScalarAffineFunction(aff_terms, const_term)

        MOI.set(
            model.optimizer,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
            f_quad,
        )
    end

    model.quadratic_objective_cache_pv = quad_aff_vars
    model.quadratic_objective_cache_pp = quad_params
    model.quadratic_objective_cache_pc = aff_params
    model.quadratic_objective_variables_associated_to_parameters_cache =
        terms_with_variables_associated_to_parameters

    return
end

function MOI.optimize!(model::ParametricOptimizer)
    if !isempty(model.updated_parameters)
        update_parameters!(model)
    end
    MOI.optimize!(model.optimizer)
    if model.evaluate_duals
        calculate_dual_of_parameters(model)
    end
    return
end

end # module
