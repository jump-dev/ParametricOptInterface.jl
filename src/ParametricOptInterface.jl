module ParametricOptInterface

using MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const POI = ParametricOptInterface

struct Parameter <: MOI.AbstractScalarSet
    val::Float64
end

mutable struct ParametricOptimizer{T, OT <: MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT 
    parameters::Dict{MOI.VariableIndex, T}
    updated_parameters::Dict{MOI.VariableIndex, T}
    variables::Dict{MOI.VariableIndex, MOI.VariableIndex}
    last_index_added::Int
    affine_constraint_cache::Dict{Any, Any}
    quadratic_constraint_cache_pv::Dict{Any, Any} #param*var
    quadratic_constraint_cache_pp::Dict{Any, Any} #param*param
    quadratic_constraint_cache_pc::Dict{Any, Any} #param*cons
    affine_objective_cache::Dict{Any, Any}
    quadratic_objective_cache_pv::Dict{Any, Any}
    quadratic_objective_cache_pp::Dict{Any, Any}
    quadratic_objective_cache_pc::Dict{Any, Any}
    function ParametricOptimizer(optimizer::OT) where OT
        new{Float64, OT}(optimizer, Dict{MOI.VariableIndex, Float64}(), Dict{MOI.VariableIndex, Float64}(),
            Dict{MOI.VariableIndex, MOI.VariableIndex}(), 0, Dict{Any, Any}(), Dict{Any, Any}(),
            Dict{Any, Any}(), Dict{Any, Any}(), Dict{Any, Any}())
    end
end


function MOI.add_variable(model::ParametricOptimizer)
    model.last_index_added += 1
    v_p = MOI.VariableIndex(model.last_index_added)
    v = MOI.add_variable(model.optimizer)
    model.variables[v_p] = v
    return v_p
end

function MOI.add_constrained_variable(model::ParametricOptimizer, set::Parameter)
    model.last_index_added += 1
    p = MOI.VariableIndex(model.last_index_added)
    model.parameters[p] = set.val
    return p, MOI.ConstraintIndex{MOI.SingleVariable, typeof(set)}(model.last_index_added)
end

function MOI.add_constraint(model::ParametricOptimizer, f::MOI.SingleVariable, set::MOI.AbstractScalarSet) 
    if haskey(model.parameters, f.variable)
        error("Cannot constrain a parameter")
   elseif haskey(model.variables, f.variable)
       return MOI.add_constraint(model.optimizer, f, set)
   else
       error("Variable not in the model")
   end
end

function MOI.add_constraint(model::ParametricOptimizer, f::MOI.ScalarAffineFunction{T}, set::MOI.AbstractScalarSet) where T   
    if !any(haskey(model.parameters, f.terms[i].variable_index) for i = 1:length(f.terms))
        return MOI.add_constraint(model.optimizer, f, set) 
    else
        vars = MOI.ScalarAffineTerm{T}[]
        params = MOI.ScalarAffineTerm{T}[]

        for i in f.terms
            if haskey(model.variables, i.variable_index)
                push!(vars, i)
            elseif haskey(model.parameters, i.variable_index)
                push!(params, i)
            else
                error("Constraint uses a variable that is not in the model")
            end
        end
        param_constant = 0
        for j in params
            param_constant += j.coefficient * model.parameters[j.variable_index]
        end
        fvar = MOI.ScalarAffineFunction(vars, f.constant + param_constant)
        ci = MOIU.normalize_and_add_constraint(model.optimizer, fvar, set)     
        model.affine_constraint_cache[ci] = params
        return ci
    end
end

function update_constant!(s::MOI.LessThan{T}, val) where T
    MOI.LessThan{T}(s.upper - val)
end

function update_constant!(s::MOI.GreaterThan{T}, val) where T
    MOI.GreaterThan{T}(s.lower - val)
end

function update_constant!(s::MOI.EqualTo{T}, val) where T
    MOI.EqualTo{T}(s.value - val)
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
                set = update_constant!(set, param_constant)
                MOI.set(model.optimizer, MOI.ConstraintSet(), ci, set)
            end
        end

        for (oi, fobj) in model.affine_objective_cache
            objective_constant = 0
            for j in fobj
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

        # for (ci, fparam) in model.quadratic_constraint_cache_pc
        #     param_constant = 0
        #     for j in fparam
        #         if haskey(model.updated_parameters, j.variable_index)
        #             param_old = model.parameters[j.variable_index]
        #             param_new = model.updated_parameters[j.variable_index]
        #             aux = param_new - param_old
        #             param_constant += j.coefficient * aux
        #         end
        #     end
        #     if param_constant != 0
        #         set = MOI.get(model.optimizer, MOI.ConstraintSet(), ci)
        #         set = update_constant!(set, param_constant)
        #         MOI.set(model.optimizer, MOI.ConstraintSet(), ci, set)
        #     end
        # end

        # for (oi, fobj) in model.quadratic_objective_cache_pc
        #     objective_constant = 0
        #     for j in fobj
        #         if haskey(model.updated_parameters, j.variable_index)
        #             param_old = model.parameters[j.variable_index]
        #             param_new = model.updated_parameters[j.variable_index]
        #             aux = param_new - param_old
        #             objective_constant += j.coefficient * aux
        #         end
        #     end
        #     if objective_constant != 0
        #         F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
        #         f = MOI.get(model.optimizer, MOI.ObjectiveFunction{F}())
        #         fvar = MOI.ScalarQuadraticFunction(f.affine_terms, f.quadratic_terms, f.constant + objective_constant)
        #         MOI.set(model.optimizer,MOI.ObjectiveFunction{F}(), fvar)       
        #     end
        # end



        # for (ci, fparam) in model.quadratic_constraint_cache_pp
        #     param_constant = 0
        #     for j in fparam
        #         if haskey(model.updated_parameters, j.variable_index)
        #             param_old = model.parameters[j.variable_index]
        #             param_new = model.updated_parameters[j.variable_index]
        #             aux = param_new - param_old
        #             param_constant += j.coefficient * aux
        #         end
        #     end
        #     if param_constant != 0
        #         set = MOI.get(model.optimizer, MOI.ConstraintSet(), ci)
        #         set = update_constant!(set, param_constant)
        #         MOI.set(model.optimizer, MOI.ConstraintSet(), ci, set)
        #     end
        # end

        # for (oi, fobj) in model.quadratic_objective_cache_pp
        #     objective_constant = 0
        #     for j in fobj
        #         if haskey(model.updated_parameters, j.variable_index)
        #             param_old = model.parameters[j.variable_index]
        #             param_new = model.updated_parameters[j.variable_index]
        #             aux = param_new - param_old
        #             objective_constant += j.coefficient * aux
        #         end
        #     end
        #     if objective_constant != 0
        #         F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
        #         f = MOI.get(model.optimizer, MOI.ObjectiveFunction{F}())
        #         fvar = MOI.ScalarQuadraticFunction(f.affine_terms, f.quadratic_terms, f.constant + objective_constant)
        #         MOI.set(model.optimizer,MOI.ObjectiveFunction{F}(), fvar)       
        #     end
        # end

        # for (ci, fparam) in model.quadratic_constraint_cache_pv
        #     param_constant = 0
        #     for j in fparam
        #         if haskey(model.updated_parameters, j.variable_index)
        #             param_old = model.parameters[j.variable_index]
        #             param_new = model.updated_parameters[j.variable_index]
        #             aux = param_new - param_old
        #             param_constant += j.coefficient * aux
        #         end
        #     end
        #     if param_constant != 0
        #         set = MOI.get(model.optimizer, MOI.ConstraintSet(), ci)
        #         set = update_constant!(set, param_constant)
        #         MOI.set(model.optimizer, MOI.ConstraintSet(), ci, set)
        #     end
        # end

        # for (oi, fobj) in model.quadratic_objective_cache_pv
        #     objective_constant = 0
        #     for j in fobj
        #         if haskey(model.updated_parameters, j.variable_index)
        #             param_old = model.parameters[j.variable_index]
        #             param_new = model.updated_parameters[j.variable_index]
        #             aux = param_new - param_old
        #             objective_constant += j.coefficient * aux
        #         end
        #     end
        #     if objective_constant != 0
        #         F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
        #         f = MOI.get(model.optimizer, MOI.ObjectiveFunction{F}())
        #         fvar = MOI.ScalarQuadraticFunction(f.affine_terms, f.quadratic_terms, f.constant + objective_constant)
        #         MOI.set(model.optimizer,MOI.ObjectiveFunction{F}(), fvar)       
        #     end
        # end

        for (i, val) in model.updated_parameters
            model.parameters[i] = val
        end
        empty!(model.updated_parameters)
    end
    MOI.optimize!(model.optimizer)
end

function MOI.get(model::ParametricOptimizer, attr::MOI.VariablePrimal, v::MOI.VariableIndex)
    if haskey(model.parameters, v)
         return model.parameters[v]
    elseif haskey(model.variables, v)
        return MOI.get(model.optimizer, attr, model.variables[v])
    else
        error("Variable not in the model")
    end
end

function MOI.set(model::ParametricOptimizer, ::MOI.ConstraintSet, cp::MOI.ConstraintIndex{MOI.SingleVariable, P}, set::P) where {P <: Parameter}
    p = MOI.VariableIndex(cp.value)
    if haskey(model.parameters, p)
        return model.updated_parameters[p] = set.val
    else
        error("Parameter not in the model")
    end
end

function MOI.set(model::ParametricOptimizer, attr::MOI.ObjectiveFunction{F}, f::F) where {F <: MOI.ScalarAffineFunction{T}} where T
    if !any(haskey(model.parameters, f.terms[i].variable_index) for i = 1:length(f.terms))
        MOI.set(model.optimizer, attr, f) 
        return
    else
        vars = MOI.ScalarAffineTerm{T}[]
        params = MOI.ScalarAffineTerm{T}[]
        for i in f.terms
            if haskey(model.variables, i.variable_index)
                push!(vars, i)
            elseif haskey(model.parameters, i.variable_index)
                push!(params, i)
            else
                error("Objective function uses a variable that is not in the model")
            end
        end
        param_constant = 0
        for j in params
            param_constant += j.coefficient * model.parameters[j.variable_index]
        end
        fvar = MOI.ScalarAffineFunction(vars, f.constant + param_constant)
        co = MOI.set(model.optimizer, attr, fvar)
        model.affine_objective_cache[co] = params
        return
    end
end

function MOI.set(model::ParametricOptimizer, attr::MOI.ObjectiveFunction{F}, v::F) where {F <: MOI.SingleVariable} 
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
        MOI.PrimalStatus,
    }
}
    return MOI.get(model.optimizer, attr)
end


### new functions

function MOI.add_constraint(model::ParametricOptimizer, f::MOI.VectorOfVariables, set::MOI.AbstractVectorSet) where T   
    if any(haskey(model.parameters, f.variables) for i = 1:length(f.variables))
        error("VectorOfVariables does not allow parameters")
    else
        return MOI.add_constraint(model.optimizer, f, set)
    end
end

function MOI.add_constraint(model::ParametricOptimizer, f::MOI.VectorAffineFunction{T}, set::MOI.AbstractVectorSet) where T   
    if any(haskey(model.parameters, f.terms[i].variable_index) for i = 1:length(f.terms))
        error("VectorAffineFunction does not allow parameters")
    else
        return MOI.add_constraint(model.optimizer, f, set)
    end
end


function MOI.add_constraint(model::ParametricOptimizer, f::MOI.ScalarQuadraticFunction{T}, set::MOI.AbstractScalarSet) where T   
    # if there are no parameters in the quadratic or affine terms, simply add the variable
    if (!any(haskey(model.parameters, f.affine_terms[i].variable_index) for i = 1:length(f.affine_terms)) && 
        !any(haskey(model.parameters, f.quadratic_terms[j].variable_index_1) for j = 1:length(f.quadratic_terms)) &&
        !any(haskey(model.parameters, f.quadratic_terms[j].variable_index_2) for j = 1:length(f.quadratic_terms)))

        return MOI.add_constraint(model.optimizer, f, set) 

    else
    # let's work the AffineTerms first

        aff_params = MOI.ScalarAffineTerm{T}[] #outside declaration so it has default value

        if any(haskey(model.parameters, f.affine_terms[i].variable_index) for i = 1:length(f.affine_terms))

            aff_vars = MOI.ScalarAffineTerm{T}[]

            for i in f.affine_terms
                if haskey(model.variables, i.variable_index)
                    push!(aff_vars, i)
                elseif haskey(model.parameters, i.variable_index)
                    push!(aff_params, i)
                else
                    error("Constraint uses a variable that is not in the model")
                end
            end
            
            aff_constant = 0
            for j in aff_params
                aff_constant += j.coefficient * model.parameters[j.variable_index]
            end

            f.constant += aff_constant 

        else
            aff_vars = model.affine_terms
        end

        
        quad_params = MOI.ScalarQuadraticTerm{T}[] #outside declaration so it has default value
        quad_aff_vars = MOI.ScalarQuadraticTerm{T}[] #outside declaration so it has default value; parameter as variable_index_1

        if any(haskey(model.parameters, f.quadratic_terms[i].variable_index_1)) || any(haskey(model.parameters, f.quadratic_terms[i].variable_index_2)) 
            
            quad_terms = MOI.ScalarQuadraticTerm{T}[]  

            for i in f.quadratic_terms
                if haskey(model.variables, i.variable_index_1) && haskey(model.variables, i.variable_index_2)
                    push!(quad_terms, i) # if there are only variables, it remains a quadratic term

                elseif haskey(model.parameters, i.variable_index_1) && haskey(model.variables, i.variable_index_2)
                    # This is the case when i.variable_index_1 is a parameter and i.variable_index_2 is a variable.
                    # Thus, it creates an affine term. Convention: param as 1, var as 2
                    aux = MOI.ScalarQuadraticTerm(i.coefficient, i.variable_index_1, i.variable_index_2)
                    push!(quad_aff_vars, aux)  

                elseif haskey(model.variables, i.variable_index_1) && haskey(model.parameters, i.variable_index_2)
                    # Check convention defined above
                    aux = MOI.ScalarQuadraticTerm(i.coefficient, i.variable_index_2, i.variable_index_1)
                    push!(quad_aff_vars, aux)          

                elseif haskey(model.parameters, i.variable_index_1) && haskey(model.parameters, i.variable_index_2)
                    # This is the case where both variable_index_1,2 are actually parameters
                    aux = MOI.ScalarQuadraticTerm(i.coefficient, i.variable_index_1, i.variable_index_2)
                    push!(quad_params, aux)

                else
                    error("Constraint uses a variable that is not in the model")
                end
            end

        else
            quad_terms = f.quadratic_terms
        end

        aff_terms = MOI.ScalarAffineTerm{T}[]

        for i in quad_aff_vars
            aux = MOI.ScalarAffineTerm(i.coefficient * model.parameters[i.variable_index_1], i.variable_index_2)
            push!(aff_terms, aux)
        end

        push!(aff_terms, aff_vars)

        const_term = f.constant

        for j in aff_params
            const_term += j.coefficient * model.parameters[j.variable_index]
        end

        for j in quad_params
            const_term += j.coefficient * model.parameters[j.variable_index_1] * model.parameters[j.variable_index_2]
        end


        f_quad = MOI.ScalarQuadraticFunction(
                    aff_terms,
                    quad_terms,
                    const_term 
                )
        
        ci = MOIU.normalize_and_add_constraint(model.optimizer, f_quad, set)

        model.quadratic_constraint_cache_pv[ci] = quad_aff_vars
        model.quadratic_constraint_cache_pp[ci] = quad_params
        model.quadratic_constraint_cache_pc[ci] = aff_params

        return ci
    end

end


function MOI.set(model::ParametricOptimizer, attr::MOI.ObjectiveFunction{F}, f::F) where {F <: MOI.ScalarQuadraticFunction{T}} where T
    if (!any(haskey(model.parameters, f.affine_terms[i].variable_index) for i = 1:length(f.affine_terms)) && 
        !any(haskey(model.parameters, f.quadratic_terms[j].variable_index_1) for j = 1:length(f.quadratic_terms)) &&
        !any(haskey(model.parameters, f.quadratic_terms[j].variable_index_2) for j = 1:length(f.quadratic_terms)))
        
        MOI.set(model.optimizer, attr, f) 
        @show("Success")
        return
    else
        aff_params = MOI.ScalarAffineTerm{T}[] #outside declaration so it has default value

        if any(haskey(model.parameters, f.affine_terms[i].variable_index) for i = 1:length(f.affine_terms))     

            aff_vars = MOI.ScalarAffineTerm{T}[]

            for i in f.affine_terms
                if haskey(model.variables, i.variable_index)
                    push!(aff_vars, i)
                elseif haskey(model.parameters, i.variable_index)
                    push!(aff_params, i)
                else
                    error("Constraint uses a variable that is not in the model")
                end
            end
            
            aff_constant = 0
            for j in aff_params
                aff_constant += j.coefficient * model.parameters[j.variable_index]
            end

            f.constant += aff_constant 

        else
            aff_vars = model.affine_terms
        end

        
        quad_params = MOI.ScalarQuadraticTerm{T}[] #outside declaration so it has default value
        quad_aff_vars = MOI.ScalarQuadraticTerm{T}[] #outside declaration so it has default value; parameter as variable_index_1

        if (any(haskey(model.parameters, f.quadratic_terms[i].variable_index_1) for i = 1:length(f.quadratic_terms)) ||
             any(haskey(model.parameters, f.quadratic_terms[i].variable_index_2) for i = 1:length(f.quadratic_terms)))
            
            quad_terms = MOI.ScalarQuadraticTerm{T}[]  

            for i in f.quadratic_terms
                if haskey(model.variables, i.variable_index_1) && haskey(model.variables, i.variable_index_2)
                    push!(quad_terms, i) # if there are only variables, it remains a quadratic term

                elseif haskey(model.parameters, i.variable_index_1) && haskey(model.variables, i.variable_index_2)
                    # This is the case when i.variable_index_1 is a parameter and i.variable_index_2 is a variable.
                    # Thus, it creates an affine term. Convention: param as 1, var as 2
                    aux = MOI.ScalarQuadraticTerm(i.coefficient, i.variable_index_1, i.variable_index_2)
                    push!(quad_aff_vars, aux)  

                elseif haskey(model.variables, i.variable_index_1) && haskey(model.parameters, i.variable_index_2)
                    # Check convention defined above
                    aux = MOI.ScalarQuadraticTerm(i.coefficient, i.variable_index_2, i.variable_index_1)
                    push!(quad_aff_vars, aux)          

                elseif haskey(model.parameters, i.variable_index_1) && haskey(model.parameters, i.variable_index_2)
                    # This is the case where both variable_index_1,2 are actually parameters
                    aux = MOI.ScalarQuadraticTerm(i.coefficient, i.variable_index_1, i.variable_index_2)
                    push!(quad_params, aux)

                else
                    error("Constraint uses a variable that is not in the model")
                end
            end

        else
            quad_terms = f.quadratic_terms
        end

        aff_terms = MOI.ScalarAffineTerm{T}[]

        for i in quad_aff_vars
            aux = MOI.ScalarAffineTerm(i.coefficient * model.parameters[i.variable_index_1], i.variable_index_2)
            push!(aff_terms, aux)
        end

        aff_terms = vcat(aff_terms, aff_vars)

        const_term = f.constant

        for j in aff_params
            const_term += j.coefficient * model.parameters[j.variable_index]
        end

        for j in quad_params
            const_term += j.coefficient * model.parameters[j.variable_index_1] * model.parameters[j.variable_index_2]
        end


        f_quad = MOI.ScalarQuadraticFunction(
                    aff_terms,
                    quad_terms,
                    const_term 
                )

        co = MOI.set(model.optimizer, attr, f_quad)

        model.quadratic_objective_cache_pv[co] = quad_aff_vars #
        model.quadratic_objective_cache_pc[co] = quad_params
        model.quadratic_objective_cache_pp[co] = aff_params

        return
    end
end



end # module

