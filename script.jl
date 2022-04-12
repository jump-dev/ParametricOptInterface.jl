using Revise
using ParametricOptInterface
using Ipopt
using JuMP
using GLPK

function MOIU.default_copy_to(dest::MOI.ModelLike, src::MOI.ModelLike)
    if !MOI.supports_incremental_interface(dest)
        error("Model $(typeof(dest)) does not support copy_to.")
    end
    MOI.empty!(dest)
    vis_src = MOI.get(src, MOI.ListOfVariableIndices())
    index_map = MOI.IndexMap()
    # The `NLPBlock` assumes that the order of variables does not change (#849)
    # Therefore, all VariableIndex and VectorOfVariable constraints are added
    # seprately, and no variables constrained-on-creation are added.
    has_nlp = MOI.NLPBlock() in MOI.get(src, MOI.ListOfModelAttributesSet())
    constraints_not_added = if has_nlp
        Any[
            MOI.get(src, MOI.ListOfConstraintIndices{F,S}()) for
            (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent()) if
            MOIU._is_variable_function(F)
        ]
    else
        Any[
            MOIU._try_constrain_variables_on_creation(dest, src, index_map, S)
            for S in MOIU.sorted_variable_sets_by_cost(dest, src)
        ]
    end
    MOIU._copy_free_variables(dest, index_map, vis_src)
    # Copy variable attributes
    # @show index_map
    MOIU.pass_attributes(dest, src, index_map, vis_src)
    # Copy model attributes
    MOIU.pass_attributes(dest, src, index_map)
    # Copy constraints
    MOIU._pass_constraints(dest, src, index_map, constraints_not_added)
    MOIU.final_touch(dest, index_map)
    return index_map
end

MOI.get(model::Ipopt.Optimizer, attr::MOI.Name) = ""

# m = Model(() -> ParametricOptInterface.Optimizer(Ipopt.Optimizer()));
m = direct_model(ParametricOptInterface.Optimizer(Ipopt.Optimizer()));
MOI.set(m, MOI.Silent(), true)
@variable(m, x >= 0);
@variable(m, y >= 1);
@variable(m, z in ParametricOptInterface.Parameter(10));
@constraint(m, x + y + z >= 0);
@NLobjective(m, Min, x^2 + y^2 + z);
# @objective(m, Min, x + y)
optimize!(m)
objective_value(m)

MOI.set(m, ParametricOptInterface.ParameterValue(), z, 2.0)

optimize!(m);
objective_value(m)

