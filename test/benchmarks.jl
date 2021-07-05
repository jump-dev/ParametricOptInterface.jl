# TODO we should implement benchmarks for every method implemented
function add_constrained_variable(n::Int, opt)
    optimizer = POI.ParametricOptimizer(opt)
    for _ in 1:n
        MOI.add_constrained_variable(optimizer, POI.Parameter(0))
    end
    return
end

function add_constraint_svf(n::Int, opt)
    optimizer = POI.ParametricOptimizer(opt)
    x = MOI.add_variables(optimizer, n)
    for i in 1:n
        MOI.add_constraint(optimizer, MOI.SingleVariable(x[i]), MOI.GreaterThan(1.0))
    end
    return
end

function add_constraint_saf(n::Int, opt)
    optimizer = POI.ParametricOptimizer(opt)
    x = MOI.add_variables(optimizer, n)
    for i in 1:n
        y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))
        con = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 0.5], [x[i], y]), 0.0)
        MOI.add_constraint(optimizer, con, MOI.GreaterThan(1.0))
    end
    return
end

n = 1000
@benchmark add_constrained_variable($n, GLPK.Optimizer())
@benchmark add_constraint_svf($n, GLPK.Optimizer())
@benchmark add_constraint_saf($n, GLPK.Optimizer())
