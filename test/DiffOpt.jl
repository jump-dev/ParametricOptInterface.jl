using DiffOpt
using ParametricOptInterface

const POI = ParametricOptInterface

using JuMP, DiffOpt, GLPK
# Create a model using the wrapper
solver = () -> DiffOpt.diff_optimizer(GLPK.Optimizer)
model = Model(() -> POI.Optimizer(solver()))

# model = Model(() -> POI.Optimizer(GLPK.Optimizer()))
@variable(model, x)
@variable(model, p in POI.Parameter(1.0))
@constraint(model, cons, x + p >= 3)
@objective(model, Min, 2x)
optimize!(model)
MOI.set(model, POI.ParameterValue(), p, 2.0)
optimize!(model)


model = Model(solver)
MOI.set(model, MOI.TimeLimitSec(), 60 * 1000)
attr = MOI.RawOptimizerAttribute("tm_lim")