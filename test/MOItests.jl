using MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities


Q = [4.0 1.0; 1.0 2.0]
q = [1.0; 1.0]
G = [1.0 1.0; 1.0 0.0; 0.0 1.0; -1.0 -1.0; -1.0 0.0; 0.0 -1.0]
h = [1.0; 0.7; 0.7; -1.0; 0.0; 0.0];

model = Ipopt.Optimizer()
x = MOI.add_variables(model, 2)

# define objective
quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
for i in 1:2
    for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
        push!(
            quad_terms, 
            MOI.ScalarQuadraticTerm(Q[i,j],x[i],x[j])
        )
    end
end

objective_function = MOI.ScalarQuadraticFunction(
                        MOI.ScalarAffineTerm.(q, x),
                        quad_terms,
                        0.0
                    )
MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

# add constraints
for i in 1:6
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], x), 0.0),
        MOI.LessThan(h[i])
    )
end

MOI.get(model, MOI.ObjectiveValue())
MOI.get(model, MOI.VariablePrimal(), x)
