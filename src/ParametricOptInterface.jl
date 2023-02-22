module ParametricOptInterface

import MathOptInterface as MOI

include("optimizer.jl")
include("variables.jl")
include("constraints.jl")
include("checks_for_parameters.jl")

end