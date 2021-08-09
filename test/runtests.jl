using Test
using ParametricOptInterface
using MathOptInterface
using GLPK
using Ipopt
using ECOS
using JuMP

const POI = ParametricOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

const ATOL = 1e-4

include("production_problem_test.jl")
include("basic_tests.jl")
include("quad_tests.jl")
include("vector_affine_tests.jl")
include("jump_tests.jl")

