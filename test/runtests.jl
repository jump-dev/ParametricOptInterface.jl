using Test
using ParametricOptInterface
using MathOptInterface
using GLPK
using Ipopt
using ECOS
using JuMP

const POI = ParametricOptInterface
const MOIU = MOI.Utilities
const MOIT = MOI.Test

const ATOL = 1e-4

include("MOI_wrapper.jl")
include("production_problem_test.jl")
include("basic_tests.jl")
include("quad_tests.jl")
include("vector_affine_tests.jl")
include("jump_tests.jl")
