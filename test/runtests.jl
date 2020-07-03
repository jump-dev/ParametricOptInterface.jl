using Test, ParametricOptInterface, MathOptInterface, GLPK, Ipopt

const POI = ParametricOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

const ATOL = 1e-4

include("production_problem_test.jl")
include("quadratic_tests.jl")


