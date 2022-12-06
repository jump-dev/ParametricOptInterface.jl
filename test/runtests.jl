# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Test
using ParametricOptInterface
using MathOptInterface
using GLPK
using Ipopt
using ECOS
using SCS
using LinearAlgebra
using JuMP

const POI = ParametricOptInterface
const MOIU = MOI.Utilities
const MOIT = MOI.Test

const ATOL = 1e-4

include("production_problem_test.jl")
include("MOI_wrapper.jl")
include("basic_tests.jl")
include("dual_tests.jl")
include("quad_tests.jl")
include("sdp_tests.jl")
include("vector_affine_tests.jl")
include("modifications_tests.jl")
include("jump_tests.jl")
include("nlp_test.jl")
