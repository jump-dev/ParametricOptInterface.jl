# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using JuMP
using Test

import GLPK
import Ipopt
import SCS

import LinearAlgebra
import ParametricOptInterface

using DiffOpt

const POI = ParametricOptInterface

const ATOL = 1e-4

convex_solver = () -> DiffOpt.diff_optimizer(GLPK.Optimizer)
ipopt_solver = Ipopt.Optimizer

function canonical_compare(f1, f2)
    return MOI.Utilities.canonical(f1) ≈ MOI.Utilities.canonical(f2)
end

include("moi_tests.jl")
include("jump_tests.jl")

for name in names(@__MODULE__; all = true)
    if startswith("$name", "test_")
        @testset "$(name)" begin
            getfield(@__MODULE__, name)(convex_solver, ipopt_solver)
        end
    end
end
