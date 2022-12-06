# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using JuMP
using Test

import GLPK
import Ipopt
import SCS

import ParametricOptInterface

const POI = ParametricOptInterface

const ATOL = 1e-4

include("moi_tests.jl")
include("jump_tests.jl")

for name in names(@__MODULE__; all = true)
    if startswith("$name", "test_")
        @testset "$(name)" begin
            getfield(@__MODULE__, name)()
        end
    end
end
