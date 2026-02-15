# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Test

@testset "$file" for file in filter!(startswith("test_"), readdir(@__DIR__))
    include(joinpath(@__DIR__, file))
end
