using Test

import ParametricOptInterface as POI

for file in readdir(@__DIR__)
    if file in ["runtests.jl"]
        continue
    end
    @testset "$(file)" begin
        include(file)
    end
end
