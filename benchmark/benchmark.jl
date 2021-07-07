push!(LOAD_PATH, "./src")
using ParametricOptInterface, MathOptInterface, GLPK

const POI = ParametricOptInterface
const MOI = MathOptInterface

MOI.Benchmarks.@add_benchmark function add_constrained_variable(new_model)
    model = new_model()
    for _ in 1:10_000
        MOI.add_constrained_variable(model, POI.Parameter(0))
    end
    return
end

MOI.Benchmarks.@add_benchmark function add_constraint_svf(new_model)
    model = new_model()
    x = MOI.add_variables(model, 10_000)
    for i in 1:10_000
        MOI.add_constraint(model, MOI.SingleVariable(x[i]), MOI.GreaterThan(1.0))
    end
    return
end

MOI.Benchmarks.@add_benchmark function add_constraint_saf(new_model)
    model = new_model()
    x = MOI.add_variables(model, 10_000)
    for i in 1:10_000
        y, cy = MOI.add_constrained_variable(model, POI.Parameter(0))
        con = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 0.5], [x[i], y]), 0.0)
        MOI.add_constraint(model, con, MOI.GreaterThan(1.0))
    end
    return
end

function print_help()
    println("""
    Usage
        benchmark.jl [arg] [name]
    [arg]
        --new       Begin a new benchmark comparison
        --compare   Run another benchmark and compare to existing
    [name]          A name for the benchmark test
    Examples
        git checkout master
        julia benchmark.jl --new master
        git checkout approach_1
        julia benchmark.jl --new approach_1
        git checkout approach_2
        julia benchmark.jl --compare master
        julia benchmark.jl --compare approach_1
    """)
end

if length(ARGS) != 2
    print_help()
else
    const Benchmarks = GLPK.MOI.Benchmarks
    const suite = Benchmarks.suite(() -> POI.ParametricOptimizer(GLPK.Optimizer()); exclude = [r"delete", r"copy"])
    if ARGS[1] == "--new"
        Benchmarks.create_baseline(
            suite, ARGS[2]; directory = @__DIR__, verbose = true
        )
    elseif ARGS[1] == "--compare"
        Benchmarks.compare_against_baseline(
            suite, ARGS[2]; directory = @__DIR__, verbose = true
        )
    else
        print_help()
    end
end