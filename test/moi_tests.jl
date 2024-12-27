# Copyright (c) 2020: Tomás Gutierrez and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function test_basic_tests()
    """
        min x₁ + y
            x₁ + y = 2
            x₁,x₂ ≥ 0

        opt
            x* = {2-y,0}
            obj = 2
    """
    optimizer = POI.Optimizer(GLPK.Optimizer())
    MOI.set(optimizer, MOI.Silent(), true)
    x = MOI.add_variables(optimizer, 2)
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    z = MOI.VariableIndex(4)
    cz = MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{Float64}}(4)
    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end
    @test MOI.is_valid(optimizer, x[1])
    @test MOI.is_valid(optimizer, y)
    @test !MOI.is_valid(optimizer, z)
    @test MOI.is_valid(optimizer, cy)
    @test !MOI.is_valid(optimizer, cz)
    @test_throws ErrorException("Cannot constrain a parameter") MOI.add_constraint(
        optimizer,
        y,
        MOI.EqualTo(0.0),
    )
    @test_throws ErrorException("Variable not in the model") MOI.add_constraint(
        optimizer,
        z,
        MOI.GreaterThan(0.0),
    )
    cons1 = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], [x[1], y]),
        0.0,
    )
    c1 = MOI.add_constraint(optimizer, cons1, MOI.EqualTo(2.0))
    obj_func = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], [x[1], y]),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 2
    @test MOI.get(optimizer, MOI.VariablePrimal(), x[1]) == 2
    @test_throws ErrorException("Variable not in the model") MOI.get(
        optimizer,
        MOI.VariablePrimal(),
        z,
    )
    @test MOI.get(optimizer, POI.ListOfPureVariableIndices()) ==
          MOI.VariableIndex[MOI.VariableIndex(1), MOI.VariableIndex(2)]
    @test MOI.get(optimizer, POI.ListOfParameterIndices()) ==
          POI.ParameterIndex[POI.ParameterIndex(1)]
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(1.0))
    @test_throws ErrorException("Parameter not in the model") MOI.set(
        optimizer,
        MOI.ConstraintSet(),
        cz,
        MOI.Parameter(1.0),
    )
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 2
    @test MOI.get(optimizer, MOI.VariablePrimal(), x[1]) == 1
    """
        min x₁ + x₂
            x₁ + y = 2
            x₁,x₂ ≥ 0

        opt
            x* = {2-y,0}
            obj = 2-y
    """
    new_obj_func = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], [x[1], x[2]]),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        new_obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 1
    @test MOI.supports(optimizer, MOI.VariableName(), MOI.VariableIndex)
    @test MOI.get(optimizer, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    @test MOI.get(optimizer, MOI.VariableName(), x[1]) == ""
    @test MOI.get(optimizer, MOI.ConstraintName(), c1) == ""
    MOI.set(optimizer, MOI.ConstraintName(), c1, "ctr123")
    @test MOI.get(optimizer, MOI.ConstraintName(), c1) == "ctr123"
    return
end

function test_basic_special_cases_of_getters()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in =
        MOI.Utilities.CachingOptimizer(MOI.Utilities.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)
    A = [2.0 1.0; 1.0 2.0]
    a = [1.0, 1.0]
    c = [2.0, 1.0]
    x = MOI.add_variables(optimizer, 2)
    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], x[2], y))
    constraint_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, [x[1], y]),
        0.0,
    )
    cons_index =
        MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(25.0))
    obj_func = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(A[2, 2], x[2], y)],
        MOI.ScalarAffineTerm.(c, [x[1], x[2]]),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    @test MOI.get(optimizer, MOI.ObjectiveFunctionType()) ==
          MOI.ScalarQuadraticFunction{Float64}
    @test MOI.get(optimizer, MOI.NumberOfVariables()) == 3
    return
end

function test_modification_multiple()
    model = POI.Optimizer(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(model, 3)
    saf = MOI.ScalarAffineFunction(
        [
            MOI.ScalarAffineTerm(1.0, x[1]),
            MOI.ScalarAffineTerm(1.0, x[2]),
            MOI.ScalarAffineTerm(1.0, x[3]),
        ],
        0.0,
    )
    ci1 = MOI.add_constraint(model, saf, MOI.LessThan(1.0))
    ci2 = MOI.add_constraint(model, saf, MOI.LessThan(2.0))
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        saf,
    )
    fc1 = MOI.get(model, MOI.ConstraintFunction(), ci1)
    @test MOI.coefficient.(fc1.terms) == [1.0, 1.0, 1.0]
    fc2 = MOI.get(model, MOI.ConstraintFunction(), ci2)
    @test MOI.coefficient.(fc2.terms) == [1.0, 1.0, 1.0]
    obj = MOI.get(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    )
    @test MOI.coefficient.(obj.terms) == [1.0, 1.0, 1.0]
    changes_cis = [
        MOI.ScalarCoefficientChange(MOI.VariableIndex(1), 4.0)
        MOI.ScalarCoefficientChange(MOI.VariableIndex(1), 0.5)
        MOI.ScalarCoefficientChange(MOI.VariableIndex(3), 2.0)
    ]
    MOI.modify(model, [ci1, ci2, ci2], changes_cis)
    fc1 = MOI.get(model, MOI.ConstraintFunction(), ci1)
    @test MOI.coefficient.(fc1.terms) == [4.0, 1.0, 1.0]
    fc2 = MOI.get(model, MOI.ConstraintFunction(), ci2)
    @test MOI.coefficient.(fc2.terms) == [0.5, 1.0, 2.0]
    changes_obj = [
        MOI.ScalarCoefficientChange(MOI.VariableIndex(1), 4.0)
        MOI.ScalarCoefficientChange(MOI.VariableIndex(2), 10.0)
        MOI.ScalarCoefficientChange(MOI.VariableIndex(3), 2.0)
    ]
    MOI.modify(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        changes_obj,
    )

    obj = MOI.get(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    )
    @test MOI.coefficient.(obj.terms) == [4.0, 10.0, 2.0]
    return
end

function test_moi_glpk()
    # TODO see why tests error or fail
    MOI.Test.runtests(
        MOI.Bridges.full_bridge_optimizer(
            POI.Optimizer(GLPK.Optimizer()),
            Float64,
        ),
        MOI.Test.Config();
        exclude = [
            # GLPK returns INVALID_MODEL instead of INFEASIBLE
            "test_constraint_ZeroOne_bounds_3",
            # Upstream issue: https://github.com/jump-dev/MathOptInterface.jl/issues/1431
            "test_model_LowerBoundAlreadySet",
            "test_model_UpperBoundAlreadySet",
        ],
    )
    return
end

function test_moi_ipopt()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.Bridges.full_bridge_optimizer(
            POI.Optimizer(Ipopt.Optimizer()),
            Float64,
        ),
    )
    MOI.set(model, MOI.Silent(), true)
    # Without fixed_variable_treatment set, duals are not computed for variables
    # that have lower_bound == upper_bound.
    MOI.set(
        model,
        MOI.RawOptimizerAttribute("fixed_variable_treatment"),
        "make_constraint",
    )
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            atol = 1e-4,
            rtol = 1e-4,
            optimal_status = MOI.LOCALLY_SOLVED,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.DualObjectiveValue,
                MOI.ObjectiveBound,
            ],
        );
        exclude = String[
            # Tests purposefully excluded:
            #  - Upstream: ZeroBridge does not support ConstraintDual
            "test_conic_linear_VectorOfVariables_2",
            #  - Excluded because this test is optional
            "test_model_ScalarFunctionConstantNotZero",
            #  - Excluded because Ipopt returns NORM_LIMIT instead of
            #    DUAL_INFEASIBLE
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            #  - Excluded because Ipopt returns INVALID_MODEL instead of
            #    LOCALLY_SOLVED
            "test_linear_VectorAffineFunction_empty_row",
            #  - Excluded because Ipopt returns LOCALLY_INFEASIBLE instead of
            #    INFEASIBLE
            "INFEASIBLE",
            "test_conic_linear_INFEASIBLE",
            "test_conic_linear_INFEASIBLE_2",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_",
            #  - Excluded due to upstream issue
            "test_model_LowerBoundAlreadySet",
            "test_model_UpperBoundAlreadySet",
            #  - CachingOptimizer does not throw if optimizer not attached
            "test_model_copy_to_UnsupportedAttribute",
            "test_model_copy_to_UnsupportedConstraint",
            #  - POI throws a ErrorException if user tries to modify parametric
            #    functions
            "test_objective_get_ObjectiveFunction_ScalarAffineFunction",
        ],
    )
    return
end

function test_moi_ListOfConstraintTypesPresent()
    N = 10
    ipopt = Ipopt.Optimizer()
    model = POI.Optimizer(ipopt)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, N / 2)
    y =
        first.(
            MOI.add_constrained_variable.(
                model,
                MOI.Parameter.(ones(Int(N / 2))),
            ),
        )

    MOI.add_constraint(
        model,
        MOI.ScalarQuadraticFunction(
            MOI.ScalarQuadraticTerm.(1.0, x, y),
            MOI.ScalarAffineTerm{Float64}[],
            0.0,
        ),
        MOI.GreaterThan(1.0),
    )
    result = MOI.get(model, MOI.ListOfConstraintTypesPresent())
    expected = [
        (MOI.ScalarQuadraticFunction{Float64}, MOI.GreaterThan{Float64}),
        (MOI.VariableIndex, MOI.Parameter{Float64}),
    ]
    @test Set(result) == Set(expected)
    @test length(result) == length(expected)
    return
end

function test_production_problem_example()
    optimizer = POI.Optimizer(GLPK.Optimizer())
    c = [4.0, 3.0]
    A1 = [2.0, 1.0, 1.0]
    A2 = [1.0, 2.0, 1.0]
    b1 = 4.0
    b2 = 4.0
    x = MOI.add_variables(optimizer, length(c))
    @test typeof(x[1]) == MOI.VariableIndex
    w, cw = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    z, cz = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    @test MOI.get(optimizer, MOI.VariablePrimal(), w) == 0
    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end
    cons1 = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.(A1, [x[1], x[2], y]),
        0.0,
    )
    MOI.add_constraint(optimizer, cons1, MOI.LessThan(b1))
    cons2 = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.(A2, [x[1], x[2], z]),
        0.0,
    )
    MOI.add_constraint(optimizer, cons2, MOI.LessThan(b2))
    @test cons1.terms[1].coefficient == 2
    @test POI._parameter_in_model(optimizer, cons2.terms[3].variable)
    obj_func = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([c[1], c[2], 3.0], [x[1], x[2], w]),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(optimizer)
    MOI.get(optimizer, MOI.TerminationStatus())
    MOI.get(optimizer, MOI.PrimalStatus())
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 28 / 3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4 / 3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 4 / 3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cy), 5 / 3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cz), 2 / 3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cw), -3.0, atol = ATOL)
    MOI.get(optimizer, MOI.VariablePrimal(), w)
    MOI.get(optimizer, MOI.VariablePrimal(), y)
    MOI.get(optimizer, MOI.VariablePrimal(), z)
    MOI.set(optimizer, MOI.ConstraintSet(), cw, MOI.Parameter(2.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(1.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cz, MOI.Parameter(1.0))
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.VariablePrimal(), w) == 2.0
    @test MOI.get(optimizer, MOI.VariablePrimal(), y) == 1.0
    @test MOI.get(optimizer, MOI.VariablePrimal(), z) == 1.0
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 13.0, atol = ATOL)
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [1.0, 1.0]
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cy), 5 / 3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cz), 2 / 3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cw), -3.0, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cw, MOI.Parameter(0.0))
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.VariablePrimal(), w) == 0.0
    @test MOI.get(optimizer, MOI.VariablePrimal(), y) == 1.0
    @test MOI.get(optimizer, MOI.VariablePrimal(), z) == 1.0
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 7, atol = ATOL)
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [1.0, 1.0]
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(-5.0))
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 12.0, atol = ATOL)
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [3.0, 0.0]
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cy), 0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cz), 4, atol = ATOL)
    return
end

function test_production_problem_example_duals()
    optimizer = POI.Optimizer(GLPK.Optimizer())
    c = [4.0, 3.0]
    A1 = [2.0, 1.0, 3.0]
    A2 = [1.0, 2.0, 0.5]
    b1 = 4.0
    b2 = 4.0
    x = MOI.add_variables(optimizer, length(c))
    @test typeof(x[1]) == MOI.VariableIndex
    w, cw = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    z, cz = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    @test MOI.get(optimizer, MOI.VariablePrimal(), w) == 0
    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end
    cons1 = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.(A1, [x[1], x[2], y]),
        0.0,
    )
    ci1 = MOI.add_constraint(optimizer, cons1, MOI.LessThan(b1))
    cons2 = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.(A2, [x[1], x[2], z]),
        0.0,
    )
    ci2 = MOI.add_constraint(optimizer, cons2, MOI.LessThan(b2))
    @test cons1.terms[1].coefficient == 2
    @test POI._parameter_in_model(optimizer, cons2.terms[3].variable)
    obj_func = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([c[1], c[2], 2.0], [x[1], x[2], w]),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(optimizer)
    MOI.get(optimizer, MOI.TerminationStatus())
    MOI.get(optimizer, MOI.PrimalStatus())
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 28 / 3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4 / 3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 4 / 3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cy), 5, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cz), 2 / 6, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cw), -2.0, atol = ATOL)
    @test ≈(
        MOI.get(optimizer, MOI.ConstraintDual(), cy),
        -3 * MOI.get(optimizer, MOI.ConstraintDual(), ci1),
        atol = 1e-4,
    )
    @test ≈(
        MOI.get(optimizer, MOI.ConstraintDual(), cz),
        -0.5 * MOI.get(optimizer, MOI.ConstraintDual(), ci2),
        atol = 1e-4,
    )
    MOI.set(optimizer, MOI.ConstraintSet(), cw, MOI.Parameter(2.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(1.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cz, MOI.Parameter(1.0))
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 7.0, atol = ATOL)
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [0.0, 1.0]
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cy), 9.0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cz), 0.0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cw), -2.0, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cw, MOI.Parameter(0.0))
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 3.0, atol = ATOL)
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [0.0, 1.0]
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cy), 9.0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cz), 0.0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cw), -2.0, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(-5.0))
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 14.0, atol = ATOL)
    @test ≈(
        MOI.get.(optimizer, MOI.VariablePrimal(), x),
        [3.5, 0.0],
        atol = ATOL,
    )
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cy), 0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cz), 2, atol = ATOL)
    return
end

function test_production_problem_example_parameters_for_duals_and_intervals()
    cached = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            GLPK.Optimizer(),
        ),
        Float64,
    )
    optimizer = POI.Optimizer(cached)
    c = [4.0, 3.0]
    A1 = [2.0, 1.0, 3.0]
    A2 = [1.0, 2.0, 0.5]
    b1 = 4.0
    b2 = 4.0
    x = MOI.add_variables(optimizer, length(c))
    @test typeof(x[1]) == MOI.VariableIndex
    w, cw = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    z, cz = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    @test MOI.get(optimizer, MOI.VariablePrimal(), w) == 0
    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end
    cons1 = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.(A1, [x[1], x[2], y]),
        0.0,
    )
    ci1 = MOI.add_constraint(optimizer, cons1, MOI.Interval(-Inf, b1))
    cons2 = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.(A2, [x[1], x[2], z]),
        0.0,
    )
    ci2 = MOI.add_constraint(optimizer, cons2, MOI.LessThan(b2))
    @test cons1.terms[1].coefficient == 2
    @test POI._parameter_in_model(optimizer, cons2.terms[3].variable)
    obj_func = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([c[1], c[2], 2.0], [x[1], x[2], w]),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(optimizer)
    MOI.get(optimizer, MOI.TerminationStatus())
    MOI.get(optimizer, MOI.PrimalStatus())
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 28 / 3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4 / 3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 4 / 3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cy), 5, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cz), 2 / 6, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cw), -2.0, atol = ATOL)
    @test ≈(
        MOI.get(optimizer, MOI.ConstraintDual(), cy),
        -3 * MOI.get(optimizer, MOI.ConstraintDual(), ci1),
        atol = 1e-4,
    )
    @test ≈(
        MOI.get(optimizer, MOI.ConstraintDual(), cz),
        -0.5 * MOI.get(optimizer, MOI.ConstraintDual(), ci2),
        atol = 1e-4,
    )
    MOI.set(optimizer, MOI.ConstraintSet(), cw, MOI.Parameter(2.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(1.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cz, MOI.Parameter(1.0))
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 7.0, atol = ATOL)
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [0.0, 1.0]
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cy), 9.0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cz), 0.0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cw), -2.0, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cw, MOI.Parameter(0.0))
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 3.0, atol = ATOL)
    @test MOI.get.(optimizer, MOI.VariablePrimal(), x) == [0.0, 1.0]
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cy), 9.0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cz), 0.0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cw), -2.0, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(-5.0))
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 14.0, atol = ATOL)
    @test ≈(
        MOI.get.(optimizer, MOI.VariablePrimal(), x),
        [3.5, 0.0],
        atol = ATOL,
    )
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cy), 0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.ConstraintDual(), cz), 2, atol = ATOL)
    return
end

function test_vector_parameter_affine_nonnegatives()
    """
        min x + y
            x - t + 1 >= 0
            y - t + 2 >= 0

        opt
            x* = t-1
            y* = t-2
            obj = 2*t-3
    """
    cached = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        SCS.Optimizer(),
    )
    model = POI.Optimizer(cached)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    t, ct = MOI.add_constrained_variable(model, MOI.Parameter(5.0))
    A = [1.0 0 -1; 0 1 -1]
    b = [1.0; 2]
    terms =
        MOI.VectorAffineTerm.(
            1:2,
            MOI.ScalarAffineTerm.(A, reshape([x, y, t], 1, 3)),
        )
    f = MOI.VectorAffineFunction(vec(terms), b)
    set = MOI.Nonnegatives(2)
    cnn = MOI.add_constraint(model, f, MOI.Nonnegatives(2))
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], [y, x]),
            0.0,
        ),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 3 atol = ATOL
    @test MOI.get(model, MOI.ConstraintPrimal(), cnn) ≈ [0.0, 0.0] atol = ATOL
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 7 atol = ATOL
    @test MOI.get(model, MOI.DualObjectiveValue()) ≈ 7 atol = ATOL
    @test MOI.get(model, MOI.ConstraintDual(), cnn) ≈ [1.0, 1.0] atol = ATOL
    MOI.set(model, POI.ParameterValue(), t, 6)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 5 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 atol = ATOL
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 9 atol = ATOL
    return
end

function test_vector_parameter_affine_nonpositives()
    """
        min x + y
            - x + t - 1 ≤ 0
            - y + t - 2 ≤ 0

        opt
            x* = t-1
            y* = t-2
            obj = 2*t-3
    """
    cached = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            SCS.Optimizer(),
        ),
        Float64,
    )
    model = POI.Optimizer(cached)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    t, ct = MOI.add_constrained_variable(model, MOI.Parameter(5.0))
    A = [-1.0 0 1; 0 -1 1]
    b = [-1.0; -2]
    terms =
        MOI.VectorAffineTerm.(
            1:2,
            MOI.ScalarAffineTerm.(A, reshape([x, y, t], 1, 3)),
        )
    f = MOI.VectorAffineFunction(vec(terms), b)
    set = MOI.Nonnegatives(2)
    cnn = MOI.add_constraint(model, f, MOI.Nonpositives(2))
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1.0, 1.0], [y, x]),
            0.0,
        ),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 3 atol = ATOL
    @test MOI.get(model, MOI.ConstraintPrimal(), cnn) ≈ [0.0, 0.0] atol = ATOL
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 7 atol = ATOL
    @test MOI.get(model, MOI.DualObjectiveValue()) ≈ 7 atol = ATOL
    @test MOI.get(model, MOI.ConstraintDual(), cnn) ≈ [-1.0, -1.0] atol = ATOL
    MOI.set(model, POI.ParameterValue(), t, 6)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 5 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 atol = ATOL
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 9 atol = ATOL
    return
end

function test_vector_soc_parameters()
    """
        Problem SOC2 from MOI

        min  x
        s.t. y ≥ 1/√2
            (x-p)² + y² ≤ 1

        in conic form:

        min  x
        s.t.  -1/√2 + y ∈ R₊
            1 - t ∈ {0}
            (t, x-p ,y) ∈ SOC₃

        opt
            x* = p - 1/√2
            y* = 1/√2
    """
    cached = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        SCS.Optimizer(),
    )
    model = POI.Optimizer(cached)
    MOI.set(model, MOI.Silent(), true)
    x, y, t = MOI.add_variables(model, 3)
    p, cp = MOI.add_constrained_variable(model, MOI.Parameter(0.0))
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    cnon = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, y))],
            [-1 / √2],
        ),
        MOI.Nonnegatives(1),
    )
    ceq = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(-1.0, t))],
            [1.0],
        ),
        MOI.Zeros(1),
    )
    A = [
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 -1
        0.0 0.0 1.0 0.0
    ]
    f = MOI.VectorAffineFunction(
        vec(
            MOI.VectorAffineTerm.(
                1:3,
                MOI.ScalarAffineTerm.(A, reshape([t, x, y, p], 1, 4)),
            ),
        ),
        zeros(3),
    )
    csoc = MOI.add_constraint(model, f, MOI.SecondOrderCone(3))
    f_error = MOI.VectorOfVariables([t, p, y])
    @test_throws ErrorException MOI.add_constraint(
        model,
        f_error,
        MOI.SecondOrderCone(3),
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ -1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ -1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), t) ≈ 1 atol = ATOL
    MOI.set(model, POI.ParameterValue(), p, 1)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1 - 1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 1 - 1 / √2 atol = ATOL
    return
end

# TODO(odow): What is this doing here!!!
function test_vector_soc_no_parameters()
    """
        Problem SOC2 from MOI

        min  x
        s.t. y ≥ 1/√2
            x² + y² ≤ 1

        in conic form:

        min  x
        s.t.  -1/√2 + y ∈ R₊
            1 - t ∈ {0}
            (t, x ,y) ∈ SOC₃

        opt
            x* = 1/√2
            y* = 1/√2
    """
    cached = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            SCS.Optimizer(),
        ),
        Float64,
    )
    model = POI.Optimizer(cached)
    MOI.set(model, MOI.Silent(), true)
    x, y, t = MOI.add_variables(model, 3)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    cnon = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, y))],
            [-1 / √2],
        ),
        MOI.Nonnegatives(1),
    )
    ceq = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(-1.0, t))],
            [1.0],
        ),
        MOI.Zeros(1),
    )
    f = MOI.VectorOfVariables([t, x, y])
    csoc = MOI.add_constraint(model, f, MOI.SecondOrderCone(3))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ -1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ -1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 1 / √2 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), t) ≈ 1 atol = ATOL
    return
end

function test_qp_no_parameters_1()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in =
        MOI.Utilities.CachingOptimizer(MOI.Utilities.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)
    Q = [4.0 1.0; 1.0 2.0]
    q = [1.0; 1.0]
    G = [1.0 1.0; 1.0 0.0; 0.0 1.0; -1.0 -1.0; -1.0 0.0; 0.0 -1.0]
    h = [1.0; 0.7; 0.7; -1.0; 0.0; 0.0]
    x = MOI.add_variables(optimizer, 2)
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
        end
    end
    objective_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(q, x),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    for i in 1:6
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i, :], x), 0.0),
            MOI.LessThan(h[i]),
        )
    end
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 1.88, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 0.3, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 0.7, atol = ATOL)
    return
end

function test_qp_no_parameters_2()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in =
        MOI.Utilities.CachingOptimizer(MOI.Utilities.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)
    A = [0.0 1.0; 1.0 0.0]
    a = [0.0, 0.0]
    c = [2.0, 1.0]
    x = MOI.add_variables(optimizer, 2)
    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(1.0))
        MOI.add_constraint(optimizer, x_i, MOI.LessThan(5.0))
    end
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2]))
    constraint_function = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2])],
        MOI.ScalarAffineTerm.(a, x),
        0.0,
    )
    MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(9.0))
    obj_func =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 11.8, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 5.0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 1.8, atol = ATOL)
    return
end

function test_qp_parameter_in_affine_constraint()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in =
        MOI.Utilities.CachingOptimizer(MOI.Utilities.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)
    Q = [3.0 2.0; 2.0 1.0]
    q = [1.0, 6.0]
    G = [2.0 3.0 1.0; 1.0 1.0 1.0]
    h = [4.0; 3.0]
    x = MOI.add_variables(optimizer, 2)
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
        end
    end
    objective_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(q, x),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    for i in 1:2
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.(G[i, :], [x[1], x[2], y]),
                0.0,
            ),
            MOI.GreaterThan(h[i]),
        )
    end
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 12.5, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 5.0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), -2.0, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(1.0))
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 5.0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 3.0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), -1.0, atol = ATOL)
    return
end

function test_qp_parameter_in_quadratic_constraint()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in =
        MOI.Utilities.CachingOptimizer(MOI.Utilities.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)
    Q = [3.0 2.0; 2.0 1.0]
    q = [1.0, 6.0, 1.0]
    G = [2.0 3.0 1.0 0.0; 1.0 1.0 0.0 1.0]
    h = [4.0; 3.0]
    x = MOI.add_variables(optimizer, 2)
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    w, cw = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
        end
    end
    objective_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(q, [x[1], x[2], y]),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.(G[1, :], [x[1], x[2], y, w]),
            0.0,
        ),
        MOI.GreaterThan(h[1]),
    )
    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.(G[2, :], [x[1], x[2], y, w]),
            0.0,
        ),
        MOI.GreaterThan(h[2]),
    )
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 12.5, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 5.0, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), -2.0, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(1.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cw, MOI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 5.7142, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 2.1428, atol = ATOL)
    @test ≈(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        -0.4285,
        atol = ATOL,
    )
    return
end

function test_qp_variable_times_variable_plus_parameter()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in =
        MOI.Utilities.CachingOptimizer(MOI.Utilities.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)
    A = [2.0 1.0; 1.0 2.0]
    a = [1.0, 1.0]
    c = [2.0, 1.0]
    x = MOI.add_variables(optimizer, 2)
    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], x[2], x[2]))
    constraint_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, [x[1], y]),
        0.0,
    )
    cons_index =
        MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(25.0))
    obj_func =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 9.0664, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4.3665, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 1 / 3, atol = ATOL)
    @test ≈(
        MOI.get(optimizer, MOI.ConstraintDual(), cy),
        -MOI.get(optimizer, MOI.ConstraintDual(), cons_index),
        atol = ATOL,
    )
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 8.6609, atol = ATOL)
    return
end

function test_qp_variable_times_variable_plus_parameter_duals()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in =
        MOI.Utilities.CachingOptimizer(MOI.Utilities.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)
    A = [2.0 1.0; 1.0 2.0]
    a = [1.0, 2.0]
    c = [2.0, 1.0]
    x = MOI.add_variables(optimizer, 2)
    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], x[2], x[2]))
    constraint_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, [x[1], y]),
        0.0,
    )
    cons_index =
        MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(25.0))

    obj_func =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 9.0664, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 4.3665, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 1 / 3, atol = ATOL)

    @test ≈(
        MOI.get(optimizer, MOI.ConstraintDual(), cy),
        -2 * MOI.get(optimizer, MOI.ConstraintDual(), cons_index),
        atol = ATOL,
    )
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 8.2376, atol = ATOL)
    return
end

function test_qp_parameter_times_variable()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in =
        MOI.Utilities.CachingOptimizer(MOI.Utilities.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)
    A = [2.0 1.0; 1.0 2.0]
    a = [1.0, 1.0]
    c = [2.0, 1.0]
    x = MOI.add_variables(optimizer, 2)
    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end
    MOI.add_constraint(optimizer, x[1], MOI.LessThan(20.0))
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], y, y))
    constraint_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, x),
        0.0,
    )
    MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(30.0))
    obj_func =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 30.25, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 0.5, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 29.25, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(2.0))
    # this implies: x1 + x2 + x1^2 + x1*y + y^2 <= 30
    # becomes: 2 * x1 + x2 + x1^2 <= 30 - 4 = 26
    # then x1 = 0 and x2 = 26 and obj = 26
    # is x1 = eps >= 0, x2 = 26 - 2 * eps - eps^2 and
    # obj = 26 - 2 * eps - eps^2 + 2 * eps = 26 - eps ^2 (eps == 0 to maximize)
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 26.0, atol = ATOL)
    return
end

function test_qp_variable_times_parameter()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in =
        MOI.Utilities.CachingOptimizer(MOI.Utilities.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)
    A = [2.0 1.0; 1.0 2.0]
    a = [1.0, 1.0]
    c = [2.0, 1.0]
    x = MOI.add_variables(optimizer, 2)
    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end
    MOI.add_constraint(optimizer, x[1], MOI.LessThan(20.0))
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], y, y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], y, x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))
    constraint_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, x),
        0.0,
    )
    MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(30.0))
    obj_func =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 30.25, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 0.5, atol = ATOL)
    @test ≈(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), 29.25, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(2.0))
    MOI.optimize!(optimizer)
    # this implies: x1 + x2 + x1^2 + x1*y + y^2 <= 30
    # becomes: 2 * x1 + x2 + x1^2 <= 30 - 4 = 26
    # then x1 = 0 and x2 = 26 and obj = 26
    # is x1 = eps >= 0, x2 = 26 - 2 * eps - eps^2 and
    # obj = 26 - 2 * eps - eps^2 + 2 * eps = 26 - eps ^2 (eps == 0 to maximize)
    MOI.optimize!(optimizer)
    @test ≈(MOI.get(optimizer, MOI.ObjectiveValue()), 26.0, atol = ATOL)
    return
end

function test_qp_parameter_times_parameter()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in =
        MOI.Utilities.CachingOptimizer(MOI.Utilities.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)
    A = [2.0 1.0; 1.0 2.0]
    a = [1.0, 1.0]
    c = [2.0, 1.0]
    x = MOI.add_variables(optimizer, 2)
    # x1 >= 0, x2 >= 0
    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end
    # x1 <= 20
    MOI.add_constraint(optimizer, x[1], MOI.LessThan(20.0))
    # y == 0
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    # z == 0
    z, cz = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    # x1 + x2 + y^2 + yz + z^2 <= 30
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], y, y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], y, z))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], z, z))
    constraint_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, x),
        0.0,
    )
    MOI.add_constraint(optimizer, constraint_function, MOI.LessThan(30.0))
    # max 2x1 + x2
    obj_func =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, [x[1], x[2]]), 0.0)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 50.0, atol = ATOL)
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        20.0,
        atol = ATOL,
    )
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        10.0,
        atol = ATOL,
    )
    # now x1 + x2 + y^2 + yz + z^2 <= 30
    # implies x1 + x2 <= 26
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(2.0))
    MOI.optimize!(optimizer)
    # hence x1 = 20, x2 = 6
    # and obj = 46
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 46.0, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cz, MOI.Parameter(1.0))
    # now x1 + x2 + y^2 + yz + z^2 <= 30
    # implies x1 + x2 <= 30 - 4 - 2 - 1 = 23
    # hence x1 = 20, x2 = 3
    # and obj = 43
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 43.0, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(-1.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cz, MOI.Parameter(-1.0))
    # now x1 + x2 + y^2 + yz + z^2 <= 30
    # implies x1 + x2 <= 30 -1 -1 -1 = 27
    # hence x1 = 20, x2 = 7
    # and obj = 47
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 47.0, atol = ATOL)
    return
end

function test_qp_quadratic_constant()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in =
        MOI.Utilities.CachingOptimizer(MOI.Utilities.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)
    Q = [3.0 2.0 0.0; 2.0 1.0 0.0; 0.0 0.0 1.0]
    q = [1.0, 6.0, 0.0]
    G = [2.0 3.0; 1.0 1.0]
    h = [4.0; 3.0]
    x = MOI.add_variables(optimizer, 2)
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(0.0))
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
        end
    end
    push!(quad_terms, MOI.ScalarQuadraticTerm(Q[1, 3], x[1], y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(Q[2, 3], x[2], y))
    push!(quad_terms, MOI.ScalarQuadraticTerm(Q[3, 3], y, y))
    objective_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(q, [x[1], x[2], y]),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1, :], x), 0.0),
        MOI.GreaterThan(h[1]),
    )
    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[2, :], x), 0.0),
        MOI.GreaterThan(h[2]),
    )
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 12.5, atol = ATOL)
    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        5.0,
        atol = ATOL,
    )
    @test isapprox.(
        MOI.get(optimizer, MOI.VariablePrimal(), x[2]),
        -2.0,
        atol = ATOL,
    )
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(1.0))
    MOI.optimize!(optimizer)
    # @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 5.7142, atol = ATOL)
    # @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[1]), 2.1428, atol = ATOL)
    # @test isapprox.(MOI.get(optimizer, MOI.VariablePrimal(), x[2]), -0.4285, atol = ATOL)
    return
end

function test_qp_objective_parameter_times_parameter()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in =
        MOI.Utilities.CachingOptimizer(MOI.Utilities.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)
    a = [1.0, 1.0]
    x = MOI.add_variables(optimizer, 2)
    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(1.0))
    z, cz = MOI.add_constrained_variable(optimizer, MOI.Parameter(1.0))
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    push!(quad_terms, MOI.ScalarQuadraticTerm(1.0, y, z))
    objective_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, x),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 1.0, atol = ATOL)
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        0.0,
        atol = ATOL,
    )
    err =
        ErrorException("Cannot compute the dual of a multiplicative parameter")
    @test_throws err MOI.get(optimizer, MOI.ConstraintDual(), cy)
    @test_throws err MOI.get(optimizer, MOI.ConstraintDual(), cz)
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 2.0, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cz, MOI.Parameter(3.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 6.0, atol = ATOL)
    MOI.set(optimizer, POI.ParameterValue(), y, 5)
    MOI.set(optimizer, POI.ParameterValue(), z, 5.0)
    @test_throws ErrorException MOI.set(
        optimizer,
        POI.ParameterValue(),
        MOI.VariableIndex(10872368175),
        5.0,
    )
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 25.0, atol = ATOL)
end

function test_qp_objective_affine_parameter()
    ipopt = Ipopt.Optimizer()
    MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
    opt_in =
        MOI.Utilities.CachingOptimizer(MOI.Utilities.Model{Float64}(), ipopt)
    optimizer = POI.Optimizer(opt_in)
    A = [0.0 1.0; 1.0 0.0]
    a = [2.0, 1.0]
    x = MOI.add_variables(optimizer, 2)
    for x_i in x
        MOI.add_constraint(optimizer, x_i, MOI.GreaterThan(0.0))
    end
    y, cy = MOI.add_constrained_variable(optimizer, MOI.Parameter(1.0))
    z, cz = MOI.add_constrained_variable(optimizer, MOI.Parameter(1.0))
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 1], x[1], x[1]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[1, 2], x[1], x[2]))
    push!(quad_terms, MOI.ScalarQuadraticTerm(A[2, 2], x[2], x[2]))
    objective_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(a, [y, z]),
        0.0,
    )
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 3.0, atol = ATOL)
    @test isapprox(
        MOI.get(optimizer, MOI.VariablePrimal(), x[1]),
        0,
        atol = ATOL,
    )
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(2.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 5.0, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cz, MOI.Parameter(3.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 7.0, atol = ATOL)
    MOI.set(optimizer, MOI.ConstraintSet(), cy, MOI.Parameter(5.0))
    MOI.set(optimizer, MOI.ConstraintSet(), cz, MOI.Parameter(5.0))
    MOI.optimize!(optimizer)
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 15.0, atol = ATOL)
    return
end

function test_qp_objective_parameter_in_quadratic_part()
    model = POI.Optimizer(Ipopt.Optimizer())
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    z = MOI.add_variable(model)
    p = first(MOI.add_constrained_variable.(model, MOI.Parameter(1.0)))
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))
    cons1 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 1.0], [x, y]), 0.0)
    ci1 = MOI.add_constraint(model, cons1, MOI.LessThan(4.0))
    cons2 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0], [x, y]), 0.0)
    ci2 = MOI.add_constraint(model, cons2, MOI.LessThan(4.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    obj_func = MOI.ScalarQuadraticFunction(
        [
            MOI.ScalarQuadraticTerm(1.0, x, x)
            MOI.ScalarQuadraticTerm(1.0, y, y)
        ],
        MOI.ScalarAffineTerm{Float64}[],
        0.0,
    )
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        obj_func,
    )
    MOI.set(model, POI.QuadraticObjectiveCoef(), (x, y), 2p + 3)
    @test MOI.get(model, POI.QuadraticObjectiveCoef(), (x, y)) ≈
          MOI.ScalarAffineFunction{Int64}(
        MOI.ScalarAffineTerm{Int64}[MOI.ScalarAffineTerm{Int64}(
            2,
            MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1),
        )],
        3,
    )
    @test_throws ErrorException MOI.get(
        model,
        POI.QuadraticObjectiveCoef(),
        (x, z),
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 32 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL
    MOI.set(model, POI.ParameterValue(), p, 2.0)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 128 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL
    model = POI.Optimizer(Ipopt.Optimizer())
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    p = first(MOI.add_constrained_variable.(model, MOI.Parameter(1.0)))
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))
    cons1 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 1.0], [x, y]), 0.0)
    ci1 = MOI.add_constraint(model, cons1, MOI.LessThan(4.0))
    cons2 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0], [x, y]), 0.0)
    ci2 = MOI.add_constraint(model, cons2, MOI.LessThan(4.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    obj_func = MOI.ScalarAffineFunction(
        [
            MOI.ScalarAffineTerm(1.0, x)
            MOI.ScalarAffineTerm(2.0, y)
        ],
        1.0,
    )
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        obj_func,
    )
    MOI.set(model, POI.QuadraticObjectiveCoef(), (x, y), p)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 61 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, POI.QuadraticObjectiveCoef(), (x, y)) ≈
          MOI.VariableIndex(POI.PARAMETER_INDEX_THRESHOLD + 1)
    MOI.set(model, POI.ParameterValue(), p, 2.0)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 77 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL
    model = POI.Optimizer(Ipopt.Optimizer())
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    p = first(MOI.add_constrained_variable.(model, MOI.Parameter(1.0)))
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))
    cons1 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 1.0], [x, y]), 0.0)
    ci1 = MOI.add_constraint(model, cons1, MOI.LessThan(4.0))
    cons2 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0], [x, y]), 0.0)
    ci2 = MOI.add_constraint(model, cons2, MOI.LessThan(4.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    obj_func = x
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), obj_func)
    MOI.set(model, POI.QuadraticObjectiveCoef(), (x, y), p)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 28 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL
    MOI.set(model, POI.ParameterValue(), p, 2.0)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 44 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL
    model = POI.Optimizer(Ipopt.Optimizer())
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    p = first(MOI.add_constrained_variable.(model, MOI.Parameter(1.0)))
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))
    cons1 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 1.0], [x, y]), 0.0)
    ci1 = MOI.add_constraint(model, cons1, MOI.LessThan(4.0))
    cons2 =
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0], [x, y]), 0.0)
    ci2 = MOI.add_constraint(model, cons2, MOI.LessThan(4.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, POI.QuadraticObjectiveCoef(), (x, y), p)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 16 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL
    MOI.set(model, POI.ParameterValue(), p, 2.0)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 32 / 9 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 4 / 3 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 4 / 3 atol = ATOL
    return
end
