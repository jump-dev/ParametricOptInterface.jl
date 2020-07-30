var documenterSearchIndex = {"docs":
[{"location":"manual/#Manual-1","page":"Manual","title":"Manual","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"","category":"page"},{"location":"manual/#Supported-constraints-1","page":"Manual","title":"Supported constraints","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"This is a list of supported MOI constraint functions that can handle parameters. If you try to add a parameter to  a function that is not listed here, it will return an unsupported error.","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"MOI Function\nScalarAffineFunction\nScalarQuadraticFunction","category":"page"},{"location":"manual/#Supported-objective-functions-1","page":"Manual","title":"Supported objective functions","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"MOI Function\nScalarAffineFunction\nScalarQuadraticFunction","category":"page"},{"location":"manual/#Declare-a-ParametricOptimizer-1","page":"Manual","title":"Declare a ParametricOptimizer","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"in order to use parameters, the user needs to declare a ParametricOptimizer on top of a MOI optimizer, such as GLPK.Optimizer().","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"ParametricOptimizer{T, OT <: MOI.ModelLike} <: MOI.AbstractOptimizer","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"\nusing ParametricOptInterface, MathOptInterface, GLPK\n\n# Rename ParametricOptInterface and MathOptInterface to simplify the code\nconst POI = ParametricOptInterface\nconst MOI = MathOptInterface\n\n# Define a ParametricOptimizer on top of the MOI optimizer\noptimizer = POI.ParametricOptimizer(GLPK.Optimizer())\n","category":"page"},{"location":"manual/#Parameters-1","page":"Manual","title":"Parameters","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"A Parameter is a variable with a fixed value that can be changed by the user.","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"Parameter","category":"page"},{"location":"manual/#Adding-a-new-parameter-to-a-model-1","page":"Manual","title":"Adding a new parameter to a model","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"y, cy = MOI.add_constrained_variable(optimizer, POI.Parameter(0))","category":"page"},{"location":"manual/#Changing-the-parameter-value-1","page":"Manual","title":"Changing the parameter value","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"To change a given parameter's value, access its ConstraintIndex and set it to the new value using the Parameter structure.","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"MOI.set(optimizer, MOI.ConstraintSet(), cy, POI.Parameter(2.0))","category":"page"},{"location":"#ParametricOptInterface.jl-Documentation-1","page":"Home","title":"ParametricOptInterface.jl Documentation","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"ParametricOptInterface.jl (POI for short) is a package written on top of MathOptInterface.jl that allows users to add parameters to a MOI/JuMP problem explicitely.","category":"page"},{"location":"#Installation-1","page":"Home","title":"Installation","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"To install the package you can use Pkg.add it as follows:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"pkg> add ParametricOptInterface","category":"page"},{"location":"#Contributing-1","page":"Home","title":"Contributing","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"When contributing please note that the package follows the JuMP style guide","category":"page"},{"location":"example/#Example-1","page":"Example","title":"Example","text":"","category":"section"},{"location":"example/#","page":"Example","title":"Example","text":"Let us consider the following optimization problem","category":"page"}]
}