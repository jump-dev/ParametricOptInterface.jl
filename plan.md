# Implementation Plan: Parametric Cubic Functions in POI

Note: do not change this plan during implementation. If changes are needed, create a new version of the plan called plan_v2.md to track changes to the plan.

## Overview

This document details the implementation plan for supporting parameters multiplying quadratic terms in ParametricOptInterface (POI).

### The Problem

We want to support expressions of the form:

```
c * x * y * p
```

Where:
- `c` = coefficient (constant)
- `x`, `y` = decision variables
- `p` = parameter

This creates a **cubic function** since a parameter behaves like a decision variable in the expression structure. However, MOI does not have a native cubic function type.

### Current State

POI currently supports:
- `ParametricAffineFunction`: handles `c + c*x + c*p + c*x*p`
- `ParametricQuadraticFunction`: handles `c + c*x + c*p + c*x*y + c*p*x + c*p*q`

The missing pieces are:
- `c*x*y*p` - one parameter multiplying a quadratic term (becomes quadratic after substitution)
- `c*p1*p2*x` - two parameters multiplying a variable (becomes affine after substitution)

### Approach

Since MOI lacks cubic functions, users must express `c*x*y*p` as a `ScalarNonlinearFunction`. We will:
1. Parse `ScalarNonlinearFunction` to detect if it represents exactly a cubic polynomial
2. Store the parsed data in a new `ParametricCubicFunction` structure
3. Support this **only in objectives** (not constraints)

---

## Part 1: Understanding MOI.ScalarNonlinearFunction

### Structure

```julia
MOI.ScalarNonlinearFunction(
    head::Symbol,        # Operator (:+, :*, :^, etc.)
    args::Vector{Any}    # Operands (constants, variables, nested expressions)
)
```

### Expression Tree Examples

**Example 1**: `2 * x * y * p` could be represented as:
```julia
ScalarNonlinearFunction(:*, Any[2.0, x, y, p])
```

**Example 2**: `x * y * p + 3 * x * z * q` (sum of cubic terms):
```julia
ScalarNonlinearFunction(:+, Any[
    ScalarNonlinearFunction(:*, Any[x, y, p]),
    ScalarNonlinearFunction(:*, Any[3.0, x, z, q])
])
```

### Parsing Strategy

We need to recursively traverse the expression tree and:
1. Identify if all operations are `+`, `*`, `-`, or `^` with integer exponents
2. Expand products into monomials
3. Classify each monomial by degree in variables and parameters
4. Reject if any monomial exceeds cubic degree

---

## Part 1.5: Parsing Corner Cases (Critical)

The parser must handle various equivalent representations of the same mathematical expression. This section documents the corner cases that **must** be tested and handled correctly.

### 1.5.1 Mixed Parenthesis Orderings

The same expression `c * x * y * p` can have many different tree structures:

**Flat multiplication:**
```julia
# 2 * x * y * p as a single :* node with 4 children
ScalarNonlinearFunction(:*, Any[2.0, x, y, p])
```

**Left-associative (typical from parsing `((2*x)*y)*p`):**
```julia
ScalarNonlinearFunction(:*, Any[
    ScalarNonlinearFunction(:*, Any[
        ScalarNonlinearFunction(:*, Any[2.0, x]),
        y
    ]),
    p
])
```

**Right-associative (`2*(x*(y*p))`):**
```julia
ScalarNonlinearFunction(:*, Any[
    2.0,
    ScalarNonlinearFunction(:*, Any[
        x,
        ScalarNonlinearFunction(:*, Any[y, p])
    ])
])
```

**Mixed groupings (`(2*x) * (y*p)`):**
```julia
ScalarNonlinearFunction(:*, Any[
    ScalarNonlinearFunction(:*, Any[2.0, x]),
    ScalarNonlinearFunction(:*, Any[y, p])
])
```

**Coefficient grouped with parameter (`(2*p) * (x*y)`):**
```julia
ScalarNonlinearFunction(:*, Any[
    ScalarNonlinearFunction(:*, Any[2.0, p]),
    ScalarNonlinearFunction(:*, Any[x, y])
])
```

**Parser requirement**: All of the above must produce the same result: `_ScalarCubicTerm(2.0, p, x, y)` with type `:pvv`.

### 1.5.2 Squared Variables: `p * c * x^2`

This is a valid cubic term where `variable_1 == variable_2`. Representations:

**Using power operator:**
```julia
# 3 * p * x^2
ScalarNonlinearFunction(:*, Any[
    3.0,
    p,
    ScalarNonlinearFunction(:^, Any[x, 2])
])
```

**Using explicit multiplication:**
```julia
# 3 * p * x * x
ScalarNonlinearFunction(:*, Any[3.0, p, x, x])
```

**Nested with power:**
```julia
# (3*p) * x^2
ScalarNonlinearFunction(:*, Any[
    ScalarNonlinearFunction(:*, Any[3.0, p]),
    ScalarNonlinearFunction(:^, Any[x, 2])
])
```

**Parser requirement**: All must produce `_ScalarCubicTerm(3.0, p, x, x)` with type `:pvv`.

### 1.5.3 Power Operator Variations

**x^2 (valid quadratic):**
```julia
ScalarNonlinearFunction(:^, Any[x, 2])
# Should expand to: x * x (two variables)
```

**p^2 (valid quadratic in parameters):**
```julia
ScalarNonlinearFunction(:^, Any[p, 2])
# Should expand to: p * p (two parameters)
```

**(x*y)^2 = x^2 * y^2 (degree 4 - INVALID):**
```julia
ScalarNonlinearFunction(:^, Any[
    ScalarNonlinearFunction(:*, Any[x, y]),
    2
])
# Should be rejected - exceeds cubic degree
```

**x^3 (degree 3 in one variable - special case):**
```julia
ScalarNonlinearFunction(:^, Any[x, 3])
# This is x*x*x - three variables, no parameters
# Should be REJECTED for our use case (no parameter involved)
# OR could be stored as a degenerate cubic term if we allow it
```

**Parser requirement**: Must correctly expand `^` operator and track resulting degree.

### 1.5.4 Addition/Subtraction Variations

**Sum of terms:**
```julia
# x*y*p + x*z*q
ScalarNonlinearFunction(:+, Any[
    ScalarNonlinearFunction(:*, Any[x, y, p]),
    ScalarNonlinearFunction(:*, Any[x, z, q])
])
```

**Subtraction (which is addition with negation):**
```julia
# x*y*p - 2*x
ScalarNonlinearFunction(:-, Any[
    ScalarNonlinearFunction(:*, Any[x, y, p]),
    ScalarNonlinearFunction(:*, Any[2.0, x])
])
# The second term should have coefficient -2.0
```

**Nested sums:**
```julia
# (a + b) + (c + d) with various term types
ScalarNonlinearFunction(:+, Any[
    ScalarNonlinearFunction(:+, Any[term_a, term_b]),
    ScalarNonlinearFunction(:+, Any[term_c, term_d])
])
```

**Unary minus:**
```julia
# -x*y*p (negation of a term)
ScalarNonlinearFunction(:-, Any[
    MOI.ScalarNonlinearFunction(:*, Any[x, y, p])
])
# Should produce coefficient = -1.0
```

**Parser requirement**: Must correctly propagate signs through subtraction, unary minus, and flatten additions.

### 1.5.5 Coefficient Positions

The numeric coefficient can appear anywhere in a product:

```julia
# All equivalent to 5*x*y*p:
ScalarNonlinearFunction(:*, Any[5.0, x, y, p])      # coefficient first
ScalarNonlinearFunction(:*, Any[x, 5.0, y, p])      # coefficient second
ScalarNonlinearFunction(:*, Any[x, y, 5.0, p])      # coefficient third
ScalarNonlinearFunction(:*, Any[x, y, p, 5.0])      # coefficient last
```

**Multiple coefficients (must multiply):**
```julia
# 2 * 3 * x * y * p = 6*x*y*p
ScalarNonlinearFunction(:*, Any[2.0, 3.0, x, y, p])
```

**Parser requirement**: Accumulate all numeric values by multiplication.

### 1.5.6 Term Combination

Like terms must be combined:

```julia
# x*y*p + 2*x*y*p should become 3*x*y*p
ScalarNonlinearFunction(:+, Any[
    ScalarNonlinearFunction(:*, Any[x, y, p]),
    ScalarNonlinearFunction(:*, Any[2.0, x, y, p])
])
# Result: single _ScalarCubicTerm with coefficient = 3.0 (type=:pvv)
```

**Parser requirement**: After expanding to monomials, combine terms with identical variable/parameter sets.

### 1.5.7 Nested MOI Functions

`ScalarNonlinearFunction` args can contain other MOI function types:

```julia
# ScalarAffineFunction nested inside
ScalarNonlinearFunction(:*, Any[
    MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(2.0, x)], 1.0),  # 2x + 1
    p
])
# Should expand to: 2*x*p + p (one pv term + one p term)
```

```julia
# ScalarQuadraticFunction nested inside
ScalarNonlinearFunction(:*, Any[
    MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(1.0, x, y)],  # x*y
        MOI.ScalarAffineTerm{Float64}[],
        0.0
    ),
    p
])
# Should expand to: x*y*p (one pvv term)
```

**Parser requirement**: Recursively handle `ScalarAffineFunction` and `ScalarQuadraticFunction` as leaf nodes that expand to their constituent terms.

### 1.5.8 Edge Cases Summary Table

| Expression | Tree Variations | Expected Result |
|------------|-----------------|-----------------|
| `2*x*y*p` | 5+ structures | `_ScalarCubicTerm(2.0, p, x, y)` (type=:pvv) |
| `3*p*x^2` | 3+ structures | `_ScalarCubicTerm(3.0, p, x, x)` (type=:pvv) |
| `x*y*p - 2*x` | subtraction | 1 pvv term + 1 affine (coef=-2) |
| `-x*y*p` | unary minus | `_ScalarCubicTerm(-1.0, p, x, y)` (type=:pvv) |
| `(2*3)*x*y*p` | nested coefficients | `_ScalarCubicTerm(6.0, p, x, y)` (type=:pvv) |
| `x*y*p + 2*x*y*p` | like terms | `_ScalarCubicTerm(3.0, p, x, y)` (type=:pvv) |
| `(2x+1)*p` | nested affine | 1 pv + 1 p term |
| `x^2*y^2` | power of product | REJECT (degree 4) |
| `x*y*z` | 3 vars, 0 params | REJECT (no parameter) |
| `sin(x)*p` | non-poly operator | REJECT |
| `x/y*p` | division | REJECT |
| `p*x*y` with p=0 | zero parameter value | MUST parse as cubic (see 1.5.10) |

### 1.5.9 Parser Implementation Strategy

Given these corner cases, the parser should:

1. **Normalize to monomials**: Recursively expand the tree into a sum of monomials
2. **Track factors per monomial**: Each monomial has:
   - `coefficient::Float64` (product of all numeric values)
   - `variables::Vector{MOI.VariableIndex}` (including repeats for x^2)
   - `parameters::Vector{ParameterIndex}` (including repeats for p^2)
3. **Handle operators**:
   - `:+` → collect monomials from each arg
   - `:-` → negate coefficients of args after the first (unary: negate single arg)
   - `:*` → multiply monomials (combine factors)
   - `:^` → expand to repeated factors (only for small integer exponents)
4. **Handle nested MOI functions**:
   - `ScalarAffineFunction` → expand to affine monomials
   - `ScalarQuadraticFunction` → expand to quadratic + affine monomials
5. **Combine like terms**: Group monomials by (sorted variables, sorted parameters), sum coefficients
6. **Classify after expansion**: Once we have flat, combined monomials, classification is straightforward

```julia
struct Monomial{T}
    coefficient::T
    variables::Vector{MOI.VariableIndex}   # length = variable degree
    parameters::Vector{ParameterIndex}      # length = parameter degree
end

# Total degree = length(variables) + length(parameters)
# For valid cubic: total degree <= 3 AND length(variables) <= 2
```

### 1.5.10 Parameter Values at Parse Time (CRITICAL)

**The parser must NOT consider parameter values when classifying terms.**

When parsing `p * x * y`, the parser must always create a PVV cubic term, regardless of whether `p`'s current value is 0, 1, or any other number. The parameter value only affects the *evaluation* of the term, not its *classification*.

**Why this matters:**

```julia
# User creates model with p=0
@variable(model, p in MOI.Parameter(0.0))
@objective(model, Min, p * x * y + x + y)

# First solve: p=0, so effectively minimizing x + y
optimize!(model)  # Works fine

# Later, user updates p
set_parameter_value(p, 2.0)
optimize!(model)  # Must now minimize 2*x*y + x + y
```

If the parser had ignored the `p * x * y` term because `p=0`, the second optimization would be wrong.

**Implementation requirement:**

- Parser classifies terms based on **structure** (which indices are parameters vs variables)
- Parser does NOT read parameter values from the model
- All cubic terms are stored, even if their parameters are currently zero
- During `_update_cubic_objective!`, zero-valued parameters correctly result in zero contributions

**Test cases:** See D1b and D1c in the test plan.

### 1.5.11 MOI Utilities That May Simplify Implementation

MOI provides several utilities in `MOI.Utilities` and `MOI.Nonlinear` that could simplify our work:

#### Potentially Useful Utilities

| Utility | Location | Purpose | Use Case |
|---------|----------|---------|----------|
| `substitute_variables(fn, f)` | `MOI.Utilities` | Replace variables using a mapping function | Could substitute parameter values |
| `filter_variables(keep, f)` | `MOI.Utilities` | Remove variables not satisfying predicate | Separate params from vars |
| `canonical(f)` | `MOI.Utilities` | Normalize function (combine terms, sort) | Simplify parsed result |
| `eval_variables(value_fn, model, f)` | `MOI.Nonlinear` | Evaluate expression with variable values | Evaluate with param values |
| `operate(op, T, args...)` | `MOI.Utilities` | Compose functions with operators | Build result functions |
| `map_indices(fn, f)` | `MOI.Utilities` | Remap variable indices | Index transformations |

#### Key Insight: Alternative Approach Using `substitute_variables`

Instead of building a full custom parser, we could potentially:

```julia
function _current_function_alternative(f::MOI.ScalarNonlinearFunction, model)
    # Substitute parameters with their values
    result = MOI.Utilities.substitute_variables(f) do vi
        if _is_parameter(vi)
            # Return the parameter value as a constant
            return model.parameters[p_idx(vi)]
        else
            # Keep variables as-is
            return vi
        end
    end
    # Result should now be a polynomial in variables only
    # Convert to ScalarQuadraticFunction or ScalarAffineFunction
    return result
end
```

**Limitation**: `substitute_variables` returns `ScalarNonlinearFunction` even after substitution. We'd still need to detect that the result is polynomial and convert it.

#### Recommended Approach

Use MOI utilities for:
1. **`canonical()`** - After parsing, to combine like terms and normalize
2. **`map_indices()`** - If we need to remap variable indices
3. **`operate()`** - To build the final `ScalarQuadraticFunction` or `ScalarAffineFunction`

Build custom logic for:
1. **Expression tree traversal** - To expand into monomials
2. **Polynomial detection** - To verify the expression is valid cubic
3. **Term classification** - To categorize by parameter/variable composition

This hybrid approach leverages MOI utilities where beneficial while maintaining control over the polynomial-specific logic.

---

## Part 2: Data Structures

### 2.1 Unified Cubic Term Type

We use a single unified type for all cubic terms, similar to how MOI uses `ScalarQuadraticTerm`:

```julia
"""
    _ScalarCubicTerm{T}

Represents a cubic term of the form `coefficient * index_1 * index_2 * index_3`.

Each index can be either a variable (MOI.VariableIndex) or a parameter (encoded as
VariableIndex with value > PARAMETER_INDEX_THRESHOLD).

The term type is determined by counting parameters vs variables:
- PVV (1 param, 2 vars): becomes quadratic after substitution
- PPV (2 params, 1 var): becomes affine after substitution
- PPP (3 params, 0 vars): becomes constant after substitution

# Fields
- `coefficient::T`: The numeric coefficient
- `index_1::MOI.VariableIndex`: First factor (variable or parameter)
- `index_2::MOI.VariableIndex`: Second factor (variable or parameter)
- `index_3::MOI.VariableIndex`: Third factor (variable or parameter)

# Convention
Indices are stored in canonical order:
- Parameters come before variables
- Within each group, sorted by index value
This ensures `2*p*x*y` and `2*x*p*y` produce the same term.

# Examples
```julia
# p * x * y (PVV): 1 parameter, 2 variables
_ScalarCubicTerm(2.0, p_vi, x, y)  # becomes 2*p_val*x*y

# p * q * x (PPV): 2 parameters, 1 variable
_ScalarCubicTerm(3.0, p_vi, q_vi, x)  # becomes 3*p_val*q_val*x

# p * q * r (PPP): 3 parameters, 0 variables
_ScalarCubicTerm(4.0, p_vi, q_vi, r_vi)  # becomes 4*p_val*q_val*r_val
```
"""
struct _ScalarCubicTerm{T}
    coefficient::T
    index_1::MOI.VariableIndex
    index_2::MOI.VariableIndex
    index_3::MOI.VariableIndex
end

# Helper to classify a cubic term
function _cubic_term_type(term::_ScalarCubicTerm)
    num_params = _is_parameter(term.index_1) + _is_parameter(term.index_2) + _is_parameter(term.index_3)
    if num_params == 1
        return :pvv  # 1 param, 2 vars → quadratic
    elseif num_params == 2
        return :ppv  # 2 params, 1 var → affine
    else  # num_params == 3
        return :ppp  # 3 params → constant
    end
end

# Helper to extract parameters and variables from a term
function _split_cubic_term(term::_ScalarCubicTerm)
    params = MOI.VariableIndex[]
    vars = MOI.VariableIndex[]
    for idx in (term.index_1, term.index_2, term.index_3)
        if _is_parameter(idx)
            push!(params, idx)
        else
            push!(vars, idx)
        end
    end
    return params, vars
end
```

#### 2.1.1 Summary of Cubic Term Classifications

| Classification | # Params | # Vars | After Substitution | Example |
|----------------|----------|--------|-------------------|---------|
| PVV | 1 | 2 | Quadratic: `c*p_val*x*y` | `2*p*x*y` → `6*x*y` (p=3) |
| PPV | 2 | 1 | Affine: `c*p_val*q_val*x` | `2*p*q*x` → `12*x` (p=2,q=3) |
| PPP | 3 | 0 | Constant: `c*p_val*q_val*r_val` | `2*p*q*r` → `24` (p=2,q=3,r=4) |

### 2.2 ParametricCubicFunction

Storage for a full cubic function with parametric terms.

```julia
"""
    ParametricCubicFunction{T} <: ParametricFunction{T}

Represents a cubic function where parameters multiply up to quadratic variable terms.

Supports the general form:
    constant + Σ(affine) + Σ(quadratic) + Σ(cubic)

# Fields

## Cubic terms (degree 3) - unified type
- `cubic::Vector{_ScalarCubicTerm{T}}`: All cubic terms (pvv, ppv, ppp combined)
  - Classification done at runtime via `_cubic_term_type()`
  - pvv terms → become quadratic after substitution
  - ppv terms → become affine after substitution
  - ppp terms → become constant after substitution

## Quadratic terms (degree 2) - inherited pattern from ParametricQuadraticFunction
- `pv::Vector{...}`: Parameter-variable terms (c*p*x) → become affine
- `pp::Vector{...}`: Parameter-parameter terms (c*p*q) → become constant
- `vv::Vector{...}`: Variable-variable terms (c*x*y) → stay quadratic

## Affine terms (degree 1)
- `p::Vector{...}`: Parameter affine terms (c*p) → become constant
- `v::Vector{...}`: Variable affine terms (c*x) → stay affine

## Constants (degree 0)
- `c::T`: Constant term

## Caches (for efficient updates)
- `current_quadratic_terms::Vector{MOI.ScalarQuadraticTerm{T}}`: From vv + pvv
- `current_affine_terms::Vector{MOI.ScalarAffineTerm{T}}`: From v + pv + ppv
- `current_constant::T`: From c + p + pp

# Substitution Summary

| Original Term | After Parameter Substitution |
|---------------|------------------------------|
| `pvv` (p*x*y) | quadratic term (c*p_val*x*y) |
| `ppv` (p*q*x) | affine term (c*p_val*q_val*x)|
| `ppp` (p*q*r) | constant (c*p_val*q_val*r_val)|
| `pv` (p*x)    | affine term (c*p_val*x)      |
| `pp` (p*q)    | constant (c*p_val*q_val)     |
| `vv` (x*y)    | quadratic term (unchanged)   |
| `v` (x)       | affine term (unchanged)      |
| `p` (p)       | constant (c*p_val)           |
| `c`           | constant (unchanged)         |
"""
mutable struct ParametricCubicFunction{T} <: ParametricFunction{T}
    # === Cubic terms (degree 3) - unified storage ===
    cubic::Vector{_ScalarCubicTerm{T}}  # All cubic terms (pvv, ppv, ppp)
    # Classification done at runtime via _cubic_term_type()

    # === Quadratic terms (degree 2) - same as ParametricQuadraticFunction ===
    pv::Vector{MOI.ScalarQuadraticTerm{T}}   # p*x → becomes affine
    pp::Vector{MOI.ScalarQuadraticTerm{T}}   # p*q → becomes constant
    vv::Vector{MOI.ScalarQuadraticTerm{T}}   # x*y → stays quadratic

    # === Affine terms (degree 1) ===
    p::Vector{MOI.ScalarAffineTerm{T}}       # p → becomes constant
    v::Vector{MOI.ScalarAffineTerm{T}}       # x → stays affine

    # === Constant (degree 0) ===
    c::T

    # === Caches for efficient updates (following POI pattern) ===
    # Variable pairs in pvv terms (quadratic coefficients depend on parameters)
    quadratic_data::Dict{Tuple{MOI.VariableIndex,MOI.VariableIndex}, T}
    # Variables in ppv or pv terms (affine coefficients depend on parameters)
    affine_data::Dict{MOI.VariableIndex, T}
    # Variables NOT in parameter-dependent terms (fixed coefficients)
    affine_data_np::Dict{MOI.VariableIndex, T}
    # Current constant after parameter substitution
    current_constant::T
end

# Accessors for cubic terms by type (iterate and filter)
function _cubic_pvv_terms(f::ParametricCubicFunction)
    return Iterators.filter(t -> _cubic_term_type(t) == :pvv, f.cubic)
end

function _cubic_ppv_terms(f::ParametricCubicFunction)
    return Iterators.filter(t -> _cubic_term_type(t) == :ppv, f.cubic)
end

function _cubic_ppp_terms(f::ParametricCubicFunction)
    return Iterators.filter(t -> _cubic_term_type(t) == :ppp, f.cubic)
end
```

### 2.3 Parser Result

```julia
"""
    ParsedCubicExpression{T}

Result of parsing a ScalarNonlinearFunction into cubic polynomial form.

Returns `nothing` from the parser if the expression is not a valid cubic polynomial.

Note: Like-terms should be combined during parsing (e.g., `x*y*p + 2*x*y*p` → single term with coef=3).

# Design

The structure contains:
1. A `MOI.ScalarQuadraticFunction{T}` for all non-cubic terms (quadratic, affine, constant)
2. A vector of `_ScalarCubicTerm{T}` for cubic terms only

This reuses MOI's existing structure and simplifies construction of the final function
after parameter substitution.

# Term Classification

Cubic terms are stored in a unified vector and classified at runtime:
- Use `_cubic_term_type(term)` to get `:pvv`, `:ppv`, or `:ppp`
- Use filter functions `_filter_pvv_terms()`, etc. if needed

# Note on Parameter Encoding

In the `quadratic_func`, parameters appear as `MOI.VariableIndex` with values above
`PARAMETER_INDEX_THRESHOLD`. The caller must use `_is_parameter()` to distinguish
parameter terms (pp, pv, p) from pure variable terms (vv, v, constant).
"""
struct ParsedCubicExpression{T}
    # Cubic terms (degree 3) - unified storage
    cubic_terms::Vector{_ScalarCubicTerm{T}}  # All cubic: p*x*y, p*q*x, p*q*r
    # Classification done via _cubic_term_type() at runtime

    # Non-cubic terms (degree ≤ 2) - reuse MOI's structure
    # Contains: vv, pv, pp (quadratic), v, p (affine), constant
    # Parameters encoded as VariableIndex > PARAMETER_INDEX_THRESHOLD
    quadratic_func::MOI.ScalarQuadraticFunction{T}
end

# Helper functions to filter cubic terms by type
function _filter_pvv_terms(parsed::ParsedCubicExpression)
    return filter(t -> _cubic_term_type(t) == :pvv, parsed.cubic_terms)
end

function _filter_ppv_terms(parsed::ParsedCubicExpression)
    return filter(t -> _cubic_term_type(t) == :ppv, parsed.cubic_terms)
end

function _filter_ppp_terms(parsed::ParsedCubicExpression)
    return filter(t -> _cubic_term_type(t) == :ppp, parsed.cubic_terms)
end

# Convenience accessors for non-cubic terms
function _quadratic_terms(parsed::ParsedCubicExpression)
    return parsed.quadratic_func.quadratic_terms
end

function _affine_terms(parsed::ParsedCubicExpression)
    return parsed.quadratic_func.affine_terms
end

function _constant(parsed::ParsedCubicExpression)
    return parsed.quadratic_func.constant
end
```

---

## Part 3: Core Functions

### 3.1 Parser Functions

```julia
"""
    _parse_cubic_expression(f::MOI.ScalarNonlinearFunction) -> Union{ParsedCubicExpression, Nothing}

Parse a ScalarNonlinearFunction and return a ParsedCubicExpression if it represents
a valid cubic polynomial (with parameters multiplying at most quadratic variable terms).

Returns `nothing` if the expression:
- Contains non-polynomial operations (sin, exp, etc.)
- Has degree > 3 in any monomial
- Has invalid structure

# Example
```julia
x, y = MOI.VariableIndex(1), MOI.VariableIndex(2)
p = POI.ParameterIndex(1)

# 2*x*y*p + 3*x
f = MOI.ScalarNonlinearFunction(:+, Any[
    MOI.ScalarNonlinearFunction(:*, Any[2.0, x, y, v_idx(p)]),
    MOI.ScalarNonlinearFunction(:*, Any[3.0, x])
])

result = _parse_cubic_expression(f)
# result.cubic_terms = [_ScalarCubicTerm(2.0, p, x, y)]
# result.quadratic_func.affine_terms = [ScalarAffineTerm(3.0, x)]
```
"""
function _parse_cubic_expression(f::MOI.ScalarNonlinearFunction)
    # Implementation
end
```

### 3.2 Helper Functions for Parsing

```julia
"""
    _expand_expression(f::MOI.ScalarNonlinearFunction) -> Vector{Monomial}

Recursively expand the expression tree into a list of monomials.
Each monomial tracks: coefficient, list of variables, list of parameters.
"""
function _expand_expression(f)
end

"""
    _classify_monomial(m::Monomial) -> Symbol

Classify a monomial by its structure (num_params, num_vars):

Degree 0:
- :constant - (0, 0) no variables or parameters

Degree 1:
- :affine_v - (0, 1) one variable, no parameters
- :affine_p - (1, 0) one parameter, no variables

Degree 2:
- :quadratic_vv - (0, 2) two variables, no parameters
- :quadratic_pv - (1, 1) one parameter, one variable
- :quadratic_pp - (2, 0) two parameters, no variables

Degree 3 (valid - at least one parameter):
- :cubic_pvv - (1, 2) one parameter, two variables
- :cubic_ppv - (2, 1) two parameters, one variable
- :cubic_ppp - (3, 0) three parameters, no variables

Invalid:
- :cubic_vvv - (0, 3) three variables, no parameters → REJECT (no parameter)
- :invalid - degree > 3, or any other invalid combination
"""
function _classify_monomial(m)
end

"""
    _is_polynomial_operator(head::Symbol) -> Bool

Check if the operator is valid for polynomial expressions.
Valid: :+, :-, :*, :^
Invalid: :sin, :cos, :exp, :log, :/, etc.
"""
function _is_polynomial_operator(head::Symbol)
end
```

### 3.3 Conversion Functions

```julia
"""
    ParametricCubicFunction(f::MOI.ScalarNonlinearFunction)

Construct a ParametricCubicFunction from a ScalarNonlinearFunction.

Throws an error if the expression is not a valid cubic polynomial.
"""
function ParametricCubicFunction(f::MOI.ScalarNonlinearFunction)
end

"""
    _current_function(f::ParametricCubicFunction{T}, model) where {T} -> Union{MOI.ScalarQuadraticFunction{T}, MOI.ScalarAffineFunction{T}}

Evaluate the cubic function with current parameter values and return
the appropriate MOI function type. Follows the same pattern as
ParametricQuadraticFunction._current_function.

# Implementation
1. Build quadratic terms from:
   - `vv` terms (unchanged)
   - `pvv` terms with parameter substituted: coef * p_val * x * y

2. Build affine terms from:
   - `affine_data` (variables in parameter-dependent terms, with updated coefficients)
   - `affine_data_np` (variables with fixed coefficients)

3. Use `current_constant` for the constant term

# Returns
- `ScalarQuadraticFunction{T}` if there are any quadratic terms
- `ScalarAffineFunction{T}` if all quadratic terms have zero coefficient

# Example
```julia
# f = 2*p*x*y + 3*x + 5 with p=3
# _current_function returns: 6*x*y + 3*x + 5
```
"""
function _current_function(f::ParametricCubicFunction{T}, model) where {T}
    # Build quadratic terms
    quadratic = MOI.ScalarQuadraticTerm{T}[]
    # Add vv terms (unchanged)
    append!(quadratic, f.vv)
    # Add pvv terms with parameter values substituted
    for (vars, coef) in f.quadratic_data
        if !iszero(coef)
            push!(quadratic, MOI.ScalarQuadraticTerm{T}(coef, vars[1], vars[2]))
        end
    end

    # Build affine terms
    affine = MOI.ScalarAffineTerm{T}[]
    for (v, coef) in f.affine_data
        push!(affine, MOI.ScalarAffineTerm{T}(coef, v))
    end
    for (v, coef) in f.affine_data_np
        push!(affine, MOI.ScalarAffineTerm{T}(coef, v))
    end

    # Return appropriate type
    if isempty(quadratic)
        return MOI.ScalarAffineFunction{T}(affine, f.current_constant)
    else
        return MOI.ScalarQuadraticFunction{T}(quadratic, affine, f.current_constant)
    end
end

"""
    _update_cache!(f::ParametricCubicFunction{T}, model) where {T}

Update the cached current values based on new parameter values.
Follows the same pattern as ParametricQuadraticFunction._update_cache!
"""
function _update_cache!(f::ParametricCubicFunction{T}, model) where {T}
    f.current_constant = _parametric_constant(model, f)
    f.affine_data = _parametric_affine_terms(model, f)
    f.quadratic_data = _parametric_quadratic_terms(model, f)
    return nothing
end

"""
    _parametric_constant(model, f::ParametricCubicFunction{T}) where {T}

Compute the constant term after parameter substitution.
Includes contributions from: c + p terms + pp terms + ppp cubic terms
"""
function _parametric_constant(model, f::ParametricCubicFunction{T}) where {T}
    constant = f.c
    # From affine parameter terms (p)
    for term in f.p
        constant += term.coefficient * model.parameters[p_idx(term.variable)]
    end
    # From quadratic parameter-parameter terms (pp)
    for term in f.pp
        constant += (term.coefficient / ifelse(term.variable_1 == term.variable_2, 2, 1)) *
            model.parameters[p_idx(term.variable_1)] *
            model.parameters[p_idx(term.variable_2)]
    end
    # From cubic ppp terms (all 3 indices are parameters)
    for term in _cubic_ppp_terms(f)
        params, _ = _split_cubic_term(term)
        divisor = _cubic_divisor(params)  # handles p^3, p^2*q, p*q*r cases
        constant += (term.coefficient / divisor) *
            model.parameters[p_idx(params[1])] *
            model.parameters[p_idx(params[2])] *
            model.parameters[p_idx(params[3])]
    end
    return constant
end

# Helper to compute divisor for repeated indices (for symmetric terms)
function _cubic_divisor(indices::Vector{MOI.VariableIndex})
    if indices[1] == indices[2] == indices[3]
        return 6  # p^3: divide by 3!
    elseif indices[1] == indices[2] || indices[2] == indices[3] || indices[1] == indices[3]
        return 2  # p^2*q: divide by 2!
    else
        return 1  # p*q*r: no division needed
    end
end

"""
    _parametric_affine_terms(model, f::ParametricCubicFunction{T}) where {T}

Compute affine coefficients after parameter substitution.
Includes contributions from: v terms + pv terms + ppv cubic terms
"""
function _parametric_affine_terms(model, f::ParametricCubicFunction{T}) where {T}
    terms_dict = Dict{MOI.VariableIndex, T}()
    # From pv terms (same as ParametricQuadraticFunction)
    for term in f.pv
        var = term.variable_2
        base = get(terms_dict, var, zero(T))
        terms_dict[var] = base + term.coefficient * model.parameters[p_idx(term.variable_1)]
    end
    # From ppv cubic terms (2 params, 1 var) - p * q * x becomes coef * p_val * q_val * x
    for term in _cubic_ppv_terms(f)
        params, vars = _split_cubic_term(term)
        var = vars[1]  # The single variable
        p1_val = model.parameters[p_idx(params[1])]
        p2_val = model.parameters[p_idx(params[2])]
        divisor = ifelse(params[1] == params[2], 2, 1)
        base = get(terms_dict, var, zero(T))
        terms_dict[var] = base + (term.coefficient / divisor) * p1_val * p2_val
    end
    # Add fixed affine terms from v (stored in affine_data)
    for (var, coef) in f.affine_data
        terms_dict[var] = get(terms_dict, var, zero(T)) + coef
    end
    return terms_dict
end

"""
    _parametric_quadratic_terms(model, f::ParametricCubicFunction{T}) where {T}

Compute quadratic coefficients after parameter substitution.
Includes contributions from: pvv terms (p * x * y becomes coef * p_val * x * y)
"""
function _parametric_quadratic_terms(model, f::ParametricCubicFunction{T}) where {T}
    terms_dict = Dict{Tuple{MOI.VariableIndex, MOI.VariableIndex}, T}()
    for term in _cubic_pvv_terms(f)
        params, vars = _split_cubic_term(term)
        p = params[1]  # The single parameter
        var_pair = (vars[1], vars[2])  # The two variables
        p_val = model.parameters[p_idx(p)]
        base = get(terms_dict, var_pair, zero(T))
        terms_dict[var_pair] = base + term.coefficient * p_val
    end
    return terms_dict
end

# === Delta functions for efficient updates (following POI pattern) ===

"""
    _delta_parametric_constant(model, f::ParametricCubicFunction{T}) where {T}

Compute the CHANGE in constant when parameters are updated.
Only computes delta for parameters that have been updated (not NaN).
"""
function _delta_parametric_constant(model, f::ParametricCubicFunction{T}) where {T}
    # Similar to ParametricQuadraticFunction but also handles ppp terms
    # ... (implementation follows existing pattern)
end

"""
    _delta_parametric_affine_terms(model, f::ParametricCubicFunction{T}) where {T}

Compute the CHANGE in affine coefficients when parameters are updated.
Returns Dict{MOI.VariableIndex, T} of delta values.
"""
function _delta_parametric_affine_terms(model, f::ParametricCubicFunction{T}) where {T}
    # Similar to ParametricQuadraticFunction but also handles ppv terms
    # ... (implementation follows existing pattern)
end

"""
    _delta_parametric_quadratic_terms(model, f::ParametricCubicFunction{T}) where {T}

Compute the CHANGE in quadratic coefficients when parameters are updated.
Returns Dict{Tuple{MOI.VariableIndex, MOI.VariableIndex}, T} of delta values.

This is NEW for cubic functions - quadratic terms can change when pvv parameters change.
"""
function _delta_parametric_quadratic_terms(model, f::ParametricCubicFunction{T}) where {T}
    delta_dict = Dict{Tuple{MOI.VariableIndex, MOI.VariableIndex}, T}()
    for term in _cubic_pvv_terms(f)
        params, vars = _split_cubic_term(term)
        p = p_idx(params[1])  # The single parameter
        if !isnan(model.updated_parameters[p])
            var_pair = (vars[1], vars[2])  # The two variables
            old_val = model.parameters[p]
            new_val = model.updated_parameters[p]
            delta = term.coefficient * (new_val - old_val)
            base = get(delta_dict, var_pair, zero(T))
            delta_dict[var_pair] = base + delta
        end
    end
    return delta_dict
end
```

---

## Part 4: Integration with POI

### 4.1 Optimizer Storage

Add to the `Optimizer` struct:

```julia
# In Optimizer struct definition
cubic_objective_cache::Union{Nothing, ParametricCubicFunction{T}}

# Option to warn on quadratic coefficient sign changes (can affect convexity)
warn_on_quadratic_sign_change::Bool
```

**New constructor parameter:**

```julia
function Optimizer{T}(
    optimizer::OT;
    evaluate_duals::Bool = true,
    save_original_objective_and_constraints::Bool = true,
    warn_on_quadratic_sign_change::Bool = false,  # NEW
) where {T,OT}
    # ...
end
```

**Purpose:** When `warn_on_quadratic_sign_change = true`, POI will check if any quadratic coefficient changes sign during parameter updates (e.g., from positive to negative or vice versa). A sign change in quadratic terms can change the problem's convexity, which may:
- Cause solvers to fail or produce suboptimal results
- Change the problem from having a unique solution to multiple local optima
- Affect convergence behavior

**Note:** This check is disabled by default for performance. Enable it during development/debugging if you suspect convexity issues.

### 4.2 Objective Setting

Add method for setting ScalarNonlinearFunction as objective:

```julia
function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
    f::MOI.ScalarNonlinearFunction,
)
    # 1. Attempt to parse as cubic
    parsed = _parse_cubic_expression(f)
    if parsed === nothing
        error("ScalarNonlinearFunction must be a cubic polynomial for POI")
    end

    # 2. Create ParametricCubicFunction
    cubic_func = ParametricCubicFunction(parsed)

    # 3. Clear old caches, store new cache
    _empty_objective_function_caches!(model)
    model.cubic_objective_cache = cubic_func

    # 4. Compute current function and set on inner optimizer
    current = _current_function(cubic_func, model)
    MOI.set(model.optimizer, MOI.ObjectiveFunction{typeof(current)}(), current)

    # 5. Store original for retrieval
    MOI.Utilities.set_objective(model.original_objective_cache, f)
end
```

### 4.3 Parameter Updates

Extend `_update_parameters!` to handle cubic objectives:

```julia
function _update_cubic_objective!(model::Optimizer{T}) where {T}
    if model.cubic_objective_cache === nothing
        return
    end
    pf = model.cubic_objective_cache

    # 1. Update constant (from p, pp, ppp terms)
    delta_constant = _delta_parametric_constant(model, pf)
    if !iszero(delta_constant)
        pf.current_constant += delta_constant
        F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
        MOI.modify(
            model.optimizer,
            MOI.ObjectiveFunction{F}(),
            MOI.ScalarConstantChange(pf.current_constant),
        )
    end

    # 2. Update affine terms (from pv, ppv terms)
    delta_affine = _delta_parametric_affine_terms(model, pf)
    if !isempty(delta_affine)
        # Update cache and build changes
        changes = _affine_build_change_and_up_param_func(pf, delta_affine)
        F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
        MOI.modify(model.optimizer, MOI.ObjectiveFunction{F}(), changes)
    end

    # 3. Update quadratic terms (from pvv terms) - NEW for cubic
    delta_quadratic = _delta_parametric_quadratic_terms(model, pf)
    if !isempty(delta_quadratic)
        F = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
        for ((var1, var2), delta) in delta_quadratic
            # Update cache
            old_coef = get(pf.quadratic_data, (var1, var2), zero(T))
            new_coef = old_coef + delta
            pf.quadratic_data[(var1, var2)] = new_coef

            # Check for sign change if option is enabled
            if model.warn_on_quadratic_sign_change
                _check_quadratic_sign_change(old_coef, new_coef, var1, var2)
            end

            # Apply change using MOI.ScalarQuadraticCoefficientChange
            MOI.modify(
                model.optimizer,
                MOI.ObjectiveFunction{F}(),
                MOI.ScalarQuadraticCoefficientChange(var1, var2, new_coef),
            )
        end
    end

    return
end

"""
    _check_quadratic_sign_change(old_coef, new_coef, var1, var2)

Check if a quadratic coefficient changed sign and emit a warning if so.
Sign changes can affect problem convexity.
"""
function _check_quadratic_sign_change(old_coef::T, new_coef::T, var1, var2) where {T}
    # Skip if either coefficient is zero (not a true sign change)
    if iszero(old_coef) || iszero(new_coef)
        return
    end
    # Check for sign change: positive → negative or negative → positive
    if (old_coef > zero(T)) != (new_coef > zero(T))
        @warn "Quadratic coefficient sign change detected" var1 var2 old_coef new_coef
    end
end
```

**Note**: MOI supports `ScalarQuadraticCoefficientChange` for modifying quadratic coefficients in-place. See [MOI modification documentation](https://jump.dev/MathOptInterface.jl/stable/manual/modification/).

```julia
# In update_parameters.jl, add call to cubic update:
function update_parameters!(model::Optimizer)
    _update_affine_constraints!(model)
    _update_vector_affine_constraints!(model)
    _update_quadratic_constraints!(model)
    _update_vector_quadratic_constraints!(model)
    _update_affine_objective!(model)
    _update_quadratic_objective!(model)
    _update_cubic_objective!(model)  # NEW

    # Update parameters and put NaN to indicate updated
    for (parameter_index, val) in model.updated_parameters
        if !isnan(val)
            model.parameters[parameter_index] = val
            model.updated_parameters[parameter_index] = NaN
        end
    end
    return
end
```

---

## Part 5: Test Plan

### 5.1 Test Philosophy

Tests should be:
- **Simple**: Easy to compute expected results by hand
- **Predictable**: Use integer/simple coefficients
- **Focused**: Each test validates one specific behavior
- **Complete**: Cover parameter changes and result verification

### 5.1.1 JuMP vs MOI Tests

**MOI-level tests** (`test/moi_tests.jl`):
- Parser unit tests (expression tree parsing)
- Data structure construction tests
- Low-level API verification

**JuMP-level tests** (`test/jump_tests.jl`) - **PRIMARY**:
- Full model integration tests
- Parameter update and re-optimization tests
- User-facing API validation

JuMP tests are preferred for full model validation because:
1. If MOI's ScalarNonlinearFunction is implemented correctly, JuMP will work automatically
2. JuMP syntax is closer to what users will write
3. Easier to read and verify expected behavior

### 5.2 Test Categories

#### Category A: Parser Tests

```julia
# A1: Valid cubic expression - single PVV term
@testset "parse_cubic_single_term" begin
    x, y = MOI.VariableIndex(1), MOI.VariableIndex(2)
    p_vi = v_idx(ParameterIndex(1))

    # 2 * x * y * p
    f = MOI.ScalarNonlinearFunction(:*, Any[2.0, x, y, p_vi])
    result = _parse_cubic_expression(f)

    @test result !== nothing
    pvv = _filter_pvv_terms(result)
    @test length(pvv) == 1
    @test pvv[1].coefficient == 2.0
end

# A2: Valid cubic expression - mixed terms
@testset "parse_cubic_mixed_terms" begin
    # 3*x*y*p + 2*x + 5
    # Should parse into: 1 cubic, 1 affine, 1 constant
end

# A3: Invalid expression - degree too high (4 factors)
@testset "parse_cubic_invalid_degree_4" begin
    x, y, z = MOI.VariableIndex(1), MOI.VariableIndex(2), MOI.VariableIndex(3)
    p_vi = v_idx(ParameterIndex(1))

    # x * y * z * p (degree 4) should return nothing
    f = MOI.ScalarNonlinearFunction(:*, Any[x, y, z, p_vi])
    result = _parse_cubic_expression(f)

    @test result === nothing
end

# A3b: Invalid expression - three variables, no parameter
@testset "parse_cubic_three_vars_no_param" begin
    x, y, z = MOI.VariableIndex(1), MOI.VariableIndex(2), MOI.VariableIndex(3)

    # x * y * z (3 variables, 0 parameters) should be rejected
    # This is cubic in variables but has no parameter - not useful for POI
    f = MOI.ScalarNonlinearFunction(:*, Any[x, y, z])
    result = _parse_cubic_expression(f)

    @test result === nothing
end

# A4: Invalid expression - non-polynomial operator
@testset "parse_cubic_invalid_operator" begin
    # sin(x) * p should return nothing
end

# A5: Squared variable - p * x^2
@testset "parse_cubic_squared_variable" begin
    x = MOI.VariableIndex(1)
    p_vi = v_idx(ParameterIndex(1))

    # 3 * p * x^2 using power operator
    f = MOI.ScalarNonlinearFunction(:*, Any[
        3.0,
        p_vi,
        MOI.ScalarNonlinearFunction(:^, Any[x, 2])
    ])
    result = _parse_cubic_expression(f)

    @test result !== nothing
    pvv = _filter_pvv_terms(result)
    @test length(pvv) == 1
    @test pvv[1].coefficient == 3.0
    # Check that both variables are x (squared variable)
    _, vars = _split_cubic_term(pvv[1])
    @test vars[1] == x
    @test vars[2] == x  # same variable
end

# A6: Mixed parenthesis orderings - all should give same result
@testset "parse_cubic_parenthesis_variations" begin
    x, y = MOI.VariableIndex(1), MOI.VariableIndex(2)
    p_vi = v_idx(ParameterIndex(1))

    # Flat: 2 * x * y * p
    f1 = MOI.ScalarNonlinearFunction(:*, Any[2.0, x, y, p_vi])

    # Left-associative: ((2*x)*y)*p
    f2 = MOI.ScalarNonlinearFunction(:*, Any[
        MOI.ScalarNonlinearFunction(:*, Any[
            MOI.ScalarNonlinearFunction(:*, Any[2.0, x]),
            y
        ]),
        p_vi
    ])

    # Grouped: (2*p) * (x*y)
    f3 = MOI.ScalarNonlinearFunction(:*, Any[
        MOI.ScalarNonlinearFunction(:*, Any[2.0, p_vi]),
        MOI.ScalarNonlinearFunction(:*, Any[x, y])
    ])

    r1 = _parse_cubic_expression(f1)
    r2 = _parse_cubic_expression(f2)
    r3 = _parse_cubic_expression(f3)

    # All should parse to equivalent results
    for r in [r1, r2, r3]
        @test r !== nothing
        pvv = _filter_pvv_terms(r)
        @test length(pvv) == 1
        @test pvv[1].coefficient == 2.0
    end
end

# A7: Multiple numeric coefficients
@testset "parse_cubic_multiple_coefficients" begin
    x, y = MOI.VariableIndex(1), MOI.VariableIndex(2)
    p_vi = v_idx(ParameterIndex(1))

    # 2 * 3 * x * y * p = 6*x*y*p
    f = MOI.ScalarNonlinearFunction(:*, Any[2.0, 3.0, x, y, p_vi])
    result = _parse_cubic_expression(f)

    @test result !== nothing
    pvv = _filter_pvv_terms(result)
    @test pvv[1].coefficient == 6.0
end

# A8: Subtraction handling (binary minus)
@testset "parse_cubic_subtraction" begin
    x, y = MOI.VariableIndex(1), MOI.VariableIndex(2)
    p_vi = v_idx(ParameterIndex(1))

    # x*y*p - 2*x (one cubic, one affine with negative coef)
    f = MOI.ScalarNonlinearFunction(:-, Any[
        MOI.ScalarNonlinearFunction(:*, Any[x, y, p_vi]),
        MOI.ScalarNonlinearFunction(:*, Any[2.0, x])
    ])
    result = _parse_cubic_expression(f)

    @test result !== nothing
    pvv = _filter_pvv_terms(result)
    @test length(pvv) == 1
    # Check affine term via quadratic_func
    affine = _affine_terms(result)
    v_affine = filter(t -> !_is_parameter(t.variable), affine)
    @test length(v_affine) == 1
    @test v_affine[1].coefficient == -2.0
end

# A8b: Unary minus handling
@testset "parse_cubic_unary_minus" begin
    x, y = MOI.VariableIndex(1), MOI.VariableIndex(2)
    p_vi = v_idx(ParameterIndex(1))

    # -x*y*p (negation of cubic term)
    f = MOI.ScalarNonlinearFunction(:-, Any[
        MOI.ScalarNonlinearFunction(:*, Any[x, y, p_vi])
    ])
    result = _parse_cubic_expression(f)

    @test result !== nothing
    pvv = _filter_pvv_terms(result)
    @test length(pvv) == 1
    @test pvv[1].coefficient == -1.0
end

# A9: Explicit x*x vs x^2 should be equivalent
@testset "parse_cubic_explicit_square" begin
    x = MOI.VariableIndex(1)
    p_vi = v_idx(ParameterIndex(1))

    # Using x^2
    f1 = MOI.ScalarNonlinearFunction(:*, Any[
        p_vi,
        MOI.ScalarNonlinearFunction(:^, Any[x, 2])
    ])

    # Using x*x explicitly
    f2 = MOI.ScalarNonlinearFunction(:*, Any[p_vi, x, x])

    r1 = _parse_cubic_expression(f1)
    r2 = _parse_cubic_expression(f2)

    @test r1 !== nothing
    @test r2 !== nothing
    pvv1 = _filter_pvv_terms(r1)
    pvv2 = _filter_pvv_terms(r2)
    _, vars1 = _split_cubic_term(pvv1[1])
    _, vars2 = _split_cubic_term(pvv2[1])
    @test vars1[1] == vars2[1]
    @test vars1[2] == vars2[2]
end

# A10: Division should be rejected
@testset "parse_cubic_division_rejected" begin
    x, y = MOI.VariableIndex(1), MOI.VariableIndex(2)
    p_vi = v_idx(ParameterIndex(1))

    # x/y * p - division is not a polynomial operation
    f = MOI.ScalarNonlinearFunction(:*, Any[
        MOI.ScalarNonlinearFunction(:/, Any[x, y]),
        p_vi
    ])
    result = _parse_cubic_expression(f)

    @test result === nothing
end

# A11: Two parameters times one variable (PPV term)
@testset "parse_cubic_ppv_term" begin
    x = MOI.VariableIndex(1)
    p_vi = v_idx(ParameterIndex(1))
    q_vi = v_idx(ParameterIndex(2))

    # 2 * p * q * x
    f = MOI.ScalarNonlinearFunction(:*, Any[2.0, p_vi, q_vi, x])
    result = _parse_cubic_expression(f)

    @test result !== nothing
    ppv = _filter_ppv_terms(result)
    @test length(ppv) == 1
    @test ppv[1].coefficient == 2.0
end

# A12: Three parameters (PPP term)
@testset "parse_cubic_ppp_term" begin
    p_vi = v_idx(ParameterIndex(1))
    q_vi = v_idx(ParameterIndex(2))
    r_vi = v_idx(ParameterIndex(3))

    # 3 * p * q * r
    f = MOI.ScalarNonlinearFunction(:*, Any[3.0, p_vi, q_vi, r_vi])
    result = _parse_cubic_expression(f)

    @test result !== nothing
    ppp = _filter_ppp_terms(result)
    @test length(ppp) == 1
    @test ppp[1].coefficient == 3.0
end

# A13: Like terms should be combined
@testset "parse_cubic_term_combination" begin
    x, y = MOI.VariableIndex(1), MOI.VariableIndex(2)
    p_vi = v_idx(ParameterIndex(1))

    # x*y*p + 2*x*y*p = 3*x*y*p (should combine into single term)
    f = MOI.ScalarNonlinearFunction(:+, Any[
        MOI.ScalarNonlinearFunction(:*, Any[x, y, p_vi]),
        MOI.ScalarNonlinearFunction(:*, Any[2.0, x, y, p_vi])
    ])
    result = _parse_cubic_expression(f)

    @test result !== nothing
    pvv = _filter_pvv_terms(result)
    @test length(pvv) == 1  # combined into single term
    @test pvv[1].coefficient == 3.0
end

# A14: Nested ScalarAffineFunction inside ScalarNonlinearFunction
@testset "parse_cubic_nested_affine" begin
    x = MOI.VariableIndex(1)
    p_vi = v_idx(ParameterIndex(1))

    # (2x + 1) * p = 2*x*p + p (one pv term + one p term)
    affine_func = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(2.0, x)],
        1.0
    )
    f = MOI.ScalarNonlinearFunction(:*, Any[affine_func, p_vi])
    result = _parse_cubic_expression(f)

    @test result !== nothing
    # Check quadratic terms: should have 1 pv term (2*x*p)
    quad = _quadratic_terms(result)
    pv_terms = filter(t -> _is_parameter(t.variable_1) != _is_parameter(t.variable_2), quad)
    @test length(pv_terms) == 1
    # Check affine terms: should have 1 p term (1*p)
    affine = _affine_terms(result)
    p_affine = filter(t -> _is_parameter(t.variable), affine)
    @test length(p_affine) == 1
end

# A15: Mixed cubic expression with all term types
@testset "parse_cubic_all_term_types" begin
    x, y = MOI.VariableIndex(1), MOI.VariableIndex(2)
    p_vi = v_idx(ParameterIndex(1))
    q_vi = v_idx(ParameterIndex(2))

    # x*y*p + p*q*x + p*q*p + x*y + p*x + p*q + x + p + 5
    # pvv   + ppv   + ppp   + vv  + pv  + pp  + v + p + c
    f = MOI.ScalarNonlinearFunction(:+, Any[
        MOI.ScalarNonlinearFunction(:*, Any[x, y, p_vi]),           # pvv
        MOI.ScalarNonlinearFunction(:*, Any[p_vi, q_vi, x]),        # ppv
        MOI.ScalarNonlinearFunction(:*, Any[p_vi, q_vi, p_vi]),     # ppp (p²*q)
        MOI.ScalarNonlinearFunction(:*, Any[x, y]),                  # vv
        MOI.ScalarNonlinearFunction(:*, Any[p_vi, x]),               # pv
        MOI.ScalarNonlinearFunction(:*, Any[p_vi, q_vi]),            # pp
        x,                                                            # v
        p_vi,                                                         # p
        5.0                                                           # c
    ])
    result = _parse_cubic_expression(f)

    @test result !== nothing
    # Check cubic terms via filters
    @test length(_filter_pvv_terms(result)) == 1
    @test length(_filter_ppv_terms(result)) == 1
    @test length(_filter_ppp_terms(result)) == 1

    # Check quadratic terms via quadratic_func
    quad = _quadratic_terms(result)
    vv_terms = filter(t -> !_is_parameter(t.variable_1) && !_is_parameter(t.variable_2), quad)
    pv_terms = filter(t -> _is_parameter(t.variable_1) != _is_parameter(t.variable_2), quad)
    pp_terms = filter(t -> _is_parameter(t.variable_1) && _is_parameter(t.variable_2), quad)
    @test length(vv_terms) == 1
    @test length(pv_terms) == 1
    @test length(pp_terms) == 1

    # Check affine terms via quadratic_func
    affine = _affine_terms(result)
    v_affine = filter(t -> !_is_parameter(t.variable), affine)
    p_affine = filter(t -> _is_parameter(t.variable), affine)
    @test length(v_affine) == 1
    @test length(p_affine) == 1

    # Check constant
    @test _constant(result) == 5.0
end
```

#### Category B: ParametricCubicFunction Construction

```julia
# B1: Construct from parsed expression
@testset "cubic_function_construction" begin
    # Verify all term categories are correctly stored
end

# B2: Verify _current_function produces correct quadratic
@testset "cubic_function_current" begin
    # With p=2: 3*x*y*p -> 6*x*y (quadratic term)
end
```

#### Category C: JuMP Integration Tests (Primary)

These are the main validation tests using JuMP syntax.

```julia
# C1: Basic PVV term - parameter times quadratic
function test_jump_cubic_pvv_basic()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, 0 <= y <= 10)
    @variable(model, p in MOI.Parameter(2.0))

    # Minimize: x + y + p*x*y
    # With p=2: minimize x + y + 2*x*y
    # Subject to: x + y >= 2
    @constraint(model, x + y >= 2)
    @objective(model, Min, x + y + p * x * y)

    optimize!(model)
    @test termination_status(model) == OPTIMAL
    # At p=2, optimal is x=y=1, obj = 1+1+2*1*1 = 4
    @test objective_value(model) ≈ 4.0 atol=1e-6

    # Change p to 0 (removes cross term)
    set_parameter_value(p, 0.0)
    optimize!(model)
    # At p=0, optimal is x=y=1, obj = 1+1+0 = 2
    @test objective_value(model) ≈ 2.0 atol=1e-6
end

# C2: PPV term - two parameters times one variable
function test_jump_cubic_ppv_basic()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(2.0))
    @variable(model, q in MOI.Parameter(3.0))

    # Minimize: x + p*q*x = x * (1 + p*q)
    # With p=2, q=3: minimize x * (1 + 6) = 7x
    # Subject to: x >= 1
    @constraint(model, x >= 1)
    @objective(model, Min, x + p * q * x)

    optimize!(model)
    @test termination_status(model) == OPTIMAL
    # Optimal at x=1, obj = 7
    @test objective_value(model) ≈ 7.0 atol=1e-6

    # Change p=1, q=1: minimize x*(1+1) = 2x
    set_parameter_value(p, 1.0)
    set_parameter_value(q, 1.0)
    optimize!(model)
    @test objective_value(model) ≈ 2.0 atol=1e-6
end

# C3: PPP term - three parameters (constant contribution)
function test_jump_cubic_ppp_basic()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, p in MOI.Parameter(2.0))
    @variable(model, q in MOI.Parameter(3.0))
    @variable(model, r in MOI.Parameter(4.0))

    # Minimize: x + p*q*r
    # With p=2, q=3, r=4: minimize x + 24
    # Subject to: x >= 1
    @constraint(model, x >= 1)
    @objective(model, Min, x + p * q * r)

    optimize!(model)
    @test termination_status(model) == OPTIMAL
    # Optimal at x=1, obj = 1 + 24 = 25
    @test objective_value(model) ≈ 25.0 atol=1e-6

    # Change p=1, q=1, r=1: minimize x + 1
    set_parameter_value(p, 1.0)
    set_parameter_value(q, 1.0)
    set_parameter_value(r, 1.0)
    optimize!(model)
    @test objective_value(model) ≈ 2.0 atol=1e-6
end

# C4: Mixed cubic terms
function test_jump_cubic_mixed_terms()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, 0 <= x <= 10)
    @variable(model, 0 <= y <= 10)
    @variable(model, p in MOI.Parameter(1.0))
    @variable(model, q in MOI.Parameter(1.0))

    # Minimize: p*x*y + p*q*x + x*y + p*x + x + 10
    #           pvv   + ppv   + vv  + pv  + v + c
    # (no ppp term in this test for simplicity)
    @constraint(model, x + y >= 2)
    @objective(model, Min, 1.0*p*x*y + p*q*x + x*y + p*x + x + 10)

    # With p=1, q=1:
    # minimize: x*y + x + x*y + x + x + 10 = 2*x*y + 3x + 10
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    # Verify result matches hand calculation
end

# C5: Parameter changes affect optimization correctly
function test_jump_cubic_parameter_sensitivity()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, p in MOI.Parameter(0.0))

    @constraint(model, x + y == 1)
    # Minimize: x² + y² + p*x*y
    @objective(model, Min, x^2 + y^2 + p * x * y)

    # p=0: minimize x² + y² s.t. x+y=1
    # Solution: x=y=0.5, obj = 0.25 + 0.25 = 0.5
    optimize!(model)
    @test value(x) ≈ 0.5 atol=1e-6
    @test value(y) ≈ 0.5 atol=1e-6
    @test objective_value(model) ≈ 0.5 atol=1e-6

    # p=2: minimize x² + y² + 2xy = (x+y)² s.t. x+y=1
    # Any point on x+y=1 is optimal, obj = 1
    set_parameter_value(p, 2.0)
    optimize!(model)
    @test objective_value(model) ≈ 1.0 atol=1e-6
    @test value(x) + value(y) ≈ 1.0 atol=1e-6

    # p=-2: minimize x² + y² - 2xy = (x-y)² s.t. x+y=1
    # Optimal at x=1,y=0 or x=0,y=1, obj = 0 + corner effect
    set_parameter_value(p, -2.0)
    optimize!(model)
    # (x-y)² is minimized but x+y=1, so x² + y² - 2xy
    # At x=0.5,y=0.5: 0.25 + 0.25 - 0.5 = 0
    @test objective_value(model) ≈ 0.0 atol=1e-6
end
```

#### Category D: Edge Cases

```julia
# D1: Cubic that simplifies when p=0
@testset "cubic_parameter_zero" begin
    # When p=0, x*y*p = 0
    # If all quadratic terms also vanish, result should be affine
    # Test that _current_function returns correct type
end

# D1b: Parameter initially zero, then updated to non-zero
# CRITICAL: Expression must be parsed as cubic even when p=0 initially,
# so that updating p later correctly adds the quadratic term
function test_jump_cubic_parameter_initially_zero()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, p in MOI.Parameter(0.0))  # p = 0 initially

    @constraint(model, x + y >= 2)
    # Objective: p*x*y + x + y
    # With p=0: minimize 0 + x + y = x + y (effectively affine)
    @objective(model, Min, p * x * y + x + y)

    # First solve with p=0
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test objective_value(model) ≈ 2.0 atol=1e-6  # x=y=1, obj = 0 + 1 + 1 = 2

    # NOW update p to non-zero - this is the critical test!
    # The cubic term must have been stored, even though p was 0
    set_parameter_value(p, 2.0)
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    # With p=2: minimize 2*x*y + x + y s.t. x+y>=2
    # At x=y=1: obj = 2*1*1 + 1 + 1 = 4
    @test objective_value(model) ≈ 4.0 atol=1e-6

    # Update p back to 0 - should return to original behavior
    set_parameter_value(p, 0.0)
    optimize!(model)
    @test objective_value(model) ≈ 2.0 atol=1e-6
end

# D1c: Multiple cubic terms, some parameters zero
function test_jump_cubic_partial_zero_parameters()
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(model)

    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, p in MOI.Parameter(0.0))  # p = 0 initially
    @variable(model, q in MOI.Parameter(1.0))  # q = 1

    @constraint(model, x + y >= 2)
    # Objective: p*x*y + q*x*y + x + y
    # With p=0, q=1: minimize 0 + x*y + x + y
    @objective(model, Min, p * x * y + q * x * y + x + y)

    # First solve
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    # At x=y=1: obj = 0 + 1 + 1 + 1 = 3
    @test objective_value(model) ≈ 3.0 atol=1e-6

    # Update p to 2 (now both terms contribute)
    set_parameter_value(p, 2.0)
    optimize!(model)
    # With p=2, q=1: minimize 2*x*y + x*y + x + y = 3*x*y + x + y
    # At x=y=1: obj = 3 + 1 + 1 = 5
    @test objective_value(model) ≈ 5.0 atol=1e-6

    # Set q to 0 as well
    set_parameter_value(q, 0.0)
    optimize!(model)
    # With p=2, q=0: minimize 2*x*y + 0 + x + y = 2*x*y + x + y
    # At x=y=1: obj = 2 + 1 + 1 = 4
    @test objective_value(model) ≈ 4.0 atol=1e-6
end

# D2: Cubic with negative parameter
@testset "cubic_negative_parameter" begin
    # Verify sign handling is correct
end

# D3: Cubic term where variable_1 == variable_2
@testset "cubic_squared_variable" begin
    # x^2 * p (same variable twice)
end

# D4: Sign change warning for quadratic coefficients
function test_quadratic_sign_change_warning()
    # Enable the warning option
    inner_optimizer = HiGHS.Optimizer()
    model = POI.Optimizer(inner_optimizer; warn_on_quadratic_sign_change = true)
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    p, _ = MOI.add_constrained_variable(model, MOI.Parameter(2.0))

    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))
    MOI.add_constraint(model,
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, x),
            MOI.ScalarAffineTerm(1.0, y)
        ], 0.0),
        MOI.GreaterThan(1.0)
    )

    # Objective: p*x*y (starts with positive coefficient when p=2)
    obj = MOI.ScalarNonlinearFunction(:*, Any[p, x, y])
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}(), obj)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(model)

    # Change p from +2 to -2: should trigger warning
    # The quadratic coefficient changes from +2 to -2 (sign change!)
    @test_logs (:warn, r"Quadratic coefficient sign change") begin
        MOI.set(model, POI.ParameterValue(), p, -2.0)
        POI.update_parameters!(model)
    end

    # Change p from -2 to -1: no warning (same sign)
    @test_logs min_level=Logging.Warn begin
        MOI.set(model, POI.ParameterValue(), p, -1.0)
        POI.update_parameters!(model)
    end
end

# D5: No warning when option is disabled (default)
function test_quadratic_sign_change_no_warning_by_default()
    # Default: warn_on_quadratic_sign_change = false
    inner_optimizer = HiGHS.Optimizer()
    model = POI.Optimizer(inner_optimizer)  # default options
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    p, _ = MOI.add_constrained_variable(model, MOI.Parameter(2.0))

    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))

    obj = MOI.ScalarNonlinearFunction(:*, Any[p, x, y])
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}(), obj)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(model)

    # Sign change but no warning because option is disabled
    @test_logs min_level=Logging.Warn begin
        MOI.set(model, POI.ParameterValue(), p, -2.0)
        POI.update_parameters!(model)
    end
end
```

### 5.3 Simple Example Test Case

**Test: `test_cubic_objective_simple`**

```julia
function test_cubic_objective_simple()
    # Setup: Simple QP that we can solve by hand
    #
    # minimize: x² + y² + x*y*p
    # subject to: x + y >= 1
    #             x, y >= 0
    #
    # When p = 0: minimize x² + y² s.t. x+y>=1
    #   Solution: x = y = 0.5, objective = 0.5
    #
    # When p = 2: minimize x² + y² + 2xy = (x+y)²
    #   Solution: Any point on x+y=1, objective = 1
    #   With x,y >= 0, optimal at x=1,y=0 or x=0,y=1 or any convex combo

    model = POI.Optimizer(HiGHS.Optimizer())
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    p, ci_p = MOI.add_constrained_variable(model, MOI.Parameter(0.0))

    # Bounds
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))

    # Constraint: x + y >= 1
    MOI.add_constraint(model,
        MOI.ScalarAffineFunction([
            MOI.ScalarAffineTerm(1.0, x),
            MOI.ScalarAffineTerm(1.0, y)
        ], 0.0),
        MOI.GreaterThan(1.0)
    )

    # Objective: x² + y² + x*y*p (as ScalarNonlinearFunction)
    obj = MOI.ScalarNonlinearFunction(:+, Any[
        MOI.ScalarNonlinearFunction(:^, Any[x, 2]),
        MOI.ScalarNonlinearFunction(:^, Any[y, 2]),
        MOI.ScalarNonlinearFunction(:*, Any[x, y, p])
    ])
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}(), obj)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # Solve with p = 0
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 0.5 atol=1e-6

    # Update p = 2 and re-solve
    MOI.set(model, POI.ParameterValue(), p, 2.0)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1.0 atol=1e-6
end
```

---

## Part 6: Implementation Order

### Phase 1: Foundation (Tests First)

1. **Write parser tests** (`test/parser_tests.jl`)
   - Test expression tree traversal
   - Test monomial classification
   - Test valid/invalid detection

2. **Implement parser** (`src/cubic_parser.jl`)
   - `_parse_cubic_expression`
   - `_expand_expression`
   - `_classify_monomial`
   - `_is_polynomial_operator`

3. **Validate parser tests pass**

### Phase 2: Data Structure

4. **Write ParametricCubicFunction tests**
   - Construction tests
   - `_current_function` tests

5. **Implement ParametricCubicFunction** (in `src/parametric_cubic_function.jl` - new file)
   - Struct definition
   - Constructor from `ParsedCubicExpression`
   - `_current_function` (returns `ScalarQuadraticFunction` or `ScalarAffineFunction`)
   - `_update_cache!`
   - `_original_function` (reconstruct original expression)

6. **Validate data structure tests pass**

### Phase 3: Integration

7. **Write objective integration tests**
   - Setting cubic objectives
   - Parameter updates
   - Optimization verification

8. **Implement MOI integration** (in `src/MOI_wrapper.jl`)
   - Add `cubic_objective_cache` to Optimizer
   - `MOI.set` for `ObjectiveFunction{ScalarNonlinearFunction}`
   - `MOI.get` for objective retrieval
   - Extend `_update_parameters!`

9. **Validate all tests pass**

### Phase 4: Documentation

10. **Add docstrings** to all public functions

11. **Update package documentation**

### Phase 5: Tutorials (Post-Validation)

12. **Progressive Hedging example** (after code is stable)

---

## Part 7: File Organization

**Design principle**: Most new code should be in **separate new files** to minimize changes to existing files. This keeps the codebase modular and reduces merge conflicts.

### New Files (bulk of the implementation)

```
src/
├── cubic_types.jl               # NEW: _ScalarCubicTerm{T} struct and helpers
├── cubic_parser.jl              # NEW: _parse_cubic_expression and helpers
├── parametric_cubic_function.jl # NEW: ParametricCubicFunction struct and methods
└── cubic_objective.jl           # NEW: MOI objective setting/getting for cubic

test/
├── cubic_parser_tests.jl        # NEW: Parser unit tests
└── cubic_jump_tests.jl          # NEW: JuMP integration tests for cubic
```

### Minimal Changes to Existing Files

```
src/
├── ParametricOptInterface.jl    # MODIFY: Add includes and exports (few lines)
├── MOI_wrapper.jl               # MODIFY: Add cubic_objective_cache field to Optimizer
│                                #         Add dispatch to _update_parameters!
└── update_parameters.jl         # MODIFY: Call _update_cubic_objective! (few lines)

test/
├── runtests.jl                  # MODIFY: Include new test files
```

### File Responsibilities

| File | Responsibility | Lines (est.) |
|------|----------------|--------------|
| `cubic_types.jl` | `_ScalarCubicTerm{T}`, helpers, accessors | ~60 |
| `cubic_parser.jl` | Expression tree parsing | ~200 |
| `parametric_cubic_function.jl` | Main data structure + methods | ~250 |
| `cubic_objective.jl` | MOI integration for objectives | ~150 |
| `cubic_parser_tests.jl` | Parser unit tests | ~300 |
| `cubic_jump_tests.jl` | JuMP integration tests | ~400 |

### Include Order in ParametricOptInterface.jl

```julia
# Add after existing includes:
include("cubic_types.jl")
include("cubic_parser.jl")
include("parametric_cubic_function.jl")
include("cubic_objective.jl")
```

---

## Part 8: Open Questions / Considerations

### Q1: What if the solver doesn't support quadratic objectives?

When `p` is substituted, the cubic becomes quadratic (or affine if all quadratic terms vanish).

**Considerations**:
- If the inner optimizer doesn't support quadratic objectives but all quadratic terms have zero coefficients (e.g., all PVV parameters are 0), we can still proceed with an affine objective
- If quadratic terms are non-zero and the solver doesn't support them, throw a clear error
- `_current_function` should return the simplest possible type (`ScalarAffineFunction` when possible)

**Decision**: Check `MOI.supports` dynamically based on the actual result of `_current_function`.

### Q2: Should we support cubic in constraints?

The user specified **objectives only**. This simplifies implementation significantly since:
- We don't need to handle constraint modifications
- We don't need dual computation for cubic constraints
- The inner optimizer sees only quadratic/affine constraints

**Decision**: No constraint support in this implementation.

### Q3: How to handle duals for cubic objectives?

When a parameter appears in a cubic term, the dual interpretation is more complex. For now:
- Focus on primal optimization
- Document that dual sensitivity for cubic terms is not supported initially

### Q3b: Should we accept ScalarNonlinearFunction with no cubic terms?

If a user passes a `ScalarNonlinearFunction` that parses successfully but contains only quadratic/affine/constant terms (no PVV, PPV, or PPP), should we:

**Option A**: Accept it and store in `cubic_objective_cache`
- Pro: Consistent handling of all ScalarNonlinearFunction
- Con: Overhead of cubic infrastructure for non-cubic functions

**Option B**: Reject it with a helpful error suggesting to use ScalarQuadraticFunction
- Pro: Encourages proper function types
- Con: May be overly strict

**Proposed**: Option A - accept and handle it. The overhead is minimal and it provides a smoother user experience.

### Q4: JuMP integration

Users will write `@objective(model, Min, x*y*p)` in JuMP.

**Key insight**: If we correctly implement `MOI.set` for `ObjectiveFunction{ScalarNonlinearFunction}`, JuMP will automatically work because:
1. JuMP detects that `x*y*p` involves three "variables" (including the parameter)
2. JuMP constructs a `ScalarNonlinearFunction` for expressions beyond quadratic
3. JuMP calls `MOI.set(model, ObjectiveFunction{ScalarNonlinearFunction}(), f)`
4. Our implementation parses and handles it

**Testing**: Full model tests should use JuMP syntax (`test/jump_tests.jl`) as these validate the end-to-end user experience.

---

---

## Part 9: Verification Against Codebase

This section documents verification of the plan against actual MOI, JuMP, and POI code.

### 9.1 MOI ScalarNonlinearFunction (Verified ✓)

**Location**: `MathOptInterface/src/functions.jl` (lines 276-351)

**Confirmed structure:**
```julia
struct ScalarNonlinearFunction <: AbstractScalarFunction
    head::Symbol
    args::Vector{Any}
end
```

**Confirmed valid arg types:**
- `T <: Real` (constants)
- `VariableIndex`
- `ScalarAffineFunction`
- `ScalarQuadraticFunction`
- `ScalarNonlinearFunction` (nested)

**Confirmed operators:**
- Multivariate: `:+`, `:-`, `:*`, `:^`, `:/`, `:ifelse`, `:min`, `:max`
- Unary: `:-` (negation), plus all math functions

**Plan alignment**: ✓ Our parsing strategy correctly handles these types and operators.

### 9.2 POI ParametricQuadraticFunction (Verified ✓)

**Location**: `ParametricOptInterface/src/parametric_functions.jl` (lines 18-295)

**Confirmed patterns:**
```julia
mutable struct ParametricQuadraticFunction{T} <: ParametricFunction{T}
    affine_data::Dict{MOI.VariableIndex,T}        # Variables in pv terms
    affine_data_np::Dict{MOI.VariableIndex,T}     # Variables NOT in pv terms
    pv::Vector{MOI.ScalarQuadraticTerm{T}}
    pp::Vector{MOI.ScalarQuadraticTerm{T}}
    vv::Vector{MOI.ScalarQuadraticTerm{T}}
    p::Vector{MOI.ScalarAffineTerm{T}}
    v::Vector{MOI.ScalarAffineTerm{T}}
    c::T
    set_constant::T
    current_terms_with_p::Dict{MOI.VariableIndex,T}
    current_constant::T
end
```

**Key helper functions:**
- `_split_quadratic_terms()` - Categorizes into vv/pp/pv
- `_split_affine_terms()` - Categorizes into v/p
- `_parametric_constant()` - Computes constant with parameter values
- `_parametric_affine_terms()` - Computes affine coefficients with parameters
- `_is_parameter()` / `_is_variable()` - Index classification

**Plan alignment**: ✓ Updated ParametricCubicFunction to follow the same Dict-based caching pattern.

### 9.3 Current POI Limitation (Verified ✓)

**Confirmed**: POI currently only supports up to quadratic expressions.

**From documentation** (`docs/src/manual.md`):
- Supported: `ScalarAffineFunction`, `ScalarQuadraticFunction`, `VectorAffineFunction`
- NOT supported: `ScalarNonlinearFunction`

**From tests** (`test/jump_tests.jl`, lines 854-862):
```julia
function test_jump_nlp()
    # ... nonlinear objective throws ErrorException
    @test_throws ErrorException optimize!(model)
end
```

**Plan alignment**: ✓ This confirms our implementation fills a real gap - cubic expressions with parameters are not currently supported.

### 9.4 JuMP Expression Generation

**Confirmed behavior:**
- JuMP treats parameters (via `MOI.Parameter`) as special `VariableIndex` values
- For expressions beyond quadratic degree, JuMP creates `ScalarNonlinearFunction`
- Expression `p * x * y` would generate a nonlinear function (currently rejected by POI)

**Plan alignment**: ✓ When we implement `MOI.set` for `ObjectiveFunction{ScalarNonlinearFunction}`, JuMP's `@objective(model, Min, p*x*y)` will work automatically.

### 9.5 Parameter Index Threshold

**Location**: `ParametricOptInterface/src/ParametricOptInterface.jl` (lines 21-38)

```julia
const PARAMETER_INDEX_THRESHOLD = 4_611_686_018_427_387_904
```

**Plan alignment**: ✓ Our parser must use `_is_parameter()` to distinguish parameters from variables when classifying monomials.

### 9.6 MOI Utilities Available (Verified ✓)

**Location**: `MathOptInterface/src/Utilities/functions.jl` and `MathOptInterface/src/Nonlinear/`

**Available utilities that could simplify implementation:**

| Utility | What it does | Potential use |
|---------|--------------|---------------|
| `substitute_variables(fn, f)` | Replace variables via mapping function | Limited - returns SNF not polynomial |
| `canonical(f)` | Normalize (combine terms, sort) | Post-processing parsed result |
| `map_indices(fn, f)` | Remap variable indices in tree | Index transformations |
| `operate(op, T, args...)` | Compose functions with +, -, *, etc. | Building result functions |
| `eval_variables(value_fn, model, f)` | Evaluate expression numerically | Not useful - we need symbolic result |

**Decision**: Use `canonical()` and `operate()` where beneficial. Build custom monomial expansion logic since MOI doesn't provide polynomial-specific utilities.

**Key finding**: MOI's `substitute_variables` cannot convert `ScalarNonlinearFunction` to `ScalarQuadraticFunction` - it preserves the nonlinear type. Our custom parser is necessary.

---

## Summary

This plan provides a structured approach to implementing parametric cubic functions:

1. **Parse** `ScalarNonlinearFunction` to detect valid cubic polynomials
2. **Store** in `ParametricCubicFunction` with proper term categorization
3. **Integrate** with POI's objective handling (objectives only)
4. **Test** thoroughly with simple, predictable examples
5. **Document** all functions

The key insight is that when parameters are substituted with values, a cubic function `c*x*y*p` becomes a quadratic function `c*p_val*x*y`, which existing solvers can handle.
