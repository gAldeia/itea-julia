# evaluate (for evolutionary algorithm) and callable (for ease of use) methods


# Evaluation should take a list of samples (matrix X)

function evaluate(it::IT, X::Array{T, 2}) where {T<:Number}
    it.outer_c * it.g.(it.inner_c * vec(prod(X .^ it.k', dims=2)) )
end


function evaluate(itexpr::ITexpr, X::Array{T, 2}) where {T<:Number}
    mapfoldl(it -> evaluate(it, X), .+, itexpr.ITs) .+ itexpr.intercept
end


# if a single sample is given, then we reshape it and use the correct method

function evaluate(it::IT, X::Array{T, 1}) where {T<:Number}
    evaluate(it, reshape(X, (size(X)...,1)))
end


function evaluate(itexpr::ITexpr, X::Array{T, 1}) where {T<:Number}
    evaluate(itexpr, reshape(X, (size(X)...,1)))
end


# The predict method provides an easier way of predicting new values
# without having to pass the expressions

(it::IT)(X) = evaluate(it, X)

(itexpr::ITexpr)(X) = evaluate(itexpr, X)