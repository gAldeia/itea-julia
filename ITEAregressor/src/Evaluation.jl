# evaluate (for evolutionary algorithm) and callable (for ease of use) methods


# Evaluation should take a list of samples (matrix X)

function evaluate(it::IT, X::Array{T, 2}) where {T<:Number}

    # will generate a n x 1 matrix (since X is always a matrix).
    # We want it as a vector (so that we use [:, 1])
    interaction = @view prod(X .^ it.k', dims=2)[:, 1]

    @. it.w * it.g(it.b + it.c * interaction)
end


function evaluate(itexpr::ITexpr, X::Array{T, 2}) where {T<:Number}
    out = zeros(size(X, 1))

    map(itexpr.ITs) do it
        out .+= evaluate(it, X)
    end
    
    out .+ itexpr.intercept
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