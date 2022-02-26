#import ITEAregressor: ITEA, RMSE

using IntervalArithmetic


struct IT
    g :: Function #TODO: não usar func
    k :: Array{Float64, 1} # must be floats to use the exponent function with negative powers

    inner_c :: Float64
    outer_c :: Float64

    IT(g, k) = new(g, k, 1.0, 1.0)
    IT(g, k, inner_c, outer_c) = new(g, k, inner_c, outer_c)
end

function evaluate(it::IT, X::Array{T, 2}) where {T<:Number}
    it.outer_c * it.g.(it.inner_c * vec(prod(X .^ it.k', dims=2)) )
end

# TODO: parar de usar vec? trabalhar com matrizes desde o começo?

# Versão pra chamar a IT. Ou passamos um vetor com uma ou com várias observações, sempre
# deve retornar matriz
(it::IT)(args...) = it.outer_c * it.g.(it.inner_c * prod((args...) .^ it.k', dims=2) )

it = IT(identity, [1., 2.])

# TODO: isso tá errado...
a = IntervalBox(1..4, 1..5)

# https://juliaintervals.github.io/IntervalArithmetic.jl/latest/multidim/
println( it([1., 1.]) )
println( size(it([1., 1.])) )
println(it(a))
println(IntervalBox(it(a)))