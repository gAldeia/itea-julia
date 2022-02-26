
using LRUCache #https://github.com/JuliaCollections/LRUCache.jl




import Base

struct IT
    g :: Function #TODO: não usar Function
    k :: Array{Float64, 1} # must be floats to use the exponent function with negative powers

    inner_c :: Float64
    outer_c :: Float64

    IT(g, k) = new(g, k, 1.0, 1.0)
    IT(g, k, inner_c, outer_c) = new(g, k, inner_c, outer_c)
end

# hash usado para comparar com isequal
Base.hash(it::IT, h::UInt)::UInt = hash(it.k, hash(String(Symbol(it.g)), hash(:IT, h)))

# Aqui usamos o hash para comparar
Base.:(==)(it1::IT, it2::IT) = isequal(it1.k, it2.k) && isequal(String(Symbol(it1.g)), String(Symbol(it2.g)))

# Usado para ordenar. Precisa do equal e less para isso
Base.isless(it1::IT, it2::IT) = isless(String(Symbol(it1.g)), String(Symbol(it2.g))) || isless(it1.k, it2.k)


struct ITexpr
    ITs       :: Array{IT, 1}
    intercept :: Float64

    nterms    :: Int32

    # OBS: ordenação é feita pelas its sem considerar coeficiente!!
    # A ideia é só pegar termos similares, independente dos coeficientes
    ITexpr(ITs) = new(sort(ITs), 0.0, length(ITs))
    ITexpr(ITs, intercept) = new(sort(ITs), intercept, length(ITs))
end

# TODO: adicionar nterms aqui
Base.hash(itexpr::ITexpr, h::UInt) = hash(itexpr.ITs, hash(itexpr.nterms, hash(:ITexpr, h)))
Base.:(==)(itexpr1::ITexpr, itexpr2::ITexpr) = isequal(itexpr1.ITs, itexpr2.ITs) && isequal(itexpr1.nterms, itexpr2.nterms)


# Tentando memorizar e recuperar
# https://github.com/JuliaCollections/LRUCache.jl
const fitness_memoization = LRU{ITexpr, Float64}(maxsize = 5)


function fitness(itexpr::ITexpr)
    println("Calculando fitness... (1 sec)")
    sleep(1)
    return rand()
end

function cached_fitness(itexpr::ITexpr)
    get!(fitness_memoization, itexpr) do
        fitness(itexpr)
    end
end

function Base.show(io::IO, itexpr::ITexpr)
    println("$(size(itexpr.ITs, 1))-element IT expression:")
    map(itexpr.ITs) do it
        println(" $(round(it.outer_c, digits=3)) * $(it.g)( $(round(it.inner_c, digits=3)) * p(X, $(it.k)) )")    
    end
    println(" $(itexpr.intercept)")
end

# Deve printar iguais
println(ITexpr([IT(identity, [0, 1]), IT(identity, [1, 0])]))
println(ITexpr([IT(identity, [1, 0]), IT(identity, [0, 1])]))

#
#println(sort([IT(identity, [0, 1]), IT(identity, [1, 0])]))
#println(sort([IT(identity, [1, 0]), IT(identity, [0, 1])]))

# Todas devem ser iguais
println( cached_fitness(ITexpr([IT(identity, [0, 1]), IT(identity, [1, 0])])) )
println( cached_fitness(ITexpr([IT(identity, [1, 0]), IT(identity, [0, 1])])) )
println( cached_fitness(ITexpr([IT(identity, [1, 0]), IT(identity, [0, 1])])) )

# Diferente de todas as anteriores
println( cached_fitness(ITexpr([IT(identity, [1, 1]), IT(identity, [0, 1])])) )
println( cached_fitness(ITexpr([IT(identity, [1, 1]), IT(identity, [1, 1])])) )

# Igual a anterior
println( cached_fitness(ITexpr([IT(identity, [1, 1]), IT(identity, [1, 1])])) )


# No total ele tem que calcular 3 fitness

# TODO: fazer o ITEA memorizar o fitness e os coeficientes

using StatsBase: rle

println(rle(
    [IT(identity, [1, 1]), IT(identity, [1, 1]), IT(identity, [1, 0]), IT(identity, [0, 1])]
))
