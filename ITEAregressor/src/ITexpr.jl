# Interaction-Transformation structs for representing mathematical expressions
# as the linear combination of IT terms.


"""
IT(x) = outer_c * g( inner_c * interaction(x, k) )
"""
struct IT
    g :: Function
    k :: Array{Float16, 1} # have to be float to use ^ with negative strengths.

    w :: Float64 # outer scale coefficient
    b :: Float64 # inner offset
    c :: Float64 # inner scale coefficient

    # Default values are the neutral elements for the + and * operations
    IT(g, k) = new(g, k, 1.0, 0.0, 1.0)
    IT(g, k, w, b, c) = new(g, k, w, b, c)
end


"""
ITexpr(x) = IT_1(x) + IT_2(x) + ... + intercept
"""
struct ITexpr
    ITs       :: Array{IT, 1}
    intercept :: Float64

    nterms    :: Int32 # TODO: Usar nterms e nvars no código
    nvars     :: Int32 #TODO: não ter nvars

    ITexpr(ITs, intercept=0.0) = begin
        # Lista de ITs é criada (ao menos pelas funções internas da biblioteca)
        # de forma que nunda existe termos triviais (mas pode ter repetido,
        # por isso usamos o rle). Ordenamos com base nos métodos de hash e 
        # comparação para poder usar memoization.
        unique_its, n_occurences = rle(sort(ITs))

        new(unique_its, intercept, length(unique_its), length(ITs[1].k))
    end
end


# Creating hashs for IT structs so ITExprs can be ordered and memoized. 
# Base.hash creates a hash representation of expressions, and (==) and isless
# are used in sorting methods. When creating ITexprs, you need to sort
# the ITs list to help memoization to identify similar expressions. Hashing
# and sorting does not consider the values of the coefficients or intercept.
# None of those methods are exported.

str_rep(T) = String(Symbol(T))

function Base.hash(it::IT, h::UInt)::UInt
    hash(it.k, hash(String(Symbol(it.g)), hash(:IT, h)))
end

function Base.:(==)(it1::IT, it2::IT)
    isequal(it1.k, it2.k) && isequal(str_rep(it1.g), str_rep(it2.g))
end

function Base.isless(it1::IT, it2::IT) 
    isless(str_rep(it1.g), str_rep(it2.g)) || isless(it1.k, it2.k)
end

function Base.hash(itexpr::ITexpr, h::UInt)
    hash(itexpr.ITs, hash(itexpr.nterms, hash(:ITexpr, h)))
end

function Base.:(==)(itexpr1::ITexpr, itexpr2::ITexpr) 
    isequal(itexpr1.ITs, itexpr2.ITs) && isequal(itexpr1.nterms, itexpr2.nterms)
end