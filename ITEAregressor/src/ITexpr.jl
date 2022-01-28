

"""
outer_c * g( inner_c * interaction(k) )
"""
struct IT
    g :: Function
    k :: Array{Float64, 1} # must be floats to use the exponent function with negative powers

    inner_c :: Float64
    outer_c :: Float64

    IT(g, k) = new(g, k, 1.0, 1.0)
    IT(g, k, inner_c, outer_c) = new(g, k, inner_c, outer_c)
end

# TODO: fazer itexpr e it terem mais informações (nvar, size)
struct ITexpr
    ITs       :: Array{IT, 1}
    intercept :: Float64

    # TODO: ter nterms e substituir todo uso de size no código todo

    ITexpr(ITs) = new(ITs, 0.0)
    ITexpr(ITs, intercept) = new(ITs, intercept)
end


struct ITpop
    ITexprs :: Array{ITexpr, 1}
    size::Int64 # TODO: renomear para length? popsize?

    ITpop(ITexprs) = new(ITexprs, size(ITexprs, 1))
end