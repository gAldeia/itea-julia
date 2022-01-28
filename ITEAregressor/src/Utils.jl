# TODO: to_str, size, count nodes, etc

# TODO: respeitar régua na coluna 80

function Base.show(io::IO, it::IT)
    println("$(size(it.k, 1))-element IT term:")
    println(" g       : $(it.g)")
    println(" k       : $(it.k)")
    println(" inner_c : $(it.inner_c)")
    println(" outer_c : $(it.outer_c)")
end

function Base.show(io::IO, itexpr::ITexpr)
    println("$(size(itexpr.ITs, 1))-element IT expression:")
    map(itexpr.ITs) do it
        println(" $(round(it.outer_c, digits=3)) * $(it.g)( $(round(it.inner_c, digits=3)) * p(X, $(it.k)) )")    
    end
    println(" $(itexpr.intercept)")
end


function Base.show(io::IO, itpop::ITpop)
    println("$(itpop.size)-element IT population:")
    map(itpop.ITexprs) do itexpr
        println(" $(size(itexpr.ITs, 1))-element IT expression:")
        map(itexpr.ITs) do it
            println("  $(round(it.outer_c, digits=3)) * $(it.g)( $(round(it.inner_c, digits=3)) * p(X, $(it.k)) )")    
        end
        println("  $(itexpr.intercept)")
    end
end

# TODO: um print formatado