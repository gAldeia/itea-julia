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
        println(" $(round(it.outer_c, digits=3)) * $(it.g)",
                "( $(round(it.inner_c, digits=3)) * p(X, $(it.k)) )")    
    end
    println(" $(itexpr.intercept)")
end


function Base.show(io::IO, itpop::ITpop)
    println("$(itpop.size)-element IT population:")
    map(itpop.ITexprs) do itexpr
        println(" $(size(itexpr.ITs, 1))-element IT expression:")
        map(itexpr.ITs) do it
            println("  $(round(it.outer_c, digits=3)) * $(it.g)",
                    "( $(round(it.inner_c, digits=3)) * p(X, $(it.k)) )")    
        end
        println("  $(itexpr.intercept)")
    end
end


function to_str(it::IT; digits::Int=5, labels::Array{String,1}=String[])

    outer_c = round(it.outer_c, digits=digits)
    inner_c = round(it.inner_c, digits=digits)
    
    interaction = join(map(1:length(it.k)) do i
        var_name = length(labels) >= i ? labels[i] : "x_$(i)"
        "$(var_name)^$(it.k[i])"
    end, " * ")

    "$(outer_c) * $(it.g)($(inner_c) * $(interaction))"
end


function to_str(itexpr::ITexpr; digits::Int=5, labels::Array{String,1}=String[])
    
    its = join(map(itexpr.ITs) do it
        to_str(it, digits=digits, labels=labels)
    end, " + ")

    "$(its) + $(round(itexpr.intercept))"
end


function count_nodes(it::IT)

    interaction = sum(map(it.k) do k
        k== 0 ? 0 : 3 # 3 nodes (x ^ k) for each non-zero exponent
    end)

    # 2 nodes for each coefficient: the coeff itself and its multiplication op.
    # 1 node for the transformation function
    return 5 + interaction
end


function count_nodes(itexpr::ITexpr)

    its = sum(map(itexpr.ITs) do it
        count_nodes(it) + 1 # plus 1 for each sum between terms (and 
                            # between the last term and intercept)
    end)

    # Discounting the sum node of the intercept if it is zero. Otherwise,
    # count the intercept as well
    itexpr.intercept == 0 ? its - 1 : its + 1
end


# Additional transformation functions
sqrtabs(x) = sqrt.(abs.(x))