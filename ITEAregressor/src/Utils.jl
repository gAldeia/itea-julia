function Base.show(io::IO, it::IT)
    println("$(size(it.k, 1))-element IT term:")
    println(" g (transformation function) : $(it.g)")
    println(" k (strengths vector)        : $(it.k)")
    println(" w (outer scale coefficient) : $(it.w)")
    println(" b (inner offset)            : $(it.b)")
    println(" c (inner scale coefficient) : $(it.c)")
end


# TODO: fazer um parser para pegar string e criar expr

function Base.show(io::IO, itexpr::ITexpr)
    println("$(size(itexpr.ITs, 1))-element IT expression:")
    map(itexpr.ITs) do it
        println(
            " $(round(it.w, digits=3)) * $(it.g)",
            "( $(round(it.b, digits=3)) + $(round(it.c, digits=3)) * p(X, $(it.k)) )")    
    end
    println(" $(itexpr.intercept)")
end


# function Base.show(io::IO, itpop::ITpop)
#     println("$(itpop.size)-element IT population:")
#     map(itpop.ITexprs) do itexpr
#         println(" $(size(itexpr.ITs, 1))-element IT expression:")
#         map(itexpr.ITs) do it
#             println(
#                 "  $(round(it.w, digits=3)) * $(it.g)",
#                 "( $(round(it.b, digits=3)) + $(round(it.c, digits=3)) * p(X, $(it.k)) )")    
#         end
#         println("  $(itexpr.intercept)")
#     end
# end


function to_str(it::IT; digits::Int=5, labels::Array{String,1}=String[])

    # TODO: to str ignorar quando temos expoente zero
    w = round(it.w, digits=digits)
    c = round(it.c, digits=digits)
    b = round(it.b, digits=digits)
    
    interaction = join(filter(x -> !isnothing(x), map(1:length(it.k)) do i
        it.k[i] == 0.0 ? nothing : begin
            var_name = length(labels) >= i ? labels[i] : "x_$(i)"

            "$(var_name)^$(it.k[i])"
        end
    end), " * ")

    "$(w)*$(it.g)($(b)+$(c)*($(interaction)))"
end


function to_str(itexpr::ITexpr; digits::Int=5, labels::Array{String,1}=String[])
    
    its = join(map(itexpr.ITs) do it
        to_str(it, digits=digits, labels=labels)
    end, " + ")

    "$(its) + $(round(itexpr.intercept))"
end


function count_nodes(it::IT)

    if it.w != 0.0 # w * ...
        2 + if it.b != 0 # w * g(b + ...)
            3 # g(b + ...)
        else
            1 #g(...)
        end + if it.c != 0
            2 + sum(map(it.k) do k
                k== 0 ? 0 : 3 # 3 nodes (x ^ k) for each non-zero exponent
            end)
        else
            0
        end
    else
        0
    end
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