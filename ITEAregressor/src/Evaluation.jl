# evaluate, predict (alias para evaluate), gradient

# TODO: tratamento numérico, se feito, deve ser aqui no evaluate
function evaluate(it::IT, X::Array{T, 2}) where {T<:Number}
    it.outer_c * it.g.(it.inner_c * vec(prod(X .^ it.k', dims=2)) )
end

function evaluate(itexpr::ITexpr, X::Array{T, 2}) where {T<:Number}
    mapfoldl(it -> evaluate(it, X), .+, itexpr.ITs) .+ itexpr.intercept
end


# Providenciando facilidade para quando o usuário passa só 1 input. Recebe qualquer
# array e possivelmente manda um erro na hora da chamada da função que recebe uma
# matriz
evaluate(it::IT, X::Array{Any, 1}) = evaluate(it, reshape(X, (size(X)...,1)))
evaluate(itexpr::ITexpr, X::Array{Any, 1}) = evaluate(itexpr, reshape(X, (size(X)...,1)))


# Predict é um alias para evaluate
const predict = evaluate


function gradient(it::IT, X::Array{T, 2}) where {T<:Number}

    nsamples, nvars = size(X)

    gradients = zeros(size(X))

    # Interação sem considerar a regra do expoente da derivada
    p_eval = vec(prod(X .^ it.k', dims=2))

    for i in 1:nvars
        power_rule = copy(it.k)
        power_rule[i] -= 1

        p_partial_eval = it.k[i] * vec(prod(X .^ power_rule', dims=2))

        # obtendo a derivada da função de transformação
        g = x -> derivative(it.g, x)
        
        # Regra do expoente e da cadeia
        gradients[:, i] = it.outer_c * g.(it.inner_c * p_eval) .* (it.inner_c * p_partial_eval)
    end

    gradients
end


function gradient(itexpr::ITexpr, X::Array{T, 2}) where {T<:Number}
    mapfoldl(it -> gradient(it, X), .+, itexpr.ITs) .+ itexpr.intercept
end


gradient(it::IT, X::Array{Any, 1}) = gradient(it, reshape(X, (size(X)...,1)))
gradient(itexpr::ITexpr, X::Array{Any, 1}) = gradient(itexpr, reshape(X, (size(X)...,1)))

# TODO: implementar mais coisas que recebem 1 ou 2 argumentos se fizer sentido