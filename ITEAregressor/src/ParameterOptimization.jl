"""
Ajusta apenas os parâmetros lineares (outer_c) de uma expressão ITexpr
"""
function ordinary_least_squares_adj(
    itexpr::ITexpr, X::Array{T, 2}, Y::Array{T, 1}) where {T<:Number}

    nsamples, nvars = size(X)
    nitterms = size(itexpr.ITs, 1)

    try
        # ones no final para ser a coluna do fit intercepto
        Xhat = hcat(map(it -> evaluate(it, X), itexpr.ITs)..., ones(nsamples))
    
        # vec transforma (se já não for) em vetor coluna
        beta = pinv(Xhat' * Xhat) * Xhat' * vec(Y)

        intercept = beta[end]
        ITs = map(1:nitterms) do i
            g       = itexpr.ITs[i].g
            k       = itexpr.ITs[i].k
            outer_c = itexpr.ITs[i].outer_c

            IT(g, k, 1.0, outer_c * beta[i])
        end
    
        ITexpr(ITs, intercept)
    catch err 
        if isa(err, DomainError)
            return itexpr
        else
            rethrow(err)
        end
    end
end


"""
se a itexpr tem n termos, então theta deve ter 2n + 1 elementos 
(um inner e outer para cada termo + intercepto, na ordem)
inner1, outer1, inner2, outer2, ..., intercept
"""
function _replace_parameters(
    itexpr::ITexpr, theta::Array{Float64, 1}) where {T<:Number}
    
    nitterms = size(itexpr.ITs, 1)

    ITs = map(collect(1:nitterms)) do i
        g = itexpr.ITs[i].g
        k = itexpr.ITs[i].k

        IT(g, k, theta[2i], theta[2i+1])
    end

    ITexpr(ITs, theta[end])
end


# TODO: melhorar uso de dispatch, colocar tipo de retorno e argumento em tudo

# TODO: revisar isso (pq o erro não é pequeno igual OLS pra um problema trivial?)
function levenberg_marquardt_adj(
    itexpr::ITexpr, X::Array{T, 2}, Y::Array{T, 1}) where {T<:Number}

    #TODO: usar y minúsculo

    # Evaluate replacing constants without modifying them
    jacobian = (X, theta) -> begin
        # Cada coluna é a derivada em função de um dos parâmetros de theta
        J = Array{Float64}(undef, size(X, 1), 2*size(itexpr.ITs, 1)+1)

        # Lembrando que theta é:
        # inner1, outer1, inner2, outer2, ..., intercept
        for i in 1:size(itexpr.ITs, 1)
            inner_c = theta[2i]
            outer_c = theta[2i+1]
            k       = itexpr.ITs[i].k
            g       = itexpr.ITs[i].g

            # TODO: fazer ele gerar esse mapa (dict) para não ficar
            # recriando aqui

            # obtendo a derivada da função de transformação
            gprime = x -> derivative(g, x)
            
            p_eval = vec(prod(X .^ k', dims=2))

            J[:, 2i]   = outer_c * gprime.(inner_c * p_eval) .* p_eval
            J[:, 2i+1] = g.(inner_c * p_eval)
        end
        
        J[:, end] .= 1.0
        
        J
    end

    model = (X, theta) -> begin
        #println("---")
        #println(theta)
        pred = zeros(size(X, 1))

        for i in 1:length(itexpr.ITs)
            it = itexpr.ITs[i]

            #println(theta[2i+1], theta[2i])
            #println(size(prod(X .^ it.k', dims=2)))
            #println(size(theta[2i] * prod(X .^ it.k', dims=2) ))
            #println(size(it.g.( theta[2i] * prod(X .^ it.k', dims=2) )))
            #println(size(theta[2i+1] * it.g.( theta[2i] * prod(X .^ it.k', dims=2) )))
            #println(size(pred))
            
            # TODO: definir função interação p(X, k) e diminuir isso no código
            pred[:] .+= theta[2i+1] * it.g.( vec(theta[2i] * prod(X .^ it.k', dims=2)) )
            
        end

        pred .+ theta[end]
    end

    # melhorar isso
    p0 = zeros(2*size(itexpr.ITs, 1)+1)
    for i in 1:size(itexpr.ITs, 1)
        p0[2i]   = itexpr.ITs[i].inner_c
        p0[2i+1] = itexpr.ITs[i].outer_c
    end
    p0[end] = itexpr.intercept

    #p0 = ones(2*size(itexpr.ITs, 1)+1)

    try
        fit = curve_fit(model, jacobian, X, Y, p0, maxIter=10)

        theta = fit.param
        #println(theta)
        
        return _replace_parameters(itexpr, theta)
    catch err
        if isa(err, DomainError)
            return itexpr
        else
            rethrow(err)
        end
    end
end

# TODO: trust region method

# TODO: gradient method