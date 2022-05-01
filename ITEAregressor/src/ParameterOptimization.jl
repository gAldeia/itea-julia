function _get_p0(itexpr::ITexpr; fit_intercept=true)

    p0 = if fit_intercept
        zeros(3*size(itexpr.ITs, 1)+1) # Starting point
    else
        zeros(3*size(itexpr.ITs, 1))
    end

    if fit_intercept
        p0[end] = itexpr.intercept
    end

    # setting p0 to zero for the b offset coefficients
    for i in 1:length(itexpr.ITs)
        it = itexpr.ITs[i]
        p0[3i-2:3i] .= it.w, it.b, it.c
    end

    p0
end

"""Function that takes an itexpr and creates a two argument model and jacobian.

The model, that was previously a single parameter function model(X), is now
a function model(X, theta), where theta is a vector of the free parameters.

The jacobian is also a two parameter function jacobian(X, theta), which will
return the jacobian function of the model w.r.t. the theta parameters.

Theta should be an array with the coefficients:
    w1, b1, c1, w2, b2, c2, ..., intercept
    (if fit_intercept is false, then it is not included in theta vector)
"""
function _create_model_and_jacobian(itexpr::ITexpr; fit_intercept=true)

    model = (X, theta) -> begin
        pred = zeros(size(X, 1))

        for i in 1:length(itexpr.ITs)
            w, b, c = theta[3i-2:3i]
            g, k = itexpr.ITs[i].g, itexpr.ITs[i].k
            
            pred[:] .+= w * g.(b .+ vec(c * prod(X .^ k', dims=2)) )
        end

        fit_intercept ? pred .+ theta[end] : pred
    end

    # obtendo a derivada da função de transformação
    gprimes = Dict(it.g => x -> derivative(it.g, x) for it in itexpr.ITs)

    jacobian = (X, theta) -> begin
        # Cada coluna é a derivada em função de um dos parâmetros de theta
        J = if fit_intercept
            # last column represents the derivative w.r.t. intercept
            ones(size(X, 1), 3*size(itexpr.ITs, 1)+1)
        else
            ones(size(X, 1), 3*size(itexpr.ITs, 1))
        end

        for i in 1:size(itexpr.ITs, 1)
            w, b, c = theta[3i-2:3i]
            g, k = itexpr.ITs[i].g, itexpr.ITs[i].k

            p_eval = vec(prod(X .^ k', dims=2))

            # Partial derivatives w.r.t. w, b and c.
            J[:, 3i-2] = g.(b .+ c * p_eval)
            J[:, 3i-1] = w * gprimes[g].(b .+ c * p_eval)
            J[:, 3i]   = w .* p_eval .* gprimes[g].(b .+ c * p_eval)
        end
        
        J
    end

    return model, jacobian, _get_p0(itexpr, fit_intercept=fit_intercept)
end


"""
se a itexpr tem n termos, então theta deve ter 3n + 1 elementos 
(um inner e outer para cada termo + intercepto, na ordem)
w1, b1, c1, w2, b2, c2, ..., intercept
"""
function _replace_parameters(
    itexpr::ITexpr, theta::Array{Float64, 1};
    fit_intercept=true) where {T<:Number}
    
    ITs = map(collect(1:size(itexpr.ITs, 1))) do i
        IT(itexpr.ITs[i].g, itexpr.ITs[i].k, theta[3i-2:3i]...)
    end

    fit_intercept ? ITexpr(ITs, theta[end]) : ITexpr(ITs)
end


# -----------------------------------------

"""
Ajusta apenas os parâmetros lineares (outer_c) de uma expressão ITexpr
"""
function ordinary_least_squares_adj(
    itexpr::ITexpr, X::Array{T, 2}, y::Array{T, 1};
    fit_intercept=true) where {T<:Number}

    nsamples, nvars = size(X)
    p0 = _get_p0(itexpr, fit_intercept=fit_intercept)

    try
        # ones no final para ser a coluna do fit intercepto
        Xhat = if fit_intercept
            # TODO: EVALUATE AQUI não deve considerar intercept ou coeficiente externo...
            hcat(map(it -> evaluate(it, X), itexpr.ITs)..., ones(nsamples))
        else
            hcat(map(it -> evaluate(it, X), itexpr.ITs)...)
        end
    
        # vec transforma (se já não for) em vetor coluna
        beta = pinv(Xhat' * Xhat) * Xhat' * vec(y)

        for i in 1:size(itexpr.ITs, 1)
            
            p0[3i-2] = beta[i] # setting outer scale coeff
        end

        # Dealing with the intercept
        if fit_intercept 
            p0[end] = beta[end]
        end

        return p0 #_replace_parameters(itexpr, theta, fit_intercept=fit_intercept)
    catch err 
        if isa(err, DomainError) || isa(err, LAPACKException)
            p0
        else
            rethrow(err)
        end
    end
end


function levenberg_marquardt_adj(
    itexpr::ITexpr, X::Array{T, 2}, y::Array{T, 1};
    maxIter=10, fit_intercept=true) where {T<:Number}

    residual_inplace! = (out, theta) -> begin
        #println("model\n ", F)
        eval = zeros(size(X, 1))

        for i in 1:length(itexpr.ITs)
            b, c = theta[2i-1:2i]
            w, g, k = itexpr.ITs[i].w, itexpr.ITs[i].g, itexpr.ITs[i].k
            
            p_eval = prod(X .^ k', dims=2)
            @. eval += w * g(b + c * p_eval)
        end

        out .= abs2.(y .- (fit_intercept ? eval .+ theta[end] : eval) )
    end

    p0 = if fit_intercept
        zeros(3*size(itexpr.ITs, 1)+1) # Starting point
    else
        zeros(3*size(itexpr.ITs, 1))
    end

    if fit_intercept
        p0[end] = itexpr.intercept
    end

    # Pegando os coeficientes individuais
    for i in 1:length(itexpr.ITs)
        p0[3i-2:3i] .= rand(Uniform(-100, 100), 3)
    end
    
    try
        fit = lso.optimize!(
            lso.LeastSquaresProblem(x = p0, f! = residual_inplace!, output_length = size(X, 1), autodiff = :central),
            lso.LevenbergMarquardt(lso.QR()))

        return p0
    catch err
        if isa(err, DomainError) || isa(err, SingularException)
            #rethrow(err)
            return p0
            #return _get_p0(itexpr, fit_intercept=fit_intercept)
        else
            #rethrow(err)
            return p0
            return _get_p0(itexpr, fit_intercept=fit_intercept)
        end
    end
end


function levenberg_marquardt_ordinary_least_squares_adj(
    itexpr::ITexpr, X::Array{T, 2}, y::Array{T, 1};
    iterations=10, fit_intercept=true) where {T<:Number}

    # ajustar cada termo individualmente (com intercepto), depois ajustar só 
    # os coeficientes com o LM
    p0 = ones(3*size(itexpr.ITs, 1)+(fit_intercept ? 1 : 0))

    if fit_intercept
        p0[end] = 0.0
    end

    for i in 1:length(itexpr.ITs)

        itexpr_aux = ITexpr([itexpr.ITs[i]])

        residual_inplace! = (out, theta) -> begin
            eval = zeros(size(X, 1))

            b, c = theta[1:2]
            w, g, k = itexpr_aux.ITs[1].w, itexpr_aux.ITs[1].g, itexpr_aux.ITs[1].k
            
            p_eval = prod(X .^ k', dims=2)
            @. eval += w * g(b + c * p_eval)

            # Residual sum
            out .= abs2.(y .- (fit_intercept ? eval .+ theta[end] : eval) )
        end

        try
            # b, c and local intercept
            p0_aux = rand(Uniform(-100, 100), 3)

            fit = lso.optimize!(
                lso.LeastSquaresProblem(x = p0_aux, f! = residual_inplace!,
                    output_length = size(X, 1), autodiff = :central),
                lso.LevenbergMarquardt(lso.QR()), iterations=iterations)

            # ajuste é feito inplace. Queremos os coeficientes sem o intercepto
            p0[3i-1:3i] .= p0_aux[1:2]
        catch err
            if isa(err, DomainError) || isa(err, SingularException)
                #rethrow(err)
                continue
            else
                #rethrow(err) #TODO: lançar o erro aqui
                continue
            end
        end
    end

    try
        itexpr = _replace_parameters(itexpr, p0, fit_intercept=fit_intercept)

        theta2 = ordinary_least_squares_adj(itexpr, X, y, fit_intercept=fit_intercept)

        return theta2
    catch err
        if isa(err, DomainError) || isa(err, SingularException)
            #rethrow(err)
            return _get_p0(itexpr, fit_intercept=fit_intercept)
        else
            #rethrow(err)
            return _get_p0(itexpr, fit_intercept=fit_intercept)
        end
    end
end


function ordinary_least_squares_levenberg_marquardt_adj(
    itexpr::ITexpr, X::Array{T, 2}, y::Array{T, 1};
    maxIter=10, fit_intercept=true) where {T<:Number}

    # https://discourse.julialang.org/t/optimization-and-arbitrary-precision/21801
    
    theta = ordinary_least_squares_adj(itexpr, X, y, fit_intercept=fit_intercept)

    # Atualizando com OLS
    itexpr = _replace_parameters(itexpr, theta, fit_intercept=fit_intercept)

    residual_inplace! = (out, theta) -> begin
        #println("model\n ", F)
        eval = zeros(size(X, 1))

        for i in 1:length(itexpr.ITs)
            b, c = theta[2i-1:2i]
            w, g, k = itexpr.ITs[i].w, itexpr.ITs[i].g, itexpr.ITs[i].k
            
            p_eval = prod(X .^ k', dims=2)
            @. eval += w * g(b + c * p_eval)
        end

        # Residual sum
        out .= abs2.(y .- (fit_intercept ? eval .+ theta[end] : eval) )
    end

    # TODO: melhorar esse try catch para funcionar caso um dos dois métodos ainda falhe
    try
        # Criando valores aleatórios para os outros coeficientes não ajustados
        for i in 1:length(itexpr.ITs)
            theta[3i-1:3i] .= rand(Uniform(-1.0, 1.0), 2)
        end
    
        fit = lso.optimize!(
            lso.LeastSquaresProblem(x = theta, f! = residual_inplace!, output_length = size(X, 1), autodiff = :central),
            lso.LevenbergMarquardt(lso.QR()))

        return theta 
    catch err
        if isa(err, DomainError) || isa(err, SingularException)
            #println(err)
            #rethrow(err)
            return theta
            #return _get_p0(itexpr, fit_intercept=fit_intercept)
        else
            #println(err)
            return theta
            #rethrow(err)
            #return _get_p0(itexpr, fit_intercept=fit_intercept)
        end
    end
end


# Métodos para usar (o resto não deve ser usado diretamente ) -----------------
const _adj_memoization = LRU{UInt64, Array{Float64, 1}}(maxsize = ITEA_CACHE_SIZE)


function free_params_adj(itexpr, X, y;
    method::Union{String, Nothing}="gradient_descent_adj", fit_intercept=true)

    # Removendo termos que dão nan/inf na predição
    remove = zeros(length(itexpr.ITs))
    for i in 1:length(itexpr.ITs)
        try
            evaluate(itexpr.ITs[i], X)
        catch err
            if isa(err, DomainError)
                remove[i] = 1.0
            else
                rethrow(err)
            end
        end
    end
    if all(remove .> 0)
        remove[rand(1:length(remove))] = 0.0
    end
    itexpr = ITexpr(itexpr.ITs[remove .== 0])

    if typeof(method) == Nothing # type of nothing is Nothing
        return itexpr
    else
        theta = get!(_adj_memoization, hash(itexpr)) do

            eval(Symbol(method))(itexpr, X, y, fit_intercept=fit_intercept)
        end

        expr = _replace_parameters(itexpr, theta, fit_intercept=fit_intercept)
        #println(expr)
        expr
    end
end


function clear_adj_memoize()

    empty!(_adj_memoization)
end