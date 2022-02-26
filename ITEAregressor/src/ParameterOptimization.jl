"""Function that takes an itexpr and creates a two argument model and jacobian.

The model, that was previously a single parameter function model(X), is now
a function model(X, theta), where theta is a vector of the free parameters.

The jacobian is also a two parameter function jacobian(X, theta), which will
return the jacobian function of the model w.r.t. the theta parameters.

Theta should be an array with the coefficients:
    inner1, outer1, inner2, outer2, ..., intercept
    (if fit_intercept is false, then it is not included in theta vector)
"""
function _create_model_and_jacobian(itexpr::ITexpr; fit_intercept=true)

    model = (X, theta) -> begin
        pred = zeros(size(X, 1))

        for i in 1:length(itexpr.ITs)
            it = itexpr.ITs[i]
            
            pred[:] .+= theta[2i-1] * it.g.( vec(theta[2i] * prod(X .^ it.k', dims=2)) )
        end

        fit_intercept ? pred .+ theta[end] : pred
    end

    # obtendo a derivada da função de transformação
    gprimes = Dict(it.g => x -> derivative(it.g, x) for it in itexpr.ITs)

    jacobian = (X, theta) -> begin
        # Cada coluna é a derivada em função de um dos parâmetros de theta
        J = if fit_intercept
            # last column represents the derivative w.r.t. intercept
            ones(size(X, 1), 2*size(itexpr.ITs, 1)+1)
        else
            ones(size(X, 1), 2*size(itexpr.ITs, 1))
        end

        for i in 1:size(itexpr.ITs, 1)
            inner_c, outer_c = theta[2i], theta[2i-1]

            g, k = itexpr.ITs[i].g, itexpr.ITs[i].k

            p_eval = vec(prod(X .^ k', dims=2))

            J[:, 2i]   = outer_c * gprimes[g].(inner_c * p_eval) .* p_eval
            J[:, 2i-1] = g.(inner_c * p_eval)
        end
        
        J
    end

    p0 = if fit_intercept
        ones(2*size(itexpr.ITs, 1)+1) # Starting point
    else
        ones(2*size(itexpr.ITs, 1))
    end

    return model, jacobian, p0
end


"""
se a itexpr tem n termos, então theta deve ter 2n + 1 elementos 
(um inner e outer para cada termo + intercepto, na ordem)
inner1, outer1, inner2, outer2, ..., intercept
"""
function _replace_parameters(
    itexpr::ITexpr, theta::Array{Float64, 1};
    fit_intercept=true) where {T<:Number}
    
    ITs = map(collect(1:size(itexpr.ITs, 1))) do i
        g = itexpr.ITs[i].g
        k = itexpr.ITs[i].k

        # Construtor: IT(g, k, inner_c, outer_c)
        IT(g, k, theta[2i], theta[2i-1])
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

    try
        # ones no final para ser a coluna do fit intercepto
        Xhat = if fit_intercept
            hcat(map(it -> evaluate(it, X), itexpr.ITs)..., ones(nsamples))
        else
            hcat(map(it -> evaluate(it, X), itexpr.ITs)...)
        end
    
        # vec transforma (se já não for) em vetor coluna
        beta = pinv(Xhat' * Xhat) * Xhat' * vec(y)

        theta = ones( 2*size(itexpr.ITs, 1) + (fit_intercept ? 1 : 0) )

        for i in 1:size(itexpr.ITs, 1)
            
            theta[2i-1] = beta[i] # getting the outer coefficients
        end

        # Dealing with the intercept
        if fit_intercept 
            theta[end] = beta[end]
        end

        return _replace_parameters(itexpr, theta, fit_intercept=fit_intercept)
    catch err 
        if isa(err, DomainError) || isa(err, LAPACKException)
            return itexpr
        else
            rethrow(err)
        end
    end
end


function levenberg_marquardt_adj(
    itexpr::ITexpr, X::Array{T, 2}, y::Array{T, 1};
    maxIter=10, fit_intercept=true) where {T<:Number}

    model, jacobian, p0 = _create_model_and_jacobian(itexpr, fit_intercept=fit_intercept)

    try
        fit = curve_fit(model, jacobian, X, y, p0, maxIter=maxIter)

        theta = fit.param
        
        return _replace_parameters(itexpr, theta, fit_intercept=fit_intercept)
    catch err
        if isa(err, DomainError) || isa(err, SingularException)
            return itexpr
        else
            rethrow(err)
        end
    end
end


function gradient_descent_adj(
    itexpr::ITexpr, X::Array{T, 2}, y::Array{T, 1};
    maxIter=100, alpha=0.01, tolerance=0.0002, fit_intercept=true) where {T<:Number}

    model, jacobian, p0 = _create_model_and_jacobian(itexpr, fit_intercept=fit_intercept)

    iteration = 0

    prev_avg_MAE = try
        mean(abs.(model(X, p0) - y))
    catch DomainError
        Inf
    end
    
    while iteration*size(X, 1) < maxIter

        for i in collect(1:size(X, 1))
            try
                # Erro é calculado individualmente para cada observação
                error = model(X[i:i, :], p0)[1] - y[i]
                gradient_wrt_coeffs = jacobian(X[i:i, :], p0)[1, :]
    
                # Vamos na direção oposta do gradiente
                p0 .-= (alpha .* gradient_wrt_coeffs .* error)
            catch err
                if isa(err, DomainError)
                    continue
                else
                    rethrow(err)
                end
            end
        end

        try
            new_avg_MAE = mean(abs.(model(X, p0) - y))

            if abs(new_avg_MAE-prev_avg_MAE) <= tolerance
                iteration = maxIter
            else
                prev_avg_MAE = new_avg_MAE
                iteration += 1
            end
        catch DomainError
            iteration += 1
        end
    end

    return _replace_parameters(itexpr, p0, fit_intercept=fit_intercept)
end


# Métodos para usar (o resto não deve ser usado diretamente ) -----------------
const _adj_memoization = LRU{ITexpr, ITexpr}(maxsize = ITEA_CACHE_SIZE)


function free_params_adj(itexpr, X, y;
    method::Union{String, Nothing}="gradient_descent_adj", fit_intercept=true)

    if typeof(method) == Nothing # type of nothing is Nothing
        return itexpr
    else
        get!(_adj_memoization, itexpr) do

            eval(Symbol(method))(itexpr, X, y, fit_intercept=fit_intercept)
        end
    end
end


function clear_adj_memoize()

    empty!(_adj_memoization)
end